//! # **Buffer** — *Unified owned/shared data storage*
//!
//! Buffer backs most inner Array types in *Minarrow* (`IntegerArray`, `FloatArray`, `CategoricalArray`, `StringArray`, `DatetimeArray`).
//!
//! # Design
//! `Buffer<T>` abstracts over two storage backends:
//! - **Owned**: [`Vec64<T>`] — an internally aligned, 64-byte, heap-allocated vector optimised
//!   for SIMD kernels.
//! - **Shared**: [`SharedBuffer`] — a zero-copy, read-only window into externally owned memory
//!   (e.g. memory-mapped files, network streams).
//! - Whilst it would be simpler to avoid an extra abstraction *(putting `Vec64` directly in the above types)*,
//! that would mean data would be copied from those external sources breaking zero-copy guarantees.
//! Hence, we make it invisible for all common use cases through the implementation of deref, indexing,
//! and other common traits + fns.
//!
//! ## Purpose
//! - Provide `Vec`-like ergonomics and performance for owned data.
//! - Allow *in-place* processing of externally sourced buffers without copy overhead,
//!   subject to alignment constraints.
//! - Enforce copy-on-write for any mutation of shared data for safety-first.
//! - It's primarily for internal use, but advanced users may want to use it directly.
//!
//! ## Behaviour
//! - **Read-only ops** (`&[T]` slicing, iteration, parallel reads) operate directly on the
//!   backing memory regardless of ownership.
//! - **Mutating ops** (push, splice, clear, etc.) transparently convert shared buffers into
//!   owned `Vec64<T>` before modifying.
//! - All owned buffers are guaranteed 64-byte aligned for predictable SIMD performance.
//! - Shared buffers *may* be 64-byte aligned — if not, they are cloned into an aligned `Vec64<T>`
//!   on ingestion.
//!
//! ## Alignment rules
//! - `Vec64<T>` is always 64-byte aligned.
//! - Shared buffers must be validated by the caller for both type and SIMD alignment. The
//!   `from_shared` and `from_shared_raw` constructors enforce alignment checks and perform
//!   cloning if needed.
//! - The Arrow spec allows both 8-byte and 64-byte alignment; Minarrow prefers 64-byte for
//!   optimal SIMD kernels.
//!
//! ## Safety notes
//! - When using `from_shared_raw`, the pointer must be valid, correctly aligned for `T`,
//!   and must lie entirely within the lifetime of the provided `Arc<[u8]>`.
//! - For externally supplied memory, alignment should be ensured at source — *Minarrow*’s
//!   IO paths (e.g. via *Lightstream-IO*) guarantee it.
//!
//! ## Typical use
//! ```rust
//! use minarrow::{Buffer, vec64};
//!
//! // Owned buffer
//! let mut b = Buffer::from(vec64![1u32, 2, 3]);
//! b.push(4);
//! assert_eq!(b.as_slice(), &[1, 2, 3, 4]);
//!
//! // Shared buffer (read-only until mutation)
//! use std::sync::Arc;
//! let raw = vec64![10u8, 20, 30];
//! let arc: Arc<[u8]> = Arc::from(&raw[..]);
//! let shared = unsafe { Buffer::from_shared_raw(arc.clone(), arc.as_ptr(), arc.len()) };
//! assert_eq!(shared[1], 20);
//! ```
//!
//! This type is Send + Sync (subject to `T`) and implements most of the `Vec` and slice
//! interfaces for smooth interoperability.

use std::fmt::{Display, Formatter};
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut, RangeBounds};
use std::{fmt, mem};

use crate::Vec64;
use crate::structs::shared_buffer::SharedBuffer;
use crate::traits::print::MAX_PREVIEW;

/// # Buffer
///
/// Data buffer abstraction that blends the standard 64-byte aligned Vec data buffer,
/// with an externally backed and borrowed source such as memory-mapped files
/// or network streams.
///
/// This includes external:
/// 1. **Filesystem IO** (e.g., memory-mapped IPC files, datasets on disk)
/// 2. **Network IO** (e.g., `WebTransport`, `Websockets`, `gRPC` *(without Protobuf)*, etc.)
///
/// ## Purpose
/// At the cost of a layer of abstraction, it enables working with external data
/// in-place and without additional copy overhead, directly at the source.
///
/// We eliminate as much indirection as possible for the owned case (`Vec64<T>`),
/// so typical workloads remain clean and fast.
///
/// ### Behaviour:
/// - **Semantically equivalent to `Vec64<T>`** in most contexts, but may be backed by shared memory.
/// - **Read-only operations** (e.g., `&[T]` slices, SIMD kernels, iteration) work directly on the buffer
///   zero-copy, regardless of ownership.
/// - **Mutation operations** always copy the shared buffer into an owned `Vec64<T>` on first write (**even
/// in cases where there is Arc uniqueness**), for safety, as we cannot guarantee control of the source.
/// - For **owned buffers**, `Deref` and method forwarding make this behave exactly like `Vec64<T>`.
/// - The only divergence is in struct initialisers, where `.into()` is required when populating fields
///   like `IntegerArray<T>`, `FloatArray<T>`, etc. This is the unfortunate trade-off.
///
/// ### Safety:
/// - The caller is responsible for ensuring that the `Shared` buffer is also 64-byte aligned in any context
/// where it matters, such as any SIMD kernels that rely on it.
/// - In `Minarrow`, through the standard library paths that construct `Shared` buffers, for e.g., `TableStreamReader`,
/// we check for this alignment upfront and the path is faster if it is pre-aligned. `TableStreamWriter` writes it
/// aligned by default, but IPC writers from other crates *(e.g., `Arrow-Rs`, `Arrow2` etc., at the time of writing)*
/// use 8-byte alignment, and may for e.g., check at kernel run-time.
/// - The Arrow specification confirms both are valid, with 64-byte being the optimal format for SIMD.
pub struct Buffer<T> {
    storage: Storage<T>,
}

/// Internal memory ownership tracking store
/// for `Buffer`
enum Storage<T> {
    Owned(Vec64<T>),
    Shared {
        owner: SharedBuffer,
        offset: usize, // element index (not bytes)
        len: usize,    // element count
    },
}

impl<T: Clone> Buffer<T> {
    /// Construct an owned buffer from a slice, copying the data into an aligned Vec64.
    #[inline]
    pub fn from_slice(slice: &[T]) -> Self {
        let mut v = Vec64::with_capacity(slice.len());
        v.extend_from_slice(slice);
        Buffer::from_vec64(v)
    }
}

impl<T> Buffer<T> {
    /// Construct from an owned Vec64<T>.
    #[inline]
    pub fn from_vec64(v: Vec64<T>) -> Self {
        Self {
            storage: Storage::Owned(v),
        }
    }

    /// Construct a buffer as a view over a SharedBuffer (zero-copy, read-only).
    /// Caller must ensure [u8] slice is valid and aligned for T.
    ///
    /// # Behaviour
    /// - non-aligned copies into a fresh vec64
    /// - This is true even for memory mapped files, and is a notable trade-off, which can be avoided by
    /// using *Minarrow*s IPC writer from the sibling *Lightstream-IO* crate.
    #[inline]
    pub fn from_shared(owner: SharedBuffer) -> Self {
        let bytes = owner.as_slice();
        let size_of_t = std::mem::size_of::<T>();
        let ptr_usize = bytes.as_ptr() as usize;
        let align = std::mem::align_of::<T>();
        let needs_alignment = ptr_usize % 64 != 0;
        let correct_type_align = ptr_usize % align == 0;

        if needs_alignment {
            eprintln!(
                "Buffer::from_shared: underlying SharedBuffer {:p} not 64-byte aligned, cloning to owned Vec64<T>.",
                bytes.as_ptr()
            );
            assert_eq!(
                ptr_usize % align,
                0,
                "Underlying SharedBuffer is not properly aligned for T"
            );
            assert_eq!(
                bytes.len() % size_of_t,
                0,
                "Underlying SharedBuffer is not a valid T slice"
            );
            let len = bytes.len() / size_of_t;
            let mut v = Vec64::with_capacity(len);
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const T, v.as_mut_ptr(), len);
                v.set_len(len);
            }
            return Buffer::from_vec64(v);
        }

        assert!(
            correct_type_align,
            "Underlying SharedBuffer is not properly aligned for T"
        );
        assert_eq!(
            bytes.len() % size_of_t,
            0,
            "Underlying SharedBuffer is not a valid T slice"
        );
        let len = bytes.len() / size_of_t;
        Self {
            storage: Storage::Shared {
                owner,
                offset: 0,
                len,
            },
        }
    }

    /// Construct a zero-copy buffer from an Arc-backed foreign allocation.
    ///
    /// Because all `Minarrow` types work off 64-byte alignment at the outset
    /// for SIMD compatibility *(streamlining downstream management and kernel
    /// usage)*, we establish whether there is alignment during the creation
    /// process here. If an external buffer *(including network bytes, etc.)* is
    /// 64-byte aligned, it becomes a `SharedBuffer` here, where zero-copy
    /// slicing is available. However, if the data is not aligned, it raises
    /// a message and copies the data into a Vec64 aligned buffer.
    ///
    /// We provide network ready data transfer and IO that guarantees this
    /// through the *Lightstream-IO* crate, if you don't want to manage this yourself.
    ///
    /// # Safety
    /// - ptr must be valid, readable for len T elements
    /// - Must point *within* the Arc (owner) buffer
    /// - Alignment is caller's responsibility
    #[inline]
    pub unsafe fn from_shared_raw(arc: std::sync::Arc<[u8]>, ptr: *const T, len: usize) -> Self {
        assert!(!ptr.is_null());
        let align = std::mem::align_of::<T>();
        let ptr_usize = ptr as usize;

        // 64-byte alignment check for SIMD
        let needs_alignment = ptr_usize % 64 != 0;
        let correct_type_align = ptr_usize % align == 0;

        if !correct_type_align {
            panic!(
                "Buffer::from_shared_raw: pointer {ptr:p} is not aligned to {} bytes",
                align
            );
        }

        if needs_alignment {
            eprintln!(
                "Buffer::from_shared_raw: pointer {ptr:p} is not 64-byte aligned, cloning to owned Vec64<T>."
            );
            // Defensive fallback: copy to a properly aligned Vec64<T>
            let mut v = Vec64::with_capacity(len);
            unsafe { std::ptr::copy_nonoverlapping(ptr, v.as_mut_ptr(), len) };
            unsafe { v.set_len(len) };
            return Buffer::from_vec64(v);
        }

        // Wrap the Arc<[u8]> in a SharedBuffer
        let shared = SharedBuffer::from_owner(arc);

        // Compute the byte‑offset into that shared slice
        let base = shared.as_slice().as_ptr() as usize;
        let p = ptr_usize;
        let byte_offset = p
            .checked_sub(base)
            .expect("Buffer::from_shared_raw: pointer not in Arc<[u8]> region");

        // Now slice out exactly `len` T‑elements
        let byte_len = len * std::mem::size_of::<T>();
        let owner_slice = shared.slice(byte_offset..byte_offset + byte_len);

        Self {
            storage: Storage::Shared {
                owner: owner_slice,
                offset: 0,
                len,
            },
        }
    }

    /// Returns the buffer as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        match &self.storage {
            Storage::Owned(vec) => vec.as_slice(),
            Storage::Shared { owner, offset, len } => {
                let bytes = owner.as_slice();
                let size_of_t = std::mem::size_of::<T>();
                let ptr = unsafe { bytes.as_ptr().add(offset * size_of_t) };
                unsafe { std::slice::from_raw_parts(ptr as *const T, *len) }
            }
        }
    }

    /// Returns a mutable slice; will copy on write if buffer is shared.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.make_owned_mut().as_mut_slice()
    }

    /// Returns the number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Returns true if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn push(&mut self, v: T) {
        self.make_owned_mut().push(v);
    }

    #[inline]
    pub fn clear(&mut self) {
        self.make_owned_mut().clear();
    }

    #[inline]
    pub fn reserve(&mut self, addl: usize) {
        self.make_owned_mut().reserve(addl);
    }

    /// Returns the capacity in elements.
    #[inline]
    pub fn capacity(&self) -> usize {
        match &self.storage {
            Storage::Owned(vec) => vec.capacity(),
            Storage::Shared {
                owner: _,
                offset: _,
                len,
            } => {
                // Only the viewed slice is available, no reserve
                *len
            }
        }
    }
    /// Ensure owned and return &mut Vec64<T>.
    #[inline]
    fn make_owned_mut(&mut self) -> &mut Vec64<T> {
        // Already owned
        if let Storage::Owned(ref mut vec) = self.storage {
            return vec;
        }

        // We know it's Shared, so take it out by replacing with a dummy Owned.
        // This doesn't borrow `self.storage` across the replacement.
        let (owner, offset, len) =
            match mem::replace(&mut self.storage, Storage::Owned(Vec64::with_capacity(0))) {
                Storage::Shared { owner, offset, len } => (owner, offset, len),
                _ => unreachable!(),
            };

        // Build a new Vec64<T> from the shared slice
        let bytes = owner.as_slice();
        let size_of_t = std::mem::size_of::<T>();
        let ptr = unsafe { bytes.as_ptr().add(offset * size_of_t) };
        let mut new_vec = Vec64::with_capacity(len);
        unsafe {
            std::ptr::copy_nonoverlapping(ptr as *const T, new_vec.as_mut_ptr(), len);
            new_vec.set_len(len);
        }

        // Store it back as Owned
        self.storage = Storage::Owned(new_vec);

        // Now that storage is Owned, we can safely get a mutable reference
        if let Storage::Owned(ref mut vec) = self.storage {
            vec
        } else {
            unreachable!()
        }
    }

    /// Identical semantics to `Vec::splice`.
    ///
    /// If the buffer is a shared view we copy to a `Vec64<T>`
    /// and then delegate to `Vec64::splice`.
    #[inline]
    pub fn splice<'a, R, I>(&'a mut self, range: R, replace_with: I) -> impl Iterator<Item = T> + 'a
    where
        R: RangeBounds<usize>,
        I: IntoIterator<Item = T> + 'a,
        I::IntoIter: 'a,
    {
        let vec = self.make_owned_mut();
        vec.splice(range, replace_with)
    }

    /// Returns true if the buffer is a shared (zero-copy, externally owned) region.
    #[inline]
    pub fn is_shared(&self) -> bool {
        matches!(self.storage, Storage::Shared { .. })
    }

    /// Creates an owned copy of the buffer data.
    /// If the buffer is already owned, this clones the data.
    /// If the buffer is shared, this copies the data into a new owned Vec64.
    #[inline]
    pub fn to_owned_copy(&self) -> Self
    where
        T: Clone,
    {
        // Always create a fresh owned copy
        let vec: Vec64<T> = self.as_ref().iter().cloned().collect();
        Buffer::from_vec64(vec)
    }
}

impl<T: Clone> Buffer<T> {
    #[inline]
    pub fn resize(&mut self, new_len: usize, value: T) {
        self.make_owned_mut().resize(new_len, value);
    }

    #[inline]
    pub fn extend_from_slice(&mut self, s: &[T]) {
        self.make_owned_mut().extend_from_slice(s);
    }
}

#[inline]
pub fn split_at_first_align64(ptr: *const u8, len_bytes: usize) -> Option<(usize, usize)> {
    // returns (head_len, tail_len) *in bytes*, or None if already aligned
    let addr = ptr as usize;
    let misalign = addr & 63;
    if misalign == 0 {
        return None;
    }
    let head = 64 - misalign;
    if head >= len_bytes {
        // whole slice fits before next boundary → cannot split
        return None;
    }
    Some((head, len_bytes - head))
}

impl<T: Copy> Buffer<T> {
    /// Shorten this buffer to at most `new_len` elements.
    ///
    /// - If it's `Owned`, calls `Vec64::truncate`.
    /// - If it's `Shared`, zero-copy SharedBuffer view.
    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        if new_len >= self.len() {
            return;
        }
        match &mut self.storage {
            Storage::Owned(vec) => vec.truncate(new_len),
            Storage::Shared { offset: _, len, .. } => {
                *len = new_len;
            }
        }
    }
}

impl<T: Clone> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        match &self.storage {
            Storage::Owned(vec) => Buffer::from_vec64(vec.clone()),
            Storage::Shared { owner, offset, len } => Buffer {
                storage: Storage::Shared {
                    owner: owner.clone(),
                    offset: *offset,
                    len: *len,
                },
            },
        }
    }
}

impl<T: PartialEq> PartialEq for Buffer<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.deref() == other.deref()
    }
}

impl<T: PartialEq> PartialEq<Vec64<T>> for Buffer<T> {
    #[inline]
    fn eq(&self, other: &Vec64<T>) -> bool {
        self.deref() == other.deref()
    }
}

impl<T: fmt::Debug> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Buffer").field(&self.as_slice()).finish()
    }
}

impl<T> Default for Buffer<T> {
    #[inline]
    fn default() -> Self {
        Buffer::from_vec64(Vec64::default())
    }
}

impl<T> Deref for Buffer<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for Buffer<T> {
    // #[inline]
    // fn deref_mut(&mut self) -> &mut [T] { self.make_owned_mut() }

    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        match &self.storage {
            Storage::Owned(_) => self.make_owned_mut(),
            Storage::Shared { .. } => {
                // indexing via `&mut buf[0]` still panics
                panic!("Cannot mutably deref a shared buffer")
            }
        }
    }
}

impl<T> From<Vec64<T>> for Buffer<T> {
    #[inline]
    fn from(v: Vec64<T>) -> Self {
        Buffer::from_vec64(v)
    }
}

// Shared immutable case:
impl<'a, T> IntoIterator for &'a Buffer<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.deref().iter()
    }
}

// Mutable data
impl<'a, T> IntoIterator for &'a mut Buffer<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Ensure we own it, then hand back its slice iterator:
        let vec64 = self.make_owned_mut();
        vec64.iter_mut()
    }
}

impl<T> FromIterator<T> for Buffer<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from(Vec64::from_iter(iter))
    }
}

/// Consuming iterator – needed for `a.iter().zip(b)`, `collect()`, etc.
impl<T> IntoIterator for Buffer<T> {
    type Item = T;
    type IntoIter = <Vec64<T> as IntoIterator>::IntoIter;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        // Prevent double drop
        let mut this = ManuallyDrop::new(self);

        unsafe {
            match &mut this.storage {
                Storage::Owned(vec) => {
                    // Move out the Vec64<T>
                    std::ptr::read(vec).into_iter()
                }
                Storage::Shared { owner, offset, len } => {
                    let bytes = owner.as_slice();
                    let size_of_t = std::mem::size_of::<T>();
                    let ptr = bytes.as_ptr().add(*offset * size_of_t);
                    let mut v = Vec64::with_capacity(*len);
                    std::ptr::copy_nonoverlapping(ptr as *const T, v.as_mut_ptr(), *len);
                    v.set_len(*len);
                    v.into_iter()
                }
            }
        }
    }
}

impl<T> AsRef<[T]> for Buffer<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.deref()
    }
}

impl<T> AsMut<[T]> for Buffer<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.deref_mut()
    }
}

#[cfg(feature = "parallel_proc")]
impl<T: Send + Sync> Buffer<T> {
    
    #[inline]
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, T> {
        use rayon::iter::IntoParallelRefIterator;
        self.as_slice().par_iter()
    }

    #[inline]
    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, T> {
        use rayon::iter::IntoParallelRefMutIterator;
        self.make_owned_mut().par_iter_mut()
    }
}

impl<T: Display> Display for Buffer<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let kind = if self.is_shared() { "shared" } else { "owned" };
        let len = self.len();

        writeln!(f, "Buffer [{} elements] ({})", len, kind)?;

        write!(f, "[")?;

        for i in 0..usize::min(len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self[i])?;
        }

        if len > MAX_PREVIEW {
            write!(f, ", … ({} total)", len)?;
        }

        write!(f, "]")
    }
}

// SAFETY: Shared buffers are read-only and `Arc` ensures memory is valid.
// `Owned` is already `Send + Sync` via `Vec64<T>`.
unsafe impl<T: Sync> Sync for Buffer<T> {}
unsafe impl<T: Send> Send for Buffer<T> {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::vec64;

    #[test]
    fn test_owned_buffer() {
        let mut buf = Buffer::from(Vec64::from(vec![1, 2, 3]));
        assert_eq!(buf.len(), 3);
        assert_eq!(&buf[..], &[1, 2, 3]);
        buf.push(4);
        assert_eq!(&buf[..], &[1, 2, 3, 4]);
    }

    #[test]
    fn test_shared_buffer_read() {
        let data = vec64![1u8, 2, 3, 4];
        let arc: Arc<[u8]> = Arc::from(&data[..]);
        let arc_ptr = arc.as_ptr();
        let buf = unsafe { Buffer::from_shared_raw(arc.clone(), arc_ptr, arc.len()) };
        assert_eq!(buf.len(), 4);
        assert_eq!(&buf[..], &[1, 2, 3, 4]);
    }

    #[test]
    fn test_shared_buffer_cow() {
        let data = vec64![1u8, 2, 3];
        let arc: Arc<[u8]> = Arc::from(&data[..]);
        let arc_ptr = arc.as_ptr();
        let mut buf = unsafe { Buffer::from_shared_raw(arc.clone(), arc_ptr, arc.len()) };
        buf.push(4);
        assert_eq!(&buf[..], &[1, 2, 3, 4]);
    }

    #[test]
    fn test_clone_owned() {
        let buf1 = Buffer::from(vec64![1, 2, 3, 4]);
        let buf2 = buf1.clone();
        assert_eq!(buf1, buf2);
        assert_eq!(buf1.len(), buf2.len());
        assert_eq!(&buf1[..], &buf2[..]);
    }

    #[test]
    fn test_clone_shared() {
        let data = vec64![5u8, 6, 7, 8];
        let arc: Arc<[u8]> = Arc::from(&data[..]);
        let arc_ptr = arc.as_ptr();
        let buf1 = unsafe { Buffer::from_shared_raw(arc.clone(), arc_ptr, arc.len()) };
        let buf2 = buf1.clone();
        assert_eq!(buf1, buf2);
        assert_eq!(&buf1[..], &[5, 6, 7, 8]);
        assert_eq!(&buf2[..], &[5, 6, 7, 8]);
    }

    #[test]
    fn test_truncate_owned() {
        let mut buf = Buffer::from(vec64![1, 2, 3, 4, 5]);
        buf.truncate(3);
        assert_eq!(buf.len(), 3);
        assert_eq!(&buf[..], &[1, 2, 3]);
        buf.truncate(10);
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn test_truncate_shared() {
        let data = vec64![1u8, 2, 3, 4, 5];
        let arc: Arc<[u8]> = Arc::from(&data[..]);
        let arc_ptr = arc.as_ptr();
        let mut buf = unsafe { Buffer::from_shared_raw(arc.clone(), arc_ptr, arc.len()) };
        buf.truncate(3);
        assert_eq!(buf.len(), 3);
        assert_eq!(&buf[..], &[1, 2, 3]);
    }

    #[test]
    fn test_clear() {
        let mut buf = Buffer::from(vec64![1, 2, 3]);
        buf.clear();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_clear_shared_makes_owned() {
        let data = vec64![1u8, 2, 3];
        let arc: Arc<[u8]> = Arc::from(&data[..]);
        let arc_ptr = arc.as_ptr();
        let mut buf = unsafe { Buffer::from_shared_raw(arc.clone(), arc_ptr, arc.len()) };
        buf.clear();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        buf.push(10);
        assert_eq!(&buf[..], &[10]);
    }

    #[test]
    fn test_reserve() {
        let mut buf = Buffer::from(vec64![1, 2, 3]);
        let initial_cap = buf.capacity();
        buf.reserve(100);
        assert!(buf.capacity() >= initial_cap + 100);
    }

    #[test]
    fn test_capacity_shared() {
        let data = vec64![1u8, 2, 3, 4];
        let arc: Arc<[u8]> = Arc::from(&data[..]);
        let arc_ptr = arc.as_ptr();
        let buf = unsafe { Buffer::from_shared_raw(arc.clone(), arc_ptr, arc.len()) };
        assert_eq!(buf.capacity(), 4);
    }

    #[test]
    fn test_resize() {
        let mut buf = Buffer::from(vec64![1, 2, 3]);
        buf.resize(5, 99);
        assert_eq!(&buf[..], &[1, 2, 3, 99, 99]);
        buf.resize(2, 0);
        assert_eq!(&buf[..], &[1, 2]);
    }

    #[test]
    fn test_extend_from_slice() {
        let mut buf = Buffer::from(vec64![1, 2]);
        buf.extend_from_slice(&[3, 4, 5]);
        assert_eq!(&buf[..], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_splice() {
        let mut buf = Buffer::from(vec64![1, 2, 3, 4, 5]);
        let removed: Vec<_> = buf.splice(1..4, vec![10, 20]).collect();
        assert_eq!(removed, vec![2, 3, 4]);
        assert_eq!(&buf[..], &[1, 10, 20, 5]);
    }

    #[test]
    fn test_splice_shared_makes_owned() {
        let data = vec64![1u8, 2, 3, 4, 5];
        let arc: Arc<[u8]> = Arc::from(&data[..]);
        let arc_ptr = arc.as_ptr();
        let mut buf = unsafe { Buffer::from_shared_raw(arc.clone(), arc_ptr, arc.len()) };
        let removed: Vec<_> = buf.splice(2..4, vec![10]).collect();
        assert_eq!(removed, vec![3, 4]);
        assert_eq!(&buf[..], &[1, 2, 10, 5]);
    }

    #[test]
    fn test_as_slice_and_as_mut_slice() {
        let mut buf = Buffer::from(vec64![1, 2, 3]);
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
        buf.as_mut_slice()[1] = 20;
        assert_eq!(buf.as_slice(), &[1, 20, 3]);
    }

    #[test]
    fn test_as_mut_slice_shared_makes_owned() {
        let data = vec64![1u8, 2, 3];
        let arc: Arc<[u8]> = Arc::from(&data[..]);
        let arc_ptr = arc.as_ptr();
        let mut buf = unsafe { Buffer::from_shared_raw(arc.clone(), arc_ptr, arc.len()) };
        let slice = buf.as_mut_slice();
        slice[0] = 10;
        assert_eq!(&buf[..], &[10, 2, 3]);
    }

    #[test]
    fn test_equality() {
        let buf1 = Buffer::from(vec64![1, 2, 3]);
        let buf2 = Buffer::from(vec64![1, 2, 3]);
        let buf3 = Buffer::from(vec64![1, 2, 4]);
        assert_eq!(buf1, buf2);
        assert_ne!(buf1, buf3);
    }

    #[test]
    fn test_equality_with_vec64() {
        let buf = Buffer::from(vec64![1, 2, 3]);
        let vec = vec64![1, 2, 3];
        assert_eq!(buf, vec);
    }

    #[test]
    fn test_default() {
        let buf: Buffer<i32> = Buffer::default();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_from_iter() {
        let buf: Buffer<i32> = (1..=5).collect();
        assert_eq!(&buf[..], &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_into_iter_owned() {
        let buf = Buffer::from(vec64![1, 2, 3, 4]);
        let vec: Vec<_> = buf.into_iter().collect();
        assert_eq!(vec, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_into_iter_shared() {
        let data = vec64![1u8, 2, 3, 4];
        let arc: Arc<[u8]> = Arc::from(&data[..]);
        let arc_ptr = arc.as_ptr();
        let buf = unsafe { Buffer::from_shared_raw(arc.clone(), arc_ptr, arc.len()) };
        let vec: Vec<_> = buf.into_iter().collect();
        assert_eq!(vec, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_ref_iter() {
        let buf = Buffer::from(vec64![1, 2, 3]);
        let sum: i32 = buf.iter().sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_mut_iter() {
        let mut buf = Buffer::from(vec64![1, 2, 3]);
        for x in &mut buf {
            *x *= 2;
        }
        assert_eq!(&buf[..], &[2, 4, 6]);
    }

    #[test]
    fn test_mut_iter_shared_makes_owned() {
        let data = vec64![1u8, 2, 3];
        let arc: Arc<[u8]> = Arc::from(&data[..]);
        let arc_ptr = arc.as_ptr();
        let mut buf = unsafe { Buffer::from_shared_raw(arc.clone(), arc_ptr, arc.len()) };
        for x in &mut buf {
            *x *= 2;
        }
        assert_eq!(&buf[..], &[2, 4, 6]);
    }

    #[test]
    fn test_as_ref_as_mut() {
        let mut buf = Buffer::from(vec64![1, 2, 3]);
        let slice_ref: &[i32] = buf.as_ref();
        assert_eq!(slice_ref, &[1, 2, 3]);
        let slice_mut: &mut [i32] = buf.as_mut();
        slice_mut[0] = 10;
        assert_eq!(&buf[..], &[10, 2, 3]);
    }

    // #[test]
    // #[should_panic(expected = "Cannot mutably deref a shared buffer")]
    // fn test_deref_mut_shared_panics() {
    //     let data = vec64![1u8, 2, 3];
    //     let arc: Arc<[u8]> = Arc::from(&data[..]);
    //     let arc_ptr = arc.as_ptr();
    //     let buffer = SharedBuffer::from_vec64(data);
    //     let mut buf = unsafe { Buffer::from_shared(buffer) };
    //     // This should panic
    //     let _ = &mut buf[0];
    // }

    #[test]
    fn test_empty_buffer() {
        let buf: Buffer<i32> = Buffer::from(vec64![]);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_multiple_shared_views() {
        let mut data = vec64![0u8; 128];
        for i in 0..128 {
            data[i] = i as u8;
        }
        let arc: Arc<[u8]> = Arc::from(&data[..]);
        let ptr1 = arc.as_ptr(); // offset 0
        let ptr2 = unsafe { arc.as_ptr().add(64) }; // offset 64
        let buf1 = unsafe { Buffer::from_shared_raw(arc.clone(), ptr1, 64) };
        let buf2 = unsafe { Buffer::from_shared_raw(arc.clone(), ptr2, 64) };
        assert_eq!(&buf1[0..4], &[0, 1, 2, 3]);
        assert_eq!(&buf2[0..4], &[64, 65, 66, 67]);
        assert_eq!(buf1.len(), 64);
        assert_eq!(buf2.len(), 64);
        assert_eq!(buf1[63], 63);
        assert_eq!(buf2[63], 127);
    }

    #[test]
    fn test_debug_format() {
        let owned = Buffer::from(vec64![1, 2, 3]);
        assert!(format!("{:?}", owned).contains("Buffer"));
        let data = vec64![1u8, 2, 3];
        let arc: Arc<[u8]> = Arc::from(&data[..]);
        let arc_ptr = arc.as_ptr();
        let shared = unsafe { Buffer::from_shared_raw(arc.clone(), arc_ptr, arc.len()) };
        assert!(format!("{:?}", shared).contains("Buffer"));
    }

    #[cfg(feature = "parallel_proc")]
    #[test]
    fn test_par_iter() {
        use rayon::prelude::*;
        let buf = Buffer::from(vec64![1, 2, 3, 4, 5]);
        let sum: i32 = buf.par_iter().sum();
        assert_eq!(sum, 15);
    }

    #[cfg(feature = "parallel_proc")]
    #[test]
    fn test_par_iter_mut() {
        use rayon::prelude::*;
        let mut buf = Buffer::from(vec64![1, 2, 3, 4, 5]);
        buf.par_iter_mut().for_each(|x| *x *= 2);
        assert_eq!(&buf[..], &[2, 4, 6, 8, 10]);
    }

    #[cfg(feature = "parallel_proc")]
    #[test]
    fn test_par_iter_mut_shared_makes_owned() {
        use rayon::prelude::*;
        let data = vec64![1u8, 2, 3, 4, 5];
        let arc: Arc<[u8]> = Arc::from(&data[..]);
        let arc_ptr = arc.as_ptr();
        let mut buf = unsafe { Buffer::from_shared_raw(arc.clone(), arc_ptr, arc.len()) };
        buf.par_iter_mut().for_each(|x| *x *= 2);
        assert_eq!(&buf[..], &[2, 4, 6, 8, 10]);
    }
}
