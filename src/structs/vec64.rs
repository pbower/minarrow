//! # Vec64
//!
//! 64-byte aligned vector type backed by a custom allocator (`Alloc64`).
//!
//! Provides the same API as `Vec`, but guarantees the starting address
//! of the allocation is 64-byte aligned for SIMD, cache line, and
//! low-level hardware optimisations.
//!
//! Used internally for all data buffers in Minarrow, 
//! whilst remaining interoperable with standard `Vec` where required.

use std::borrow::{Borrow, BorrowMut};
use std::fmt::{Debug, Display, Formatter, Result};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::slice::{Iter, IterMut};
use std::vec::Vec;

#[cfg(feature = "parallel_proc")]
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator};

use crate::structs::allocator::Alloc64;
use crate::Buffer;

/// # Vec64
/// 
/// High-performance 64-byte aligned vector.
///
/// ## Purpose
/// A drop-in replacement for `Vec` that ensures the starting pointer is aligned to a
/// 64-byte boundary via a custom `Alloc64` allocator. This predominantly ensures
/// compatibility with SIMD processing instruction extensions such as the AVX-512.
/// These increase CPU throughput when using SIMD-friendly code like `std::simd`, or hand-rolled intrinsics.
/// 
/// Alignment can help avoid split loads/stores across cache lines and make hardware
/// prefetch more predictable during sequential scans. However, gains are workload- and
/// platform-dependent, and the Rust compiler may generate equally efficient code for
/// ordinary `Vec` in some cases.
/// 
/// ## Behaviour – Padding
/// This type does not add any padding to your data. Only the first element of the
/// allocation is guaranteed to be aligned. If you construct a buffer that mixes headers,
/// metadata, and then Arrow data pages, and you plan to extract or process the Arrow
/// portion with `Vec64::from_raw_parts` or SIMD at its offset, you must insert your own
/// zero-byte padding so that the Arrow section’s start falls on a 64-byte boundary.
/// Without that manual padding, the middle of the buffer will not be aligned and
/// unaligned access or unsafe reconstitution may fail or force a reallocation.
///
/// All library code in `Minarrow` and related kernel code automatically handles such
/// padding, and therefore this is only relevant if you leverage `Vec64` manually.
///
/// ## Notes
/// - All `Vec` APIs remain available—`Vec64` is a tuple wrapper over `Vec<T, Alloc64>`.
/// - When passing to APIs expecting a `Vec`, use `.0` to extract the inner `Vec`.
/// - Avoid mixing `Vec` and `Vec64` unless both use the same custom allocator (`Alloc64`).
/// - Alignment helps with contiguous, stride-friendly access; it does not improve
///   temporal locality or benefit random-access patterns.
#[repr(transparent)]
pub struct Vec64<T>(pub Vec<T, Alloc64>);

impl<T> Vec64<T> {
    #[inline]
    pub fn new() -> Self {
        Self(Vec::new_in(Alloc64))
    }

    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self(Vec::with_capacity_in(cap, Alloc64))
    }

    /// Useful when interpreting raw bytes that are buffered 
    /// in a Vec64 compatible manner, from network sockets etc.,
    /// to avoid needing to copy. 
    /// 
    /// # Safety
    /// - `buf` must have come from a `Vec64<u8>` that owns the allocation.
    /// - `T` must be POD (plain old data), properly aligned (which `Vec64` guarantees).
    /// - `buf.len() % size_of::<T>() == 0`
    pub unsafe fn from_vec64_u8(buf: Vec64<u8>) -> Vec64<T> {
        let byte_len = buf.len();
        let elem_size = mem::size_of::<T>();
        assert!(byte_len % elem_size == 0, "Size mismatch in from_vec64_u8");

        let ptr = buf.0.as_ptr() as *mut T;
        let len = byte_len / elem_size;
        let cap = buf.0.capacity() / elem_size;

        // Prevent Vec<u8> destructor from running
        let _ = mem::ManuallyDrop::new(buf.0);

        let vec = unsafe { Vec::from_raw_parts_in(ptr, len, cap, Alloc64) };
        Vec64(vec)
    }

    /// Takes ownership of a raw allocation.
    /// 
    /// # Safety:
    /// - `ptr` must have been allocated by `Alloc64` (or compatible 64-byte aligned allocator)
    /// - `ptr` must be valid for reads and writes for `len * size_of::<T>()` bytes
    /// - `len` must be less than or equal to `capacity`
    /// - The memory must not be aliased elsewhere
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize, capacity: usize) -> Self {
        debug_assert_eq!(
            (ptr as usize) % 64,
            0,
            "Vec64::from_raw_parts: pointer is not 64-byte aligned"
        );
        
        let vec = unsafe { Vec::from_raw_parts_in(ptr, len, capacity, Alloc64) };
        Self(vec)
    }
}

// Only require Send+Sync for parallel iterator methods
#[cfg(feature = "parallel_proc")]
impl<T: Sync + Send> Vec64<T> {
    #[inline]
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, T> {
        self.0.par_iter()
    }

    #[inline]
    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, T> {
        self.0.par_iter_mut()
    }
}

impl<T: Copy> Vec64<T> {
    #[inline]
    pub fn from_slice(slice: &[T]) -> Self {
        let mut v = Self::with_capacity(slice.len());
        // SAFETY: allocated enough capacity, and both
        // pointers are non-overlapping.
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), v.0.as_mut_ptr(), slice.len());
            v.0.set_len(slice.len());
        }
        v
    }
}

impl<T: Clone> Vec64<T> {
    #[inline]
    pub fn from_slice_clone(slice: &[T]) -> Self {
        let mut v = Self::with_capacity(slice.len());
        v.0.extend_from_slice(slice);
        v
    }
}

impl<T> Default for Vec64<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Deref for Vec64<T> {
    type Target = Vec<T, Alloc64>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Vec64<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Clone> Clone for Vec64<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T: Debug> Debug for Vec64<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        self.0.fmt(f)
    }
}

impl<T: PartialEq> PartialEq for Vec64<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: PartialEq> PartialEq<Buffer<T>> for Vec64<T> {
    #[inline]
    fn eq(&self, other: &Buffer<T>) -> bool {
        self.deref() == other.deref()
    }
}

impl<T: Display> Display for Vec64<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "[")?;
        for (i, item) in self.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{item}")?;
        }
        write!(f, "]")
    }
}

impl<T> IntoIterator for Vec64<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T, Alloc64>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Vec64<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl<'a, T> IntoIterator for &'a mut Vec64<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

impl<T> Extend<T> for Vec64<T> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.0.extend(iter)
    }
}

impl<T> FromIterator<T> for Vec64<T> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iterator = iter.into_iter();
        let mut v = if let Some(exact) = iterator.size_hint().1 {
            Vec::with_capacity_in(exact, Alloc64)
        } else {
            Vec::with_capacity_in(iterator.size_hint().0, Alloc64)
        };
        v.extend(iterator);
        Self(v)
    }
}

impl<T> From<Vec<T, Alloc64>> for Vec64<T> {
    #[inline]
    fn from(v: Vec<T, Alloc64>) -> Self {
        Self(v)
    }
}

impl<T> From<Vec64<T>> for Vec<T, Alloc64> {
    #[inline]
    fn from(v: Vec64<T>) -> Self {
        v.0
    }
}

impl<T> From<Vec<T>> for Vec64<T> {
    #[inline]
    fn from(v: Vec<T>) -> Self {
        let mut vec = Vec::with_capacity_in(v.len(), Alloc64);
        vec.extend(v);
        Self(vec)
    }
}

impl<T> From<&[T]> for Vec64<T>
where
    T: Clone
{
    #[inline]
    fn from(s: &[T]) -> Self {
        let mut v = Vec::with_capacity_in(s.len(), Alloc64);
        v.extend_from_slice(s);
        Self(v)
    }
}

impl<T> AsRef<[T]> for Vec64<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.0.as_ref()
    }
}
impl<T> AsMut<[T]> for Vec64<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut()
    }
}

impl<T> Borrow<[T]> for Vec64<T> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.0.borrow()
    }
}
impl<T> BorrowMut<[T]> for Vec64<T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.0.borrow_mut()
    }
}

#[macro_export]
macro_rules! vec64 {
    // Bool: repetition form
    (bool $elem:expr; $n:expr) => {{
        let len       = $n as usize;
        let byte_len  = (len + 7) / 8;
        let mut v     = $crate::Vec64::<u8>::with_capacity(byte_len);

        // Fill the buffer in one shot.
        let fill = if $elem { 0xFFu8 } else { 0u8 };
        v.0.resize(byte_len, fill);

        // Clear padding bits when fill == 1 and len is not a multiple of 8.
        if $elem && (len & 7) != 0 {
            let mask  = (1u8 << (len & 7)) - 1;
            let last  = byte_len - 1;
            v.0[last] &= mask;
        }
        v
    }};

    // Bool: list form
    (bool $($x:expr),+ $(,)?) => {{
        // Count elements at macro-expansion time.
        let len: usize = 0 $(+ { let _ = &$x; 1 })*;
        let byte_len   = (len + 7) / 8;
        let mut v      = $crate::Vec64::<u8>::with_capacity(byte_len);
        v.0.resize(byte_len, 0);

        // Sequentially set bits – no reallocations.
        let mut _idx = 0usize;
        $(
            if $x {
                $crate::null_masking::set_bit(&mut v.0, _idx);
            }
            _idx += 1;
        )+
        v
    }};

    // Generic forms
    () => {
        $crate::Vec64::new()
    };

    ($elem:expr; $n:expr) => {{
        let mut v = $crate::Vec64::with_capacity($n);
        v.0.resize($n, $elem);
        v
    }};

    ($($x:expr),+ $(,)?) => {{
        let mut v = $crate::Vec64::with_capacity(0 $(+ { let _ = &$x; 1 })*);
        $(v.push($x);)+
        v
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "parallel_proc")]
    #[test]
    fn test_new_and_default() {
        let v: Vec64<u32> = Vec64::new();
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), 0);

        let d: Vec64<u32> = Default::default();
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn test_with_capacity_and_alignment() {
        let v: Vec64<u64> = Vec64::with_capacity(32);
        assert_eq!(v.len(), 0);
        assert!(v.capacity() >= 32);
        // Underlying allocation must be 64-byte aligned
        assert_eq!(v.0.as_ptr() as usize % 64, 0);
    }

    #[test]
    fn test_from_slice_and_from() {
        let data = [1, 2, 3, 4, 5];
        let v = Vec64::from_slice(&data);
        assert_eq!(v.len(), 5);
        assert_eq!(&v[..], &data);

        let v2: Vec64<_> = Vec64::from(&data[..]);
        assert_eq!(&v2[..], &data);
    }

    #[test]
    fn test_vec_macro() {
        let v = vec64![1, 2, 3, 4, 5];
        assert_eq!(&v[..], &[1, 2, 3, 4, 5]);

        let v2 = vec64![7u8; 4];
        assert_eq!(&v2[..], &[7u8; 4]);
    }

    #[test]
    fn test_extend_and_from_iter() {
        let mut v = Vec64::new();
        v.extend([10, 20, 30]);
        assert_eq!(&v[..], &[10, 20, 30]);

        let v2: Vec64<_> = [100, 200].into_iter().collect();
        assert_eq!(&v2[..], &[100, 200]);
    }

    #[test]
    fn test_push_and_index() {
        let mut v = Vec64::with_capacity(2);
        v.push(123);
        v.push(456);
        assert_eq!(v[0], 123);
        assert_eq!(v[1], 456);
    }

    #[test]
    fn test_as_ref_and_as_mut() {
        let mut v = Vec64::from_slice(&[1, 2, 3]);
        assert_eq!(v.as_ref(), &[1, 2, 3]);
        v.as_mut()[1] = 99;
        assert_eq!(v[1], 99);
    }

    #[test]
    fn test_borrow_traits() {
        use std::borrow::{Borrow, BorrowMut};
        let mut v = Vec64::from_slice(&[4, 5, 6]);
        let r: &[i32] = v.borrow();
        assert_eq!(r, &[4, 5, 6]);
        let r: &mut [i32] = v.borrow_mut();
        r[0] = 42;
        assert_eq!(v[0], 42);
    }

    #[test]
    fn test_clone_partial_eq_debug_display() {
        let v = vec64![1, 2, 3];
        let c = v.clone();
        assert_eq!(v, c);
        let s = format!("{:?}", v);
        assert!(s.contains("1"));
        let s2 = format!("{}", v);
        assert_eq!(s2, "[1, 2, 3]");
    }

    #[test]
    fn test_into_iterator() {
        let v = vec64![2, 4, 6];
        let mut out = Vec::new();
        for x in v {
            out.push(x);
        }
        assert_eq!(out, vec![2, 4, 6]);
    }

    #[test]
    fn test_iter_and_iter_mut() {
        let v = vec64![1, 2, 3];
        let sum: i32 = v.iter().copied().sum();
        assert_eq!(sum, 6);

        let mut v = vec64![0, 0, 0];
        for x in &mut v {
            *x = 7;
        }
        assert_eq!(v[..], [7, 7, 7]);
    }

    #[test]
    fn test_from_std_vec() {
        let std_v = vec![1, 2, 3, 4];
        let v: Vec64<_> = std_v.clone().into();
        assert_eq!(v[..], [1, 2, 3, 4]);
    }

    #[test]
    fn test_into_std_vec() {
        let v = vec64![7, 8, 9];
        let std_v: Vec<_> = v.0.clone().to_vec();
        assert_eq!(std_v, vec![7, 8, 9]);
    }

    #[test]
    fn test_alignment_is_64() {
        let v: Vec64<u8> = Vec64::with_capacity(32);
        assert_eq!(v.0.as_ptr() as usize % 64, 0);
    }

    #[test]
    fn test_zero_sized_types() {
        let v: Vec64<()> = vec64![(); 10];
        assert_eq!(v.len(), 10);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds() {
        let v: Vec64<i32> = Vec64::new();
        let _ = v[1];
    }

    /// Utility: check that a pointer is 64-byte aligned.
    fn assert_aligned_64<T>(vec: &Vec64<T>) {
        let ptr = vec.as_ptr() as usize;
        assert_eq!(ptr % 64, 0, "Pointer {:p} not 64-byte aligned", vec.as_ptr());
    }

    #[test]
    fn test_vec64_new_alignment() {
        let v: Vec64<u32> = Vec64::new();
        // Even with capacity 0, allocation should be 64-byte aligned (when not null).
        // (Vec with cap 0 may have dangling non-null but still aligned pointer.)
        if v.capacity() > 0 {
            assert_aligned_64(&v);
        }
    }

    #[test]
    fn test_vec64_with_capacity_alignment() {
        for &n in &[1, 3, 7, 32, 1024, 4096] {
            let v: Vec64<u8> = Vec64::with_capacity(n);
            assert_aligned_64(&v);
        }
    }

    #[test]
    fn test_vec64_from_slice_alignment() {
        let data = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let v = Vec64::from_slice(&data);
        assert_aligned_64(&v);
    }

    #[test]
    fn test_vec64_macro_alignment() {
        let v = vec64![0u32; 64];
        assert_aligned_64(&v);

        let v2 = vec64![1u16, 2, 3, 4, 5];
        assert_aligned_64(&v2);
    }

    #[test]
    fn test_vec64_grow_alignment() {
        let mut v: Vec64<u64> = Vec64::with_capacity(1);
        assert_aligned_64(&v);
        for i in 0..1000 {
            v.push(i);
            assert_aligned_64(&v);
        }
    }

    #[test]
    fn test_vec64_alignment_zst() {
        let v: Vec64<()> = Vec64::with_capacity(100);
        assert_eq!(v.capacity(), usize::MAX, "ZST Vec should have 'infinite' capacity");
    }
}

#[cfg(test)]
#[cfg(feature = "parallel_proc")]
mod parallel_tests {
    use rayon::iter::ParallelIterator;

    use super::*;

    #[test]
    fn test_vec64_par_iter() {
        let v = Vec64::from_slice(&[1u32, 2, 3, 4, 5]);
        let sum: u32 = v.par_iter().sum();
        assert_eq!(sum, 15);
    }
}
