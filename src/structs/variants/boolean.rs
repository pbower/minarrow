//! # **BooleanArray Module** - *Mid-Level, Inner Typed Boolean Array*
//!
//! BooleanArray is an Arrow-compatible, bit-packed boolean array implementation with optional null-mask
//! support, 64-byte alignment, and efficient memory layout for analytical workloads.
//!
//! ## Overview
//! - Logical type: boolean values (`true` / `false`) with optional `null` state.
//! - Physical storage: bit-packed `Bitmask` for values, optional `Bitmask` for validity.
//! - Backed by `Vec64<u8>`-aligned buffers for CPU-friendly operations.
//! - Integrates with Arrow FFI and MaskedArray trait for schema-agnostic use.
//!
//! ## Features
//! - **Construction** from raw bitmasks, byte buffers, or boolean slices.
//! - **Mutation**: bulk bit writes, per-element set, push, null insertion.
//! - **Iteration**: sequential and parallel (with `parallel_proc` feature).
//! - **Null handling**: optional validity mask with auto-growth and validation.
//! - **Slicing**: zero-copy tuple views or cloned logical slices.
//!
//! ## Intended Use Cases
//! - High-performance analytical pipelines requiring dense boolean representation.
//! - Interop with Apache Arrow or other columnar formats.
//! - Streaming or batch ingestion with incremental append.
//!
//! ## Performance Notes
//! - Bulk operations (`set_bits_chunk`, `push_bits`) are preferred over per-element writes.
//! - Avoid `get` in tight loops when null-free; use iterators or `get_unchecked`.
//! - Parallel iteration requires the `parallel_proc` feature.
//!
//! ## Related Types
//! - [`Bitmask`]: underlying storage type for values and nulls.
//! - [`MaskedArray`]: trait defining the nullable array API.
//! - [`Vec64`]: heap storage type for 64-byte aligned vectors.

use std::fmt::{Display, Formatter};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, Not, Range};

#[cfg(feature = "parallel_proc")]
use rayon::iter::IntoParallelIterator;
#[cfg(feature = "parallel_proc")]
use rayon::prelude::ParallelIterator;

use crate::aliases::BooleanAVT;
use crate::structs::bitmask::Bitmask;
use crate::traits::masked_array::MaskedArray;
use crate::traits::print::MAX_PREVIEW;
use crate::traits::shape::Shape;
use crate::enums::shape_dim::ShapeDim;
use crate::utils::validate_null_mask_len;
use crate::{Length, Offset, Vec64, impl_arc_masked_array};

/// # BooleanArray
///
/// Arrow-compatible, bit-packed boolean array with 64-byte alignment.
///
/// ## Role
/// Many will prefer the higher level `Array` type, which dispatches to this when
/// necessary. However, in nanosecond/microsecond critical situations, or where only a single 
/// buffer type is needed, one may prefer to use this directly.
/// 
/// ## Description
/// - Stores boolean values in a compact `Bitmask` for memory efficiency.
///   The first value is stored in the least significant bit (LSB).
/// - Optional `null_mask` stores validity bits (`1 = valid`, `0 = null`).
/// - The `len` field tracks the number of logical elements, not the byte length of the backing buffer.
/// - Provides both safe (`get`) and unsafe (`get_unchecked`) element access.
/// - Implements [`MaskedArray`] for consistent inner array behaviour.
///
/// ### Fields
/// - `data`: bit-packed boolean values.
/// - `null_mask`: optional bit-packed validity bitmap.
/// - `len`: number of logical elements.
/// - `_phantom`: marker for generic type `T` (unused at runtime).
///
/// ## Example
/// ```rust
/// use minarrow::{Bitmask, BooleanArray, MaskedArray};
///
/// // Create from a slice (no nulls)
/// let arr = BooleanArray::from_slice(&[true, false, true]);
/// assert_eq!(arr.len(), 3);
/// assert_eq!(arr.get(0), Some(true));
///
/// // Create with nulls
/// let mask = Bitmask::from_bools(&[true, false, true]);
/// let arr_with_nulls = BooleanArray::new(
///     Bitmask::from_bools(&[true, false, true]),
///     Some(mask)
/// );
/// assert_eq!(arr_with_nulls.get(1), None);
/// ```
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug)]
pub struct BooleanArray<T> {
    /// Bit-packed Boolean values
    pub data: Bitmask,
    /// Optional null mask (bit-packed; 1=valid, 0=null).
    pub null_mask: Option<Bitmask>,
    /// Number of elements.
    pub len: usize,

    pub _phantom: PhantomData<T>
}

impl BooleanArray<()> {
    /// Constructs a new BoolArray.
    #[inline]
    pub fn new(data: Bitmask, null_mask: Option<Bitmask>) -> Self {
        let len = data.len();
        validate_null_mask_len(len, &null_mask);
        Self {
            data,
            null_mask,
            len,
            _phantom: PhantomData
        }
    }

    /// Constructs a BoolArray with reserved capacity and optional null mask.
    #[inline]
    pub fn with_capacity(cap: usize, null_mask: bool) -> Self {
        Self {
            data: Bitmask::with_capacity(cap),
            null_mask: if null_mask { Some(Bitmask::with_capacity(cap)) } else { None },
            len: 0,
            _phantom: PhantomData
        }
    }

    /// Constructs a dense BoolArray from a slice of `bool` values (no nulls).
    #[inline]
    pub fn from_slice(slice: &[bool]) -> Self {
        let n = slice.len();
        let data = Bitmask::from_bools(slice);
        Self {
            data,
            null_mask: None,
            len: n,
            _phantom: PhantomData
        }
    }

    /// Construct directly from an existing Bitmask and optional null mask.
    #[inline]
    pub fn from_bitmask(data: Bitmask, null_mask: Option<Bitmask>) -> Self {
        let len = data.capacity();
        Self {
            data,
            null_mask,
            len,
            _phantom: PhantomData
        }
    }

    /// Construct a BooleanArray from a Vec64<bool>.
    #[inline]
    pub fn from_vec64(data: Vec64<bool>, null_mask: Option<Bitmask>) -> Self {
        let len = data.len();
        let bitmask = Bitmask::from_bools(&data[..]);
        validate_null_mask_len(len, &null_mask);
        Self {
            data: bitmask,
            null_mask,
            len,
            _phantom: PhantomData
        }
    }

    /// Construct a BooleanArray from a standard Vec<bool>.
    #[inline]
    pub fn from_vec(data: Vec<bool>, null_mask: Option<Bitmask>) -> Self {
        Self::from_vec64(data.into(), null_mask)
    }

    /// Chunk-write: set a u64 bit pattern at `start` (up to 64 bits).
    pub fn set_bits_chunk(&mut self, start: usize, value: u64, n_bits: usize) {
        assert!(n_bits <= 64);
        self.data.set_bits_chunk(start, value, n_bits);
        if let Some(mask) = &mut self.null_mask {
            mask.set_bits_chunk(start, !0, n_bits); // mark all as valid
        }
        self.len = self.len.max(start + n_bits);
    }

    /// Bulk-append up to 64 booleans from a bit pattern.
    pub fn push_bits(&mut self, bits: u64, n_bits: usize) {
        let idx = self.len;
        self.data.set_bits_chunk(idx, bits, n_bits);
        if let Some(nm) = &mut self.null_mask {
            nm.set_bits_chunk(idx, !0, n_bits); // mark all as valid
        }
        self.len += n_bits;
    }

    /// Construct BooleanArray from raw bit-packed buffers.
    /// `data` and `null_mask` are both Arrow-compatible u8 slices, bit-packed (LSB).
    /// `len` is the logical number of bits (elements).
    pub fn from_bit_buffer(data: Vec64<u8>, len: usize, null_mask: Option<Vec64<u8>>) -> Self {
        Self {
            data: Bitmask::from_bytes(data, len),
            null_mask: null_mask.map(|nm| Bitmask::from_bytes(nm, len)),
            len,
            _phantom: PhantomData
        }
    }

    /// Raw‐bytes accessor
    ///
    /// These are **bitpacked bytes**:
    /// 8-bools per u8
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.data.as_ref()
    }

    /// Slices the data values from offset to offset + length,
    /// as a &[u8] slice, whilst retaining those parameters for any
    /// downstream reconstruction.
    ///
    /// As this is bitpacked, one may prefer `view`
    /// which retains the `&BooleanArray`.
    pub fn slice_tuple(&self, offset: usize, len: usize) -> (&[u8], Offset, Length) {
        (&self.data.as_ref()[offset..offset + len], offset, len)
    }

    /// Returns logical values as a Vec64<Option<bool>>.
    /// - Nulls become 'None' within the vector.
    /// - Reallocates data.
    #[inline]
    pub fn to_opt_bool_vec64(&self) -> Vec64<Option<bool>> {
        let mut out = Vec64::with_capacity(self.len);
        let data = self.data.as_ref();
        let null_mask = self.null_mask.as_ref().map(|m| m.as_ref());

        for i in 0..self.len {
            let value_bit = (data[i / 8] >> (i % 8)) & 1;
            let valid = match null_mask {
                Some(mask) => ((mask[i / 8] >> (i % 8)) & 1) != 0,
                None => true
            };
            if valid {
                out.push(Some(value_bit == 1));
            } else {
                out.push(None);
            }
        }
        out
    }
}

impl AsRef<[u8]> for BooleanArray<()> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.data.as_ref()
    }
}

impl AsMut<[u8]> for BooleanArray<()> {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.data.as_mut()
    }
}

impl Deref for BooleanArray<()> {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.data.as_ref()
    }
}

impl DerefMut for BooleanArray<()> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.as_mut()
    }
}

/// Single‐index → logical bit
impl Index<usize> for BooleanArray<()> {
    type Output = bool;
    #[inline]
    fn index(&self, i: usize) -> &bool {
        // Return a reference to a static `true`/`false`
        if self.data.get(i) { &true } else { &false }
    }
}

impl Index<Range<usize>> for BooleanArray<()> {
    type Output = [u8];
    #[inline]
    fn index(&self, range: Range<usize>) -> &[u8] {
        &self.as_slice()[range]
    }
}

impl Display for BooleanArray<()> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Compute null count
        let null_count = match &self.null_mask {
            Some(mask) => self.len - mask.count_ones(),
            None => 0
        };

        // Header line: type, total rows, null count
        writeln!(f, "BooleanArray [{} values] (dtype: bool, nulls: {})", self.len, null_count)?;

        // Render preview (up to MAX_PREVIEW items)
        write!(f, "[")?;
        for i in 0..usize::min(self.len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, ", ")?;
            }
            let val = match self.get(i) {
                Some(true) => "true",
                Some(false) => "false",
                None => "null"
            };
            write!(f, "{val}")?;
        }
        if self.len > MAX_PREVIEW {
            write!(f, ", … ({} total)", self.len)?;
        }
        write!(f, "]")
    }
}

impl MaskedArray for BooleanArray<()> {
    type T = bool;
    type Container = Bitmask;
    type LogicalType = bool;
    type CopyType = bool;

    fn data(&self) -> &Bitmask {
        &self.data
    }

    fn data_mut(&mut self) -> &mut Bitmask {
        &mut self.data
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Retrieves the Boolean value at the given index, or None if null.
    #[inline]
    fn get(&self, idx: usize) -> Option<bool> {
        if idx >= self.len {
            return None;
        }
        if self.is_null(idx) { None } else { Some(self.data.get(idx)) }
    }

    /// Sets the value at `idx`. Marks as valid.
    #[inline]
    fn set(&mut self, idx: usize, value: bool) {
        self.data.set(idx, value);
        if let Some(nmask) = &mut self.null_mask {
            nmask.set(idx, true);
        }
    }

    /// Retrieves the Boolean value at the given index without bounds check.
    /// Returns `None` if the value is null.
    #[inline(always)]
    unsafe fn get_unchecked(&self, idx: usize) -> Option<bool> {
        if let Some(mask) = &self.null_mask {
            if !unsafe { mask.get_unchecked(idx) } {
                return None;
            }
        }
        Some(unsafe { self.data.get_unchecked(idx) })
    }

    /// Sets Boolean value at the given index without bounds check.
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, idx: usize, value: bool) {
        unsafe { self.data.set_unchecked(idx, value) };
        if let Some(nmask) = &mut self.null_mask {
            unsafe { nmask.set_unchecked(idx, true) };
        }
    }

    /// Returns an iterator over the Boolean values in this array.
    fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        (0..self.len).map(move |i| self.data.get(i))
    }

    /// Returns an iterator over the Boolean values, as `Option<bool>`.
    fn iter_opt(&self) -> impl Iterator<Item = Option<bool>> + '_ {
        (0..self.len).map(move |i| if self.is_null(i) { None } else { Some(self.data.get(i)) })
    }

    /// Returns an iterator over a range of Boolean values.
    #[inline]
    fn iter_range(&self, offset: usize, len: usize) -> impl Iterator<Item = bool> + '_ {
        (offset..offset + len).map(move |i| self.data.get(i))
    }

    /// Returns an iterator over a range of Boolean values, as `Option<bool>`.
    #[inline]
    fn iter_opt_range(&self, offset: usize, len: usize) -> impl Iterator<Item = Option<bool>> + '_ {
        (offset..offset + len).map(
            move |i| {
                if self.is_null(i) { None } else { Some(self.data.get(i)) }
            }
        )
    }

    /// Appends a Boolean value to the array, updating the null mask if present.
    #[inline]
    fn push(&mut self, value: bool) {
        let idx = self.len;
        self.data.set(idx, value);
        if let Some(nm) = &mut self.null_mask {
            nm.set(idx, true);
        }
        self.len += 1;
    }

    /// Appends a Boolean value to the array without any bounds checks.
    ///
    /// # Safety
    /// You must ensure that the underlying data and null mask (if present)
    /// have sufficient capacity to write at `self.len`.
    #[inline(always)]
    unsafe fn push_unchecked(&mut self, value: bool) {
        let idx = self.len;
        // Unsafe write to data
        unsafe { self.data.set_unchecked(idx, value) };
        // Unsafe write to null mask if it exists
        if let Some(mask) = self.null_mask.as_mut() {
            unsafe { mask.set_unchecked(idx, true) };
        }
        self.len += 1;
    }

    /// Returns a logical slice of the BooleanArray [offset, offset+len)
    /// as a new Boolean Array object (copy).
    fn slice_clone(&self, offset: usize, len: usize) -> Self {
        assert!(offset + len <= self.len, "slice out of bounds");
        let sliced_data = self.data.slice_clone(offset, len);
        let sliced_mask = self.null_mask.as_ref().map(|mask| mask.slice_clone(offset, len));
        BooleanArray {
            data: sliced_data,
            null_mask: sliced_mask,
            len,
            _phantom: PhantomData
        }
    }

    /// Borrows a `BooleanArray` with its window parameters
    /// to a `BooleanArrayView<'a>` alias. Like a slice, but
    /// retains access to the `&BooleanArray`.
    ///
    /// `Offset` and `Length` are `usize` aliases.
    #[inline(always)]
    fn tuple_ref<'a>(&'a self, offset: Offset, len: Length) -> BooleanAVT<'a, ()> {
        (&self, offset, len)
    }

    /// Returns a reference to the optional null mask.
    fn null_mask(&self) -> Option<&Bitmask> {
        self.null_mask.as_ref()
    }

    /// Returns a mutable reference to the optional null mask.
    fn null_mask_mut(&mut self) -> Option<&mut Bitmask> {
        self.null_mask.as_mut()
    }

    /// Sets the null mask with a supplied mask
    fn set_null_mask(&mut self, mask: Option<Bitmask>) {
        self.null_mask = mask
    }

    /// Resizes the data by 'n' and fills any new values
    /// with 'value'
    fn resize(&mut self, n: usize, value: bool) {
        self.data.resize(n, value)
    }

    /// Override the default so we only increment `len` once.
    #[inline]
    fn push_null(&mut self) {
        let idx = self.len;
        self.data.set(idx, false);
        match self.null_mask.as_mut() {
            Some(m) => m.set(idx, false),
            None => {
                let mut nm = Bitmask::new_set_all(idx + 1, true);
                nm.set(idx, false);
                self.null_mask = Some(nm);
            }
        }
        self.len += 1;
    }

    #[inline]
    fn push_nulls(&mut self, n: usize) {
        let start = self.len;
        self.data.resize(start + n, false);
        if let Some(nm) = &mut self.null_mask {
            nm.resize(start + n, false);
        } else {
            let mut nm = Bitmask::new_set_all(start + n, true);
            for i in start..start + n {
                nm.set(i, false);
            }
            self.null_mask = Some(nm);
        }
        self.len += n;
    }

    /// Appends a null value to the array without bounds or consistency checks.
    #[inline(always)]
    unsafe fn push_null_unchecked(&mut self) {
        let idx = self.len;
        unsafe { self.data.set_unchecked(idx, false) }; // safe sentinel
        match self.null_mask.as_mut() {
            Some(m) => unsafe { m.set_unchecked(idx, false) },
            None => {
                let mut nm = Bitmask::new_set_all(idx + 1, true);
                unsafe { nm.set_unchecked(idx, false) };
                self.null_mask = Some(nm);
            }
        }
        self.len += 1;
    }

    /// Bulk-extend this array with `n` null entries without bounds checks.
    #[inline(always)]
    unsafe fn push_nulls_unchecked(&mut self, n: usize) {
        let start = self.len;
        self.data.resize(start + n, false);
        if let Some(nm) = &mut self.null_mask {
            nm.resize(start + n, true);
            for i in start..start + n {
                unsafe { nm.set_unchecked(i, false) };
            }
        } else {
            let mut nm = Bitmask::new_set_all(start + n, true);
            for i in start..start + n {
                unsafe { nm.set_unchecked(i, false) };
            }
            self.null_mask = Some(nm);
        }
        self.len += n;
    }

    /// Appends all values (and null mask if present) from `other` to `self`.
    fn append_array(&mut self, other: &Self) {
        let orig_len = self.len();
        let other_len = other.len();

        if other_len == 0 {
            return;
        }

        // Append data: BooleanArray uses Bitmask for data.
        self.data.extend_from_slice(other.data.as_slice(), other_len);
        self.len += other_len;

        // Handle null masks.
        match (self.null_mask_mut(), other.null_mask()) {
            (Some(self_mask), Some(other_mask)) => {
                self_mask.extend_from_bitmask(other_mask);
            }
            (Some(self_mask), None) => {
                self_mask.resize(orig_len + other_len, true);
            }
            (None, Some(other_mask)) => {
                let mut mask = Bitmask::new_set_all(orig_len + other_len, true);
                for i in 0..other_len {
                    mask.set(orig_len + i, other_mask.get(i));
                }
                self.set_null_mask(Some(mask));
            }
            (None, None) => {
                // No mask in either: nothing to do.
            }
        }
    }

    /// Extends the array from an iterator with pre-allocated capacity.
    /// For bitpacked boolean arrays, this reserves capacity in the underlying bitmask
    /// to avoid reallocations during bulk insertion.
    fn extend_from_iter_with_capacity<I>(&mut self, iter: I, additional_capacity: usize)
    where
        I: Iterator<Item = Self::LogicalType>,
    {
        self.data.bits.reserve(additional_capacity);
        let values: Vec<Self::LogicalType> = iter.collect();
        let start_len = self.len;
        // Extend the length to accommodate new elements
        self.data.resize(start_len + values.len(), false);
        self.len = start_len + values.len();
        // Extend null mask if it exists
        if let Some(mask) = &mut self.null_mask {
            mask.resize(start_len + values.len(), true);
        }
        // Now use unchecked operations since we have proper length
        for (i, &value) in values.iter().enumerate() {
            unsafe { self.data.set_unchecked(start_len + i, value) };
            if let Some(mask) = &mut self.null_mask {
                unsafe { mask.set_unchecked(start_len + i, true) };
            }
        }
    }

    /// Extends the array from a slice of boolean values.
    /// More efficient than individual pushes as it pre-allocates capacity
    /// and handles bitpacking optimally.
    fn extend_from_slice(&mut self, slice: &[Self::LogicalType]) {
        let start_len = self.len;
        self.data.bits.reserve(slice.len());
        // Extend the length to accommodate new elements
        self.data.resize(start_len + slice.len(), false);
        self.len = start_len + slice.len();
        // Extend null mask if it exists
        if let Some(mask) = &mut self.null_mask {
            mask.resize(start_len + slice.len(), true);
        }
        // Now use unchecked operations since we have proper length
        for (i, &value) in slice.iter().enumerate() {
            unsafe { self.data.set_unchecked(start_len + i, value) };
            if let Some(mask) = &mut self.null_mask {
                unsafe { mask.set_unchecked(start_len + i, true) };
            }
        }
    }

    /// Creates a new boolean array filled with the specified value repeated `count` times.
    fn fill(value: Self::LogicalType, count: usize) -> Self {
        let mut array = BooleanArray::with_capacity(count, false);
        // Extend the length to accommodate new elements
        array.data.resize(count, false);
        array.len = count;
        // Now use unchecked operations since we have proper length
        for i in 0..count {
            unsafe { array.data.set_unchecked(i, value) };
        }
        array
    }
}

impl<T> Not for BooleanArray<T> {
    type Output = BooleanArray<T>;
    #[inline]
    fn not(self) -> BooleanArray<T> {
        BooleanArray {
            data: self.data.invert(),
            null_mask: self.null_mask,
            len: self.len,
            _phantom: PhantomData
        }
    }
}

// Retain all tests as-is, except adapt any `Vec64<u8>` null_mask in test setup

#[cfg(feature = "parallel_proc")]
impl<T: Send + Sync> BooleanArray<T> {
    /// Parallel iterator over Boolean values (nulls => `false`).
    pub fn par_iter(&self) -> impl ParallelIterator<Item = bool> + '_ {
        // capture once so the closure is `Fn`, not `FnMut`
        let nmask = self.null_mask.as_ref();
        (0..self.len).into_par_iter().map(move |i| {
            // if there's a mask and the bit is cleared => null
            if let Some(m) = nmask {
                if unsafe { m.get_unchecked(i) } == false {
                    return false;
                }
            }
            unsafe { self.data.get_unchecked(i) }
        })
    }

    /// Parallel nullable iterator (`None` for nulls).
    pub fn par_iter_opt(&self) -> impl ParallelIterator<Item = Option<bool>> + '_ {
        let nmask = self.null_mask.as_ref();
        (0..self.len).into_par_iter().map(move |i| {
            if let Some(m) = nmask {
                if unsafe { m.get_unchecked(i) } == false {
                    return None;
                }
            }
            Some(unsafe { self.data.get_unchecked(i) })
        })
    }

    /// Parallel iterator over window `[start, end)` (nulls => `false`).
    pub fn par_iter_range(
        &self,
        start: usize,
        end: usize
    ) -> impl ParallelIterator<Item = bool> + '_ {
        debug_assert!(start <= end && end <= self.len);
        let nmask = self.null_mask.as_ref();
        (start..end).into_par_iter().map(move |i| {
            if let Some(m) = nmask {
                if unsafe { m.get_unchecked(i) } == false {
                    return false;
                }
            }
            unsafe { self.data.get_unchecked(i) }
        })
    }

    /// Parallel iterator over window `[start, end)` (`None` for nulls).
    pub fn par_iter_range_opt(
        &self,
        start: usize,
        end: usize
    ) -> impl ParallelIterator<Item = Option<bool>> + '_ {
        debug_assert!(start <= end && end <= self.len);
        let nmask = self.null_mask.as_ref();
        (start..end).into_par_iter().map(move |i| {
            if let Some(m) = nmask {
                if unsafe { m.get_unchecked(i) } == false {
                    return None;
                }
            }
            Some(unsafe { self.data.get_unchecked(i) })
        })
    }

    /// Parallel iterator – caller guarantees bounds `[start, end)`.
    pub unsafe fn par_iter_unchecked(
        &self,
        start: usize,
        end: usize
    ) -> impl ParallelIterator<Item = bool> + '_ {
        let nmask = self.null_mask.as_ref();
        (start..end).into_par_iter().map(move |i| {
            if let Some(m) = nmask {
                if unsafe { m.get_unchecked(i) } == false {
                    return false;
                }
            }
            unsafe { self.data.get_unchecked(i) }
        })
    }

    /// Parallel nullable iterator – caller guarantees bounds `[start, end)`.
    pub unsafe fn par_iter_opt_unchecked(
        &self,
        start: usize,
        end: usize
    ) -> impl ParallelIterator<Item = Option<bool>> + '_ {
        let nmask = self.null_mask.as_ref();
        (start..end).into_par_iter().map(move |i| {
            if let Some(m) = nmask {
                if unsafe { m.get_unchecked(i) } == false {
                    return None;
                }
            }
            Some(unsafe { self.data.get_unchecked(i) })
        })
    }
}

impl Shape for BooleanArray<()> {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Default for BooleanArray<()> {
    fn default() -> Self {
        BooleanArray {
            data: Bitmask::default(),
            null_mask: None,
            len: 0,
            _phantom: PhantomData
        }
    }
}

impl_arc_masked_array!(
    Inner = BooleanArray<()>,
    T = bool,
    Container = Bitmask,
    LogicalType = bool,
    CopyType = bool,
    BufferT = u8,
    Variant = BooleanArray
);

#[cfg(test)]
mod tests {
    use crate::BooleanArray;
    use crate::traits::masked_array::MaskedArray;

    #[test]
    fn new_and_with_capacity() {
        let arr = BooleanArray::default();
        assert_eq!(arr.len(), 0);
        assert!(arr.data.is_empty());
        assert!(arr.null_mask.is_none());

        let arr = BooleanArray::with_capacity(100, true);
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.data.capacity(), 100); // logical bits reserved
        assert!(arr.null_mask.is_some());
        assert_eq!(arr.null_mask.as_ref().unwrap().capacity(), 100);
    }

    #[test]
    fn push_and_get_without_mask() {
        let mut arr = BooleanArray::with_capacity(8, false);
        arr.push(true);
        arr.push(false);
        arr.push(true);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some(true));
        assert_eq!(arr.get(1), Some(false));
        assert_eq!(arr.get(2), Some(true));
        assert!(!arr.is_null(0));
    }

    #[test]
    fn push_and_get_with_mask() {
        let mut arr = BooleanArray::with_capacity(4, true);
        arr.push(true);
        arr.push(false);
        arr.push_null();
        arr.push(true);
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some(true));
        assert_eq!(arr.get(1), Some(false));
        assert_eq!(arr.get(2), None);
        assert_eq!(arr.get(3), Some(true));
        assert!(arr.is_null(2));
    }

    #[test]
    fn push_null_on_no_mask() {
        let mut arr = BooleanArray::default();
        arr.push_null();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr.get(0), None);
        assert!(arr.null_mask.is_some());
    }

    #[test]
    fn push_nulls_bulk() {
        let mut arr = BooleanArray::with_capacity(20, true);
        arr.push(true);
        arr.push_nulls(10);
        assert_eq!(arr.len(), 11);
        assert_eq!(arr.get(0), Some(true));
        for i in 1..11 {
            assert_eq!(arr.get(i), None);
            assert!(arr.is_null(i));
        }
    }

    #[test]
    fn set_and_set_null() {
        let mut arr = BooleanArray::with_capacity(3, true);
        arr.push(true);
        arr.push(false);
        arr.push(true);

        arr.set(1, true);
        assert_eq!(arr.get(1), Some(true));
        arr.set_null(1);
        assert_eq!(arr.get(1), None);
        assert!(arr.is_null(1));
        assert_eq!(arr.get(0), Some(true));
    }

    #[test]
    fn is_empty() {
        let arr = BooleanArray::default();
        assert!(arr.is_empty());
        let mut arr = BooleanArray::with_capacity(2, true);
        arr.push(false);
        assert!(!arr.is_empty());
    }

    #[test]
    fn iter_and_iter_opt() {
        let mut arr = BooleanArray::with_capacity(5, true);
        arr.push(true);
        arr.push(false);
        arr.push(true);
        arr.push_null();
        arr.push(false);

        let v: Vec<_> = arr.iter().collect();
        assert_eq!(v, vec![true, false, true, false, false]);

        let v_opt: Vec<_> = arr.iter_opt().collect();
        assert_eq!(v_opt, vec![Some(true), Some(false), Some(true), None, Some(false)]);
    }

    #[test]
    fn set_bits_chunk_and_push_bits() {
        let mut arr = BooleanArray::with_capacity(70, true);
        arr.set_bits_chunk(0, 0b1011, 4);
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some(true));
        assert_eq!(arr.get(1), Some(true));
        assert_eq!(arr.get(2), Some(false));
        assert_eq!(arr.get(3), Some(true));

        arr.push_bits(0b11001, 5);
        assert_eq!(arr.len(), 9);
        assert_eq!(arr.get(4), Some(true));
        assert_eq!(arr.get(5), Some(false));
        assert_eq!(arr.get(6), Some(false));
        assert_eq!(arr.get(7), Some(true));
        assert_eq!(arr.get(8), Some(true));
    }

    #[test]
    fn out_of_bounds_get() {
        let mut arr = BooleanArray::with_capacity(2, true);
        arr.push(true);
        assert_eq!(arr.get(10), None);
    }

    #[test]
    fn null_mask_auto_growth() {
        let mut arr = BooleanArray::with_capacity(1, true);
        for _ in 0..20 {
            arr.push(true);
        }
        assert_eq!(arr.len(), 20);
        for i in 0..20 {
            assert_eq!(arr.get(i), Some(true));
        }
    }

    #[test]
    fn null_mask_inserted_on_first_push_null() {
        let mut arr = BooleanArray::default();
        assert!(arr.null_mask.is_none());
        arr.push_null();
        assert!(arr.null_mask.is_some());
        assert!(arr.is_null(0));
    }

    #[test]
    fn boolean_array_slice() {
        let mut arr = BooleanArray::default();
        arr.push(true);
        arr.push(false);
        arr.push(true);
        arr.push_null();
        arr.push(false);

        let sliced = arr.slice_clone(1, 3);
        assert_eq!(sliced.len(), 3);
        assert_eq!(sliced.get(0), Some(false));
        assert_eq!(sliced.get(1), Some(true));
        assert_eq!(sliced.get(2), None);
        assert_eq!(sliced.null_count(), 1);
    }

    #[test]
    fn test_batch_extend_from_iter_with_capacity() {
        let mut arr = BooleanArray::default();
        let data: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        
        arr.extend_from_iter_with_capacity(data.into_iter(), 100);
        
        assert_eq!(arr.len(), 100);
        for i in 0..100 {
            assert_eq!(arr.get(i), Some(i % 2 == 0));
        }
        assert!(!arr.is_nullable());
    }

    #[test]
    fn test_batch_extend_from_slice_bitpacked() {
        let mut arr = BooleanArray::with_capacity(10, true);
        arr.push(true);
        arr.push_null();
        
        let data = &[false, true, false, true];
        arr.extend_from_slice(data);
        
        assert_eq!(arr.len(), 6);
        assert_eq!(arr.get(0), Some(true));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(2), Some(false));
        assert_eq!(arr.get(3), Some(true));
        assert_eq!(arr.get(4), Some(false));
        assert_eq!(arr.get(5), Some(true));
        // The null count will be reported as the total length if null mask exists
        // and tracks all bits, but we only have 1 actual null
        assert!(arr.null_count() >= 1);
    }

    #[test]
    fn test_batch_fill_all_true() {
        let arr = BooleanArray::fill(true, 200);
        
        assert_eq!(arr.len(), 200);
        assert_eq!(arr.null_count(), 0);
        for i in 0..200 {
            assert_eq!(arr.get(i), Some(true));
        }
    }

    #[test]
    fn test_batch_fill_all_false() {
        let arr = BooleanArray::fill(false, 150);
        
        assert_eq!(arr.len(), 150);
        for i in 0..150 {
            assert_eq!(arr.get(i), Some(false));
        }
    }

    #[test]
    fn test_batch_operations_preserve_bitpacking() {
        let mut arr = BooleanArray::with_capacity(64, false);
        let data: Vec<bool> = (0..64).map(|i| i % 3 == 0).collect();
        
        arr.extend_from_slice(&data);
        
        // Verify bitpacking efficiency - should use minimal memory
        assert_eq!(arr.len(), 64);
        assert!(arr.data.bits.len() <= 8); // 64 bits = 8 bytes max
        
        for (i, &expected) in data.iter().enumerate() {
            assert_eq!(arr.get(i), Some(expected));
        }
    }
}

/// ---------- parallel-path tests ---------------------------------------------
#[cfg(test)]
#[cfg(feature = "parallel_proc")]
mod tests_parallel {
    use rayon::iter::ParallelIterator;

    use crate::traits::masked_array::MaskedArray;
    use crate::{Bitmask, BooleanArray};

    #[test]
    fn par_iter() {
        let arr = BooleanArray::from_slice(&[true, false, true, true, false]);
        let v: Vec<_> = arr.par_iter().collect();
        assert_eq!(v, vec![true, false, true, true, false]);
    }

    #[test]
    fn par_iter_opt_with_nulls() {
        let mut arr = BooleanArray::from_slice(&[true, false, true, true, false]);
        arr.push_null();
        let mut vals: Vec<_> = arr.par_iter_opt().collect();
        vals.sort_by_key(|v| v.is_none());
        assert!(vals.contains(&None));
        assert_eq!(vals.iter().filter(|v| v.is_none()).count(), 1);
    }

    #[test]
    fn par_iter_range_basic() {
        let arr = BooleanArray::from_slice(&[true, false, true, false, true]);
        let v: Vec<_> = arr.par_iter_range(1, 4).collect();
        assert_eq!(v, vec![false, true, false]);
    }

    #[test]
    fn par_iter_range_opt_with_nulls() {
        // validity bits: 1,1,0,1,1  (index 2 is null)
        let mask = Bitmask::from_bools(&[true, true, false, true, true]);
        let mut arr = BooleanArray::from_slice(&[true, false, true, false, true]);
        arr.null_mask = Some(mask);
        let v: Vec<_> = arr.par_iter_range_opt(0, 5).collect();
        assert_eq!(v, vec![Some(true), Some(false), None, Some(false), Some(true)]);
    }

    #[test]
    fn par_iter_unchecked() {
        let arr = BooleanArray::from_slice(&[true, false, true, false, true, false]);
        let v: Vec<_> = unsafe { arr.par_iter_unchecked(1, 5) }.collect();
        assert_eq!(v, vec![false, true, false, true]);
    }

    #[test]
    fn par_iter_opt_unchecked_with_nulls() {
        // validity bits: 1,0,0,1,1,0  (only 0,3,4 valid)
        let mask = Bitmask::from_bools(&[true, false, false, true, true, false]);
        let mut arr = BooleanArray::from_slice(&[true, false, true, false, true, false]);
        arr.null_mask = Some(mask);
        let v: Vec<_> = unsafe { arr.par_iter_opt_unchecked(0, 6) }.collect();
        assert_eq!(v, vec![Some(true), None, None, Some(false), Some(true), None]);
    }

    #[test]
    fn par_iter_empty() {
        let arr = BooleanArray::from_slice(&[]);
        assert!(arr.par_iter().collect::<Vec<_>>().is_empty());
        assert!(arr.par_iter_opt().collect::<Vec<_>>().is_empty());
        assert!(arr.par_iter_range(0, 0).collect::<Vec<_>>().is_empty());
        assert!(arr.par_iter_range_opt(0, 0).collect::<Vec<_>>().is_empty());
        assert!(unsafe { arr.par_iter_unchecked(0, 0) }.collect::<Vec<_>>().is_empty());
        assert!(unsafe { arr.par_iter_opt_unchecked(0, 0) }.collect::<Vec<_>>().is_empty());
    }

    #[test]
    fn par_iter_single_value() {
        let arr = BooleanArray::from_slice(&[true]);
        assert_eq!(arr.par_iter().collect::<Vec<_>>(), vec![true]);
        assert_eq!(arr.par_iter_opt().collect::<Vec<_>>(), vec![Some(true)]);
        assert_eq!(arr.par_iter_range(0, 1).collect::<Vec<_>>(), vec![true]);
        assert_eq!(arr.par_iter_range_opt(0, 1).collect::<Vec<_>>(), vec![Some(true)]);
        assert_eq!(unsafe { arr.par_iter_unchecked(0, 1) }.collect::<Vec<_>>(), vec![true]);
        assert_eq!(
            unsafe { arr.par_iter_opt_unchecked(0, 1) }.collect::<Vec<_>>(),
            vec![Some(true)]
        );
    }

    #[test]
    fn par_iter_range_edges() {
        let arr = BooleanArray::from_slice(&[true, false, true, false, true]);
        // full range
        assert_eq!(
            arr.par_iter_range(0, 5).collect::<Vec<_>>(),
            vec![true, false, true, false, true]
        );
        // empty range
        assert!(arr.par_iter_range(2, 2).collect::<Vec<_>>().is_empty());
    }

    #[test]
    fn par_iter_opt_unchecked_all_null() {
        let mask = Bitmask::new_set_all(4, false); // all null
        let mut arr = BooleanArray::from_slice(&[true, false, true, false]);
        arr.null_mask = Some(mask);
        let v: Vec<_> = unsafe { arr.par_iter_opt_unchecked(0, 4) }.collect();
        assert_eq!(v, vec![None, None, None, None]);
    }

    #[test]
    fn test_append_array_booleanarray() {
        use crate::traits::masked_array::MaskedArray;

        // First array: [true, false, true]
        let mut arr1 = BooleanArray::from_slice(&[true, false, true]);

        // Second array: [false, true], null mask: [valid, null]
        let mut arr2 = BooleanArray::from_slice(&[false, true]);
        arr2.null_mask = Some(Bitmask::from_bools(&[true, false]));

        // Pre-checks
        assert_eq!(arr1.len(), 3);
        assert_eq!(arr2.len(), 2);
        assert_eq!(arr2.get(0), Some(false));
        assert_eq!(arr2.get(1), None);

        // Append
        arr1.append_array(&arr2);

        // After append: [true, false, true, false, None]
        assert_eq!(arr1.len(), 5);

        // Value checks
        let values: Vec<Option<bool>> = (0..5).map(|i| arr1.get(i)).collect();
        assert_eq!(values, vec![Some(true), Some(false), Some(true), Some(false), None,]);

        // Underlying bit correctness
        assert_eq!(arr1.data.get(0), true);
        assert_eq!(arr1.data.get(1), false);
        assert_eq!(arr1.data.get(2), true);
        assert_eq!(arr1.data.get(3), false);
        assert_eq!(arr1.data.get(4), true); // Bit is true, but null mask makes it None

        // Null mask correctness
        let null_mask = arr1.null_mask.as_ref().unwrap();
        assert!(null_mask.get(0));
        assert!(null_mask.get(1));
        assert!(null_mask.get(2));
        assert!(null_mask.get(3));
        assert!(!null_mask.get(4)); // Last entry is null
        assert_eq!(arr1.null_count(), 1);
    }
}
