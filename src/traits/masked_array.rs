//! Core `MaskedArray` trait, providing a common interface for all base array types,
//! including support for null masks.

use crate::{Bitmask, Length, Offset};

/// MaskedArray is implemented by all inner, nullable arrays.
/// 
/// ### Purpose
/// - MaskedArray ensures interface consistency across `BooleanArray`,
/// `CategoricalArray`, `DatetimeArray`, `FloatArray`, `IntegerArray`
/// and `StringArray`.
/// - It avoids repeition through default boilerplate implementations,
/// focusing on null value handling.
/// - This serves to enforce the base pattern contract, and is either overriden
/// on non-fixed width types (e.g., `BooleanArray`, `StringArray`), or, for fixed
/// width types (e.g., `FloatArray`, `IntegerArray`), is supported by macros.
pub trait MaskedArray {
    /// The element type (e.g. `f32`, `bool`, etc.)
    /// Or, utility type e.g., `Offsets` for cases
    /// like `String`
    type T: Default + PartialEq + Clone + Copy;

    /// The backing store (e.g. `Vec64<Self::Elem>` or `Bitmask`)
    type Container;

    /// The logical type that the data carries
    type LogicalType: Default;

    /// The type that implements `Copy` (e.g., &str)
    type CopyType: Default;

    /// **************************************************
    /// The below methods differ for the Boolean (bit-packed),
    /// and String (variable-length) variants and thus are
    /// implemented via macros for the standard variants,
    /// and then implemented on those types directly
    /// *************************************************

    /// Returns the number of elements in the array.
    fn len(&self) -> usize;

    /// Returns true if the array is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference to the underlying data.
    fn data(&self) -> &Self::Container;

    /// Returns a mutable reference to the underlying data.
    fn data_mut(&mut self) -> &mut Self::Container;

    /// Retrieves the value at the given index, or None if null or beyond length.
    fn get(&self, idx: usize) -> Option<Self::CopyType>;

    /// Sets the value at the given index, updating the null‐mask.
    fn set(&mut self, idx: usize, value: Self::LogicalType);

    /// Like `get`, but skips the `idx >= len()` check.
    unsafe fn get_unchecked(&self, idx: usize) -> Option<Self::CopyType>;

    /// Like `set`, but skips bounds checks.
    unsafe fn set_unchecked(&mut self, idx: usize, value: Self::LogicalType);

    /// Low-level accessor for when working directly with
    /// mutable array variants. 
    /// 
    /// Borrows with window parameters as a tuple, 
    /// for 'DIY' window access, retaining access to the whole original array.
    /// 
    /// `Offset` and `Length` are `usize` aliases.
    /// 
    /// For the standard zero-copy accessors, see the `View` trait.
    fn tuple_ref(&self, offset: usize, len: usize) -> (&Self, Offset, Length) {
        (&self, offset, len)
    }
    
    /// Returns an iterator over the T values in this array.
    fn iter(&self) -> impl Iterator<Item = Self::CopyType> + '_;

    /// Returns an iterator over the T values, as `Option<Self::T>`.
    fn iter_opt(&self) -> impl Iterator<Item = Option<Self::CopyType>> + '_;

    /// Returns an iterator over a range of T values in this array.
    fn iter_range(&self, offset: usize, len: usize) -> impl Iterator<Item = Self::CopyType> + '_;

    /// Returns an iterator over a range of T values, as `Option<T>`.
    fn iter_opt_range(&self, offset: usize, len: usize) -> impl Iterator<Item = Option<Self::CopyType>> + '_;

    /// Appends a value to the array, updating masks if present.
    fn push(&mut self, value: Self::LogicalType);

    /// Appends a value to the array, updating masks if present, 
    /// without bounds checks.
    /// 
    /// # Safety
    /// The caller must make sure there is enough pre-allocated
    /// size in the array, and no thread contention.
    unsafe fn push_unchecked(&mut self, value: Self::LogicalType);

    /// Returns a logical slice of the MaskedArray<Self::T> [offset, offset+len)
    /// as a new MaskedArray<Self::T> object via clone.
    /// 
    /// Prefer `View` trait slicers for zero-copy.
    fn slice_clone(&self, offset: usize, len: usize) -> Self;

    /// Resizes the array to contain `n` elements, via a call into self.data.resize().
    ///
    /// If `n` is greater than the current length, new elements are added using `T::default()`.
    /// If `n` is smaller, the array is truncated. This only affects the data buffer,
    /// not the null mask.
    fn resize(&mut self, n: usize, value: Self::LogicalType);

    /// **************************************************
    ///  We handle null masks consistently across all variants
    /// and thus their implementation sits on the trait, other
    /// than trait methods that need to access data state.
    /// **************************************************

    /// Returns a reference to the optional null mask.
    fn null_mask(&self) -> Option<&Bitmask>;

    /// Returns true if the value at the given index is null.
    #[inline]
    fn is_null(&self, idx: usize) -> bool {
        match &self.null_mask() {
            Some(mask) => !mask.get(idx),
            None => false
        }
    }

    /// Checks if the array has a null bitmask.
    fn is_nullable(&self) -> bool {
        self.null_mask().is_some()
    }

    /// Returns the total number of nulls.
    fn null_count(&self) -> usize {
        match self.null_mask().as_ref() {
            Some(mask) => mask.count_zeros(),
            None => 0
        }
    }

    /// Append a null value to the array, creating mask if needed
    #[inline]
    fn push_null(&mut self) {
        self.push(Self::LogicalType::default());
        let i = self.len() - 1;
        match self.null_mask_mut() {
            Some(m) => m.set(i, false),
            None => {
                let mut m = Bitmask::new_set_all(self.len(), true);
                m.set(i, false);
                self.set_null_mask(Some(m));
            }
        }
    }

    /// Returns a mutable reference to the optional null mask.
    fn null_mask_mut(&mut self) -> Option<&mut Bitmask>;

    /// Sets the null mask.
    fn set_null_mask(&mut self, mask: Option<Bitmask>);

    /// Appends a null value _without_ any bounds‐checks on the mask.
    ///
    /// # Safety
    /// You must ensure that after `push`, the data and mask (if present)
    /// have capacity for the new index, or you risk OOB on either.
    #[inline(always)]
    unsafe fn push_null_unchecked(&mut self) {
        // first, append a default element
        let idx = self.len();
        unsafe { self.set_unchecked(idx, Self::LogicalType::default()) };

        if let Some(mask) = self.null_mask_mut() {
            // mark null
            unsafe { mask.set_unchecked(idx, false) };
        } else {
            // initialise a new mask and mark this slot null
            let mut m = Bitmask::new_set_all(idx, true);
            unsafe { m.set_unchecked(idx, false) };
            self.set_null_mask(Some(m));
        }
    }

    /// Marks the value at the given index as null.
    #[inline]
    fn set_null(&mut self, idx: usize) {
        if let Some(nmask) = &mut self.null_mask_mut() {
            if nmask.len() <= idx {
                nmask.resize(idx + 1, true);
            }
            nmask.set(idx, false);
        } else {
            let mut m = Bitmask::new_set_all(self.len(), true);
            m.set(idx, false);
            self.set_null_mask(Some(m));
        }
    }

    /// Like `set_null`, but skips bounds checks.
    #[inline(always)]
    unsafe fn set_null_unchecked(&mut self, idx: usize) {
        if let Some(mask) = self.null_mask_mut() {
            mask.set(idx, false);
        } else {
            let mut m = Bitmask::new_set_all(self.len(), true);
            m.set(idx, false);
            self.set_null_mask(Some(m));
        }
    }

    /// Bulk-extend this array with `n` null entries
    #[inline]
    fn push_nulls(&mut self, n: usize) {
        let start = self.len();
        let end = start + n;

        self.resize(end, Self::LogicalType::default());

        if let Some(mask) = self.null_mask_mut() {
            mask.resize(end, false);
        } else {
            let mut m = Bitmask::new_set_all(end, true);
            for i in start..end {
                m.set(i, false);
            }
            self.set_null_mask(Some(m));
        }
    }

    /// Bulk-extend this array with `n` null entries, using unchecked mask writes.
    ///
    /// # Safety
    /// Caller must ensure there are no data races across threads on this mask.
    #[inline(always)]
    unsafe fn push_nulls_unchecked(&mut self, n: usize) {
        let start = self.len();
        let end = start + n;

        self.resize(end, Self::LogicalType::default());

        if let Some(mask) = self.null_mask_mut() {
            mask.resize(end, true);
            for i in 0..n {
                unsafe { mask.set_unchecked(start + i, false) };
            }
        } else {
            let mut m = Bitmask::new_set_all(end, true);
            for i in start..end {
                unsafe { m.set_unchecked(i, false) };
            }
            self.set_null_mask(Some(m));
        }
    }

    /// Appends all values (and null mask if present) from `other` to `self`.
    ///
    /// The appended array must be of the same concrete type and element type.
    /// 
    /// If this array is wrapped in a `FieldArray`, it will not be possible to
    /// mutate the array without reconstructing first, and a `ChunkedArray`
    /// is an alternative option.
    fn append_array(&mut self, other: &Self);

}
