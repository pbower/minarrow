//! # NumericArrayView Module - *Windowed View over a NumericArray*
//!
//! `NumericArrayV` is a **read-only, windowed view** over a [`NumericArray`].
//! It groups all integer and float variants and exposes a zero-copy slice
//! `[offset .. offset + len)` for fast, indexable access.
//!
//! ## Role
//! - Lets APIs accept either a full `NumericArray` or a pre-sliced view.
//! - Avoids deep copies while enabling per-window operations and previews.
//! - Can cache per-window null counts to speed up repeated scans.
//!
//! ## Behaviour
//! - Works across numeric variants (ints + floats) behind `NumericArray`.
//! - Provides convenience accessors like [`get_f64`](NumericArrayV::get_f64) that
//!   upcast to `f64` for uniform downstream handling.
//! - Slicing returns another borrowed view; data buffers are not cloned.
//!
//! ## Threading
//! - Not thread-safe: uses `Cell` to cache the window’s null count.
//! - For parallel use, create per-thread views via [`slice`](NumericArrayV::slice).
//!
//! ## Interop
//! - Convert to an owned `NumericArray` of the window via
//!   [`to_numeric_array`](NumericArrayV::to_numeric_array).
//! - Lift to `Array` with [`as_array`](NumericArrayV::as_array) when you need
//!   enum-level APIs.
//!
//! ## Invariants
//! - `offset + len <= array.len()`
//! - `len` is the logical row count of this view.

use std::cell::Cell;
use std::fmt::{self, Debug, Display, Formatter};

use crate::structs::views::bitmask_view::BitmaskV;
use crate::traits::print::MAX_PREVIEW;
use crate::{Array, ArrayV, FieldArray, MaskedArray, NumericArray};

/// # NumericArrayView
///
/// Read-only, zero-copy view over a `[offset .. offset + len)` window of a
/// [`NumericArray`].
///
/// ## Purpose
/// - Return an indexable subrange without cloning buffers.
/// - Optionally cache per-window null counts for faster repeated passes.
///
/// ## Behaviour
/// - Groups integer and float variants under one enum.
/// - Upcasts via [`get_f64`](Self::get_f64) for uniform handling.
/// - Further slicing yields another borrowed view.
///
/// ## Fields
/// - `array`: backing [`NumericArray`] (enum over numeric types).
/// - `offset`: starting index into the backing array.
/// - `len`: logical number of elements in the view.
/// - `null_count`: cached `Option<usize>` for this window (internal).
///
/// ## Notes
/// - Not thread-safe due to `Cell`. Create per-thread views with [`slice`](Self::slice).
/// - Use [`to_numeric_array`](Self::to_numeric_array) to materialise the window.
#[derive(Clone, PartialEq)]
pub struct NumericArrayV {
    pub array: NumericArray,
    pub offset: usize,
    len: usize,
    null_count: Cell<Option<usize>>
}

impl NumericArrayV {
    /// Creates a new `NumericArrayView` with the given offset and length.
    pub fn new(array: NumericArray, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= array.len(),
            "NumericArrayView: window out of bounds (offset + len = {}, array.len = {})",
            offset + len,
            array.len()
        );
        Self {
            array,
            offset,
            len,
            null_count: Cell::new(None)
        }
    }

    /// Creates a new `NumericArrayView` with a precomputed null count.
    pub fn with_null_count(
        array: NumericArray,
        offset: usize,
        len: usize,
        null_count: usize
    ) -> Self {
        assert!(
            offset + len <= array.len(),
            "NumericArrayView: window out of bounds (offset + len = {}, array.len = {})",
            offset + len,
            array.len()
        );
        Self {
            array,
            offset,
            len,
            null_count: Cell::new(Some(null_count))
        }
    }

    /// Returns `true` if the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the underlying array as an `Array` enum value.
    ///
    /// Useful to access its inner methods
    #[inline]
    pub fn as_array(&self) -> Array {
        Array::NumericArray(self.array.clone()) // Arc clone for data buffer
    }

    /// Returns the value at logical index `i` as `f64`, or `None` if out of bounds or null.
    ///
    /// Converts any numeric types to `f64`, simplifying usage by avoiding explicit
    /// enum matches in caller code.
    ///
    /// # Notes
    /// - Returns `None` if `i` is out of bounds or the value is null.
    /// - Upcasts integer and float types to `f64` for uniform downstream handling.
    #[inline]
    pub fn get_f64(&self, i: usize) -> Option<f64> {
        if i >= self.len {
            return None;
        }
        let phys_idx = self.offset + i;
        match &self.array {
            NumericArray::Int32(arr) => arr.get(phys_idx).map(|v| v as f64),
            NumericArray::Int64(arr) => arr.get(phys_idx).map(|v| v as f64),
            NumericArray::UInt32(arr) => arr.get(phys_idx).map(|v| v as f64),
            NumericArray::UInt64(arr) => arr.get(phys_idx).map(|v| v as f64),
            NumericArray::Float32(arr) => arr.get(phys_idx).map(|v| v as f64),
            NumericArray::Float64(arr) => arr.get(phys_idx),
            NumericArray::Null => None,
            #[cfg(feature = "extended_numeric_types")]
            _ => unreachable!("get_f64: not implemented for extended numeric types")
        }
    }

    /// Unchecked, returns None for nulls, skips bounds check.
    ///
    /// Converts any numeric types to `f64`, simplifying usage by avoiding explicit
    /// enum matches in caller code.
    #[inline]
    pub unsafe fn get_f64_unchecked(&self, i: usize) -> Option<f64> {
        let phys_idx = self.offset + i;
        match &self.array {
            NumericArray::Int32(arr) => unsafe { arr.get_unchecked(phys_idx) }.map(|v| v as f64),
            NumericArray::Int64(arr) => unsafe { arr.get_unchecked(phys_idx) }.map(|v| v as f64),
            NumericArray::UInt32(arr) => unsafe { arr.get_unchecked(phys_idx) }.map(|v| v as f64),
            NumericArray::UInt64(arr) => unsafe { arr.get_unchecked(phys_idx) }.map(|v| v as f64),
            NumericArray::Float32(arr) => unsafe { arr.get_unchecked(phys_idx) }.map(|v| v as f64),
            NumericArray::Float64(arr) => unsafe { arr.get_unchecked(phys_idx) },
            NumericArray::Null => None,
            #[cfg(feature = "extended_numeric_types")]
            _ => unreachable!("get_f64_unchecked: not implemented for extended numeric types")
        }
    }

    /// Returns a windowed view into a sub-range of this view.
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(offset + len <= self.len, "NumericArrayView::slice: out of bounds");
        Self {
            array: self.array.clone(),
            offset: self.offset + offset,
            len,
            null_count: Cell::new(None)
        }
    }

    /// Materialise a deep copy as an owned NumericArray for the window.
    pub fn to_numeric_array(&self) -> NumericArray {
        self.as_array().slice_clone(self.offset, self.len).num()
    }

    /// Returns the end index of the view.
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Returns the view as a tuple `(array, offset, len)`.
    #[inline]
    pub fn as_tuple(&self) -> (NumericArray, usize, usize) {
        (self.array.clone(), self.offset, self.len)
    }

    /// Returns the length of the window
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the number of nulls in the view.
    #[inline]
    pub fn null_count(&self) -> usize {
        if let Some(count) = self.null_count.get() {
            return count;
        }
        let count = match self.array.null_mask() {
            Some(mask) => mask.to_window(self.offset, self.len).count_zeros(),
            None => 0
        };
        self.null_count.set(Some(count));
        count
    }

    /// Returns the null mask as a windowed `BitmaskView`.
    #[inline]
    pub fn null_mask_view(&self) -> Option<BitmaskV> {
        self.array.null_mask().map(|mask| mask.to_window(self.offset, self.len))
    }

    /// Sets the cached null count for the view.
    #[inline]
    pub fn set_null_count(&self, count: usize) {
        self.null_count.set(Some(count));
    }
}

impl From<NumericArray> for NumericArrayV {
    fn from(array: NumericArray) -> Self {
        let len = array.len();
        NumericArrayV {
            array,
            offset: 0,
            len,
            null_count: Cell::new(None)
        }
    }
}

impl From<FieldArray> for NumericArrayV {
    fn from(field_array: FieldArray) -> Self {
        match field_array.array {
            Array::NumericArray(arr) => {
                let len = arr.len();
                NumericArrayV {
                    array: arr,
                    offset: 0,
                    len,
                    null_count: Cell::new(None)
                }
            }
            _ => panic!("FieldArray does not contain a NumericArray")
        }
    }
}

impl From<Array> for NumericArrayV {
    fn from(array: Array) -> Self {
        match array {
            Array::NumericArray(arr) => {
                let len = arr.len();

                NumericArrayV {
                    array: arr,
                    offset: 0,
                    len,
                    null_count: Cell::new(None)
                }
            }
            _ => panic!("Array is not a NumericArray")
        }
    }
}

impl From<ArrayV> for NumericArrayV {
    fn from(view: ArrayV) -> Self {
        let (array, offset, len) = view.as_tuple();
        match array {
            Array::NumericArray(inner) => Self {
                array: inner,
                offset,
                len,
                null_count: Cell::new(None)
            },
            _ => panic!("From<ArrayView>: expected NumericArray variant")
        }
    }
}

impl Debug for NumericArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("NumericArrayView")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("array", &self.array)
            .field("cached_null_count", &self.null_count.get())
            .finish()
    }
}

impl Display for NumericArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let dtype = match &self.array {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(_) => "Int8",
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(_) => "Int16",
            NumericArray::Int32(_) => "Int32",
            NumericArray::Int64(_) => "Int64",
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(_) => "UInt8",
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(_) => "UInt16",
            NumericArray::UInt32(_) => "UInt32",
            NumericArray::UInt64(_) => "UInt64",
            NumericArray::Float32(_) => "Float32",
            NumericArray::Float64(_) => "Float64",
            NumericArray::Null => "Null"
        };

        writeln!(
            f,
            "NumericArrayView<{dtype}> [{} rows] (offset: {}, nulls: {})",
            self.len(),
            self.offset,
            self.null_count()
        )?;

        let max = self.len().min(MAX_PREVIEW);
        for i in 0..max {
            match self.get_f64(i) {
                Some(v) => writeln!(f, "  {v}")?,
                None => writeln!(f, "  ·")?
            }
        }

        if self.len() > MAX_PREVIEW {
            writeln!(f, "  ... ({} more)", self.len() - MAX_PREVIEW)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{Array, Bitmask, IntegerArray, NumericArray, vec64};

    #[test]
    fn test_numeric_array_view_basic_indexing_and_slice() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(100);
        arr.push(200);
        arr.push(300);
        arr.push(400);

        let numeric = NumericArray::Int32(Arc::new(arr));
        let view = NumericArrayV::new(numeric.clone(), 1, 2);

        assert_eq!(view.len(), 2);
        assert_eq!(view.offset, 1);

        // Valid indices
        assert_eq!(view.get_f64(0), Some(200.0));
        assert_eq!(view.get_f64(1), Some(300.0));
        assert_eq!(view.get_f64(2), None);

        // Slicing the view produces the correct sub-window
        let sub = view.slice(1, 1);
        assert_eq!(sub.len(), 1);
        assert_eq!(sub.get_f64(0), Some(300.0));
        assert_eq!(sub.get_f64(1), None);
    }

    #[test]
    fn test_numeric_array_view_null_count_and_cache() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        arr.push(3);
        arr.push(4);

        // Null mask: only index 2 is null
        let mut mask = Bitmask::new_set_all(4, true);
        mask.set(2, false);
        arr.null_mask = Some(mask);

        let numeric = NumericArray::Int32(Arc::new(arr));
        let view = NumericArrayV::new(numeric.clone(), 0, 4);
        assert_eq!(view.null_count(), 1, "Null count should detect one null");
        // Should use cached value next time
        assert_eq!(view.null_count(), 1);

        // Subwindow which excludes the null
        let view2 = view.slice(0, 2);
        assert_eq!(view2.null_count(), 0);
        // Subwindow which includes only the null
        let view3 = view.slice(2, 2);
        assert_eq!(view3.null_count(), 1);
    }

    #[test]
    fn test_numeric_array_view_with_supplied_null_count() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(5);
        arr.push(6);

        let numeric = NumericArray::Int32(Arc::new(arr));
        let view = NumericArrayV::with_null_count(numeric.clone(), 0, 2, 99);
        // Should always report the supplied cached value
        assert_eq!(view.null_count(), 99);
        view.set_null_count(101);
        assert_eq!(view.null_count(), 101);
    }

    #[test]
    fn test_numeric_array_view_to_numeric_array_and_as_tuple() {
        let mut arr = IntegerArray::<i32>::default();
        for v in 10..20 {
            arr.push(v);
        }
        let numeric = NumericArray::Int32(Arc::new(arr));
        let view = NumericArrayV::new(numeric.clone(), 4, 3);
        let arr2 = view.to_numeric_array();
        // Copy should be [14, 15, 16]
        if let NumericArray::Int32(a2) = arr2 {
            assert_eq!(a2.data, vec64![14, 15, 16]);
        } else {
            panic!("Unexpected variant");
        }

        // as_tuple returns correct metadata
        let tup = view.as_tuple();
        assert_eq!(&tup.0, &numeric);
        assert_eq!(tup.1, 4);
        assert_eq!(tup.2, 3);
    }

    #[test]
    fn test_numeric_array_view_null_mask_view() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(2);
        arr.push(4);
        arr.push(6);

        let mut mask = Bitmask::new_set_all(3, true);
        mask.set(0, false);
        arr.null_mask = Some(mask);

        let numeric = NumericArray::Int32(Arc::new(arr));
        let view = NumericArrayV::new(numeric, 1, 2);
        let mask_view = view.null_mask_view().expect("Should have mask");
        assert_eq!(mask_view.len(), 2);
        // Should map to bits 1 and 2 of original mask
        assert!(mask_view.get(0));
        assert!(mask_view.get(1));
    }

    #[test]
    fn test_numeric_array_view_from_numeric_array_and_array() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);

        let numeric = NumericArray::Int32(Arc::new(arr));
        let view_from_numeric = NumericArrayV::from(numeric.clone());
        assert_eq!(view_from_numeric.len(), 2);
        assert_eq!(view_from_numeric.get_f64(0), Some(1.0));

        let array = Array::NumericArray(numeric);
        let view_from_array = NumericArrayV::from(array);
        assert_eq!(view_from_array.len(), 2);
        assert_eq!(view_from_array.get_f64(1), Some(2.0));
    }

    #[test]
    #[should_panic(expected = "Array is not a NumericArray")]
    fn test_numeric_array_view_from_array_panics_on_wrong_variant() {
        let array = Array::Null;
        let _view = NumericArrayV::from(array);
    }
}
