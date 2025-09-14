//! # **ArrayView Module** - *Windowed View over an Array*
//!
//! `ArrayV` is a **logical, read-only, zero-copy view** into a contiguous window
//! `[offset .. offset + len)` of any [`Array`] variant.
//!
//! ## Purpose
//! - Provides indexable, bounds-checked access to a subrange of an array without copying buffers.
//! - Caches null counts per view for efficient repeated queries.
//! - Acts as a unifying abstraction for windowed operations across all array types.
//!
//! ## Behaviour
//! - All indices are **relative** to the view's start.
//! - Internally retains an `Arc` reference to the parent array's buffers.
//! - Windowing and slicing are O(1) operations (pointer + metadata updates only).
//! - Cached null counts are stored in an `OnceLock` for thread-safe lazy initialization.
//!
//! ## Threading
//! - Thread-safe for sharing across threads (uses `OnceLock` for null count caching).
//! - Safe to share via `Arc` for parallel processing.
//!
//! ## Interop
//! - Convert back to a full array via [`to_array`](ArrayV::to_array).
//! - Promote to `(Array, offset, len)` tuple with [`as_tuple`](ArrayV::as_tuple).
//! - Access raw data pointer and element size via [`data_ptr_and_byte_len`](ArrayV::data_ptr_and_byte_len).
//!
//! ## Invariants
//! - `offset + len <= array.len()`
//! - `len` reflects the **logical** number of elements in the view.

use std::fmt::{self, Debug, Display, Formatter};
use std::sync::OnceLock;

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::traits::concatenate::Concatenate;
use crate::traits::print::MAX_PREVIEW;
use crate::traits::shape::Shape;
use crate::{Array, BitmaskV, FieldArray, MaskedArray, TextArray};

/// # ArrayView
///
/// Logical, windowed view over an `Array`.
///
/// ## Purpose
/// This is used to return an indexable view over a subset of the array.
/// Additionally, it can be used to cache null counts for those regions,
/// which can be used to speed up calculations.
///
/// ## Behaviour
/// - Indices are always relative to the window.
/// - Holds a reference to the original `Array` and window bounds.
/// - Windowing uses an arc clone
/// - All access (get/index, etc.) is offset-correct and bounds-checked.
/// - Null count is computed once (on demand or at creation) and cached for subsequent use.
///
/// ## Notes
/// - Use [`slice`](Self::slice) to derive smaller views without data copy.
/// - Use [`to_array`](Self::to_array) to materialise as an owned array.
#[derive(Clone, PartialEq)]
pub struct ArrayV {
    pub array: Array, // contains Arc<inner>
    pub offset: usize,
    len: usize,
    null_count: OnceLock<usize>,
}

impl ArrayV {
    /// Construct a windowed view of `array[offset..offset+len)`, with optional precomputed null count.
    #[inline]
    pub fn new(array: Array, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= array.len(),
            "ArrayView: window out of bounds (offset + len = {}, array.len = {})",
            offset + len,
            array.len()
        );
        Self {
            array,
            offset,
            len,
            null_count: OnceLock::new(),
        }
    }

    /// Construct a windowed view, supplying a precomputed null count.
    #[inline]
    pub fn with_null_count(array: Array, offset: usize, len: usize, null_count: usize) -> Self {
        assert!(
            offset + len <= array.len(),
            "ArrayView: window out of bounds (offset + len = {}, array.len = {})",
            offset + len,
            array.len()
        );
        let lock = OnceLock::new();
        let _ = lock.set(null_count); // Pre-initialize with the provided count
        Self {
            array,
            offset,
            len,
            null_count: lock,
        }
    }

    /// Return the logical length of the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the value at logical index `i` within the window, or `None` if out of bounds or null.
    #[inline]
    pub fn get<T: MaskedArray + 'static>(&self, i: usize) -> Option<T::CopyType> {
        if i >= self.len {
            return None;
        }
        self.array.inner::<T>().get(self.offset + i)
    }

    /// Returns the value at logical index `i` within the window (unchecked).
    #[inline]
    pub fn get_unchecked<T: MaskedArray + 'static>(&self, i: usize) -> Option<T::CopyType> {
        unsafe { self.array.inner::<T>().get_unchecked(self.offset + i) }
    }

    /// Returns the string value at logical index `i` within the window, or `None` if out of bounds or null.
    #[inline]
    pub fn get_str(&self, i: usize) -> Option<&str> {
        if i >= self.len {
            return None;
        }
        match &self.array {
            Array::TextArray(TextArray::String32(arr)) => arr.get_str(self.offset + i),
            #[cfg(feature = "large_string")]
            Array::TextArray(TextArray::String64(arr)) => arr.get_str(self.offset + i),
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical8(arr)) => arr.get_str(self.offset + i),
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical16(arr)) => arr.get_str(self.offset + i),
            Array::TextArray(TextArray::Categorical32(arr)) => arr.get_str(self.offset + i),
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical64(arr)) => arr.get_str(self.offset + i),
            _ => None,
        }
    }

    /// Returns the string value at logical index `i` within the window.
    ///
    /// # Safety
    /// Skips bounds checks, but will still return `None` if null.
    #[inline]
    pub unsafe fn get_str_unchecked(&self, i: usize) -> Option<&str> {
        match &self.array {
            Array::TextArray(TextArray::String32(arr)) => {
                if arr.is_null(self.offset + i) {
                    None
                } else {
                    Some(unsafe { arr.get_str_unchecked(self.offset + i) })
                }
            }
            #[cfg(feature = "large_string")]
            Array::TextArray(TextArray::String64(arr)) => {
                if arr.is_null(self.offset + i) {
                    None
                } else {
                    Some(unsafe { arr.get_str_unchecked(self.offset + i) })
                }
            }
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical8(arr)) => {
                if arr.is_null(self.offset + i) {
                    None
                } else {
                    Some(unsafe { arr.get_str_unchecked(self.offset + i) })
                }
            }
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical16(arr)) => {
                if arr.is_null(self.offset + i) {
                    None
                } else {
                    Some(unsafe { arr.get_str_unchecked(self.offset + i) })
                }
            }
            Array::TextArray(TextArray::Categorical32(arr)) => {
                if arr.is_null(self.offset + i) {
                    None
                } else {
                    Some(unsafe { arr.get_str_unchecked(self.offset + i) })
                }
            }
            #[cfg(feature = "extended_categorical")]
            Array::TextArray(TextArray::Categorical64(arr)) => {
                if arr.is_null(self.offset + i) {
                    None
                } else {
                    Some(unsafe { arr.get_str_unchecked(self.offset + i) })
                }
            }
            _ => None,
        }
    }

    /// Returns a new window view into a sub-range of this view.
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(offset + len <= self.len, "ArrayView::slice: out of bounds");
        Self {
            array: self.array.clone(), // arc clone
            offset: self.offset + offset,
            len,
            null_count: OnceLock::new(),
        }
    }

    /// Materialise a deep copy as an owned `Array` for the window.
    #[inline]
    pub fn to_array(&self) -> Array {
        self.array.slice_clone(self.offset, self.len)
    }

    /// Returns a pointer and metadata for raw access
    ///
    /// This is not logical length - it is total raw bytes in the buffer,
    /// so for non-fixed width types such as bit-packed booleans
    /// or strings, please factor this in accordingly.
    #[inline]
    pub fn data_ptr_and_byte_len(&self) -> (*const u8, usize, usize) {
        let (ptr, _total_len, elem_size) = self.array.data_ptr_and_byte_len();
        let windowed_ptr = unsafe { ptr.add(self.offset * elem_size) };
        (windowed_ptr, self.len, elem_size)
    }

    /// Returns the exclusive end index of the window (relative to parent array).
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Returns the underlying window as a tuple: (Array, offset, len).
    ///
    /// Note: This clones the Arc-wrapped Array.
    #[inline]
    pub fn as_tuple(&self) -> (Array, usize, usize) {
        (self.array.clone(), self.offset, self.len) // arc clone
    }

    /// Returns a reference tuple: (&Array, offset, len).
    ///
    /// This avoids cloning the Arc and returns a reference with a lifetime
    /// tied to this ArrayV.
    #[inline]
    pub fn as_tuple_ref(&self) -> (&Array, usize, usize) {
        (&self.array, self.offset, self.len)
    }

    /// Returns the null count in the window, caching the result after first calculation.
    #[inline]
    pub fn null_count(&self) -> usize {
        *self
            .null_count
            .get_or_init(|| match self.array.null_mask() {
                Some(mask) => mask.view(self.offset, self.len).count_zeros(),
                None => 0,
            })
    }

    /// Returns a windowed view over the underlying null mask, if any.
    #[inline]
    pub fn null_mask_view(&self) -> Option<BitmaskV> {
        self.array
            .null_mask()
            .map(|mask| mask.view(self.offset, self.len))
    }

    /// Set the cached null count (advanced use only).
    ///
    /// Returns Ok(()) if the value was set, or Err(count) if it was already initialized.
    /// This is thread-safe and can only succeed once per ArrayV instance.
    #[inline]
    pub fn set_null_count(&self, count: usize) -> Result<(), usize> {
        self.null_count.set(count).map_err(|_| count)
    }
}

/// Array -> ArrayView
///
/// Uses Offset 0 and length self.len()
impl From<Array> for ArrayV {
    fn from(array: Array) -> Self {
        let len = array.len();
        ArrayV {
            array,
            offset: 0,
            len,
            null_count: OnceLock::new(),
        }
    }
}

/// FieldArray -> ArrayView
///
/// Takes self.array then offset 0, length self.len())
impl From<FieldArray> for ArrayV {
    fn from(field_array: FieldArray) -> Self {
        let len = field_array.len();
        ArrayV {
            array: field_array.array,
            offset: 0,
            len,
            null_count: OnceLock::new(),
        }
    }
}

/// NumericArrayView -> ArrayView
///
/// Converts via NumericArrayV.into() -> Array -> ArrayV
#[cfg(feature = "views")]
impl From<crate::NumericArrayV> for ArrayV {
    fn from(numeric_view: crate::NumericArrayV) -> Self {
        numeric_view.into()
    }
}

/// TextArrayView -> ArrayView
///
/// Converts via TextArrayV.into() -> Array -> ArrayV
#[cfg(feature = "views")]
impl From<crate::TextArrayV> for ArrayV {
    fn from(text_view: crate::TextArrayV) -> Self {
        text_view.into()
    }
}

/// TemporalArrayView -> ArrayView
///
/// Converts via TemporalArrayV.into() -> Array -> ArrayV
#[cfg(all(feature = "views", feature = "datetime"))]
impl From<crate::TemporalArrayV> for ArrayV {
    fn from(temporal_view: crate::TemporalArrayV) -> Self {
        temporal_view.into()
    }
}

// We do not implement `Index` as `ArrayView` cannot safely return
// a reference to an element.

impl Debug for ArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArrayView")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("array", &self.array)
            .field("cached_null_count", &self.null_count.get())
            .finish()
    }
}

impl Display for ArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let nulls = self.null_count();
        let head_len = self.len.min(MAX_PREVIEW);

        writeln!(
            f,
            "ArrayView [{} values] (offset: {}, nulls: {})",
            self.len, self.offset, nulls
        )?;

        // Take a view into the head_len elements
        let display_view = Self {
            array: self.array.clone(), // arc clone
            offset: self.offset,
            len: head_len,
            null_count: OnceLock::new(), // Create new lock for this view
        };

        // Delegate to the inner array's Display
        for line in format!("{display_view}").lines() {
            writeln!(f, "  {line}")?;
        }

        if self.len > MAX_PREVIEW {
            writeln!(f, "  ... ({} more rows)", self.len - MAX_PREVIEW)?;
        }

        Ok(())
    }
}

impl Shape for ArrayV {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Concatenate for ArrayV {
    /// Concatenates two array views by materializing both to owned arrays,
    /// concatenating them, and wrapping the result back in a view.
    ///
    /// # Notes
    /// - This operation copies data from both views to create owned arrays.
    /// - The resulting view has offset=0 and length equal to the combined length.
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Materialize both views to owned arrays
        let self_array = self.to_array();
        let other_array = other.to_array();

        // Concatenate the owned arrays
        let concatenated = self_array.concat(other_array)?;

        // Wrap the result in a new view
        Ok(ArrayV::from(concatenated))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{Array, Bitmask, IntegerArray, NumericArray, vec64};

    #[test]
    fn test_array_view_basic_indexing_and_slice() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(11);
        arr.push(22);
        arr.push(33);
        arr.push(44);

        let array = Array::NumericArray(NumericArray::Int32(Arc::new(arr)));
        let view = ArrayV::new(array, 1, 2);

        // Basic indexing within window
        assert_eq!(view.len(), 2);
        assert_eq!(view.offset, 1);
        assert_eq!(view.get::<IntegerArray<i32>>(0), Some(22));
        assert_eq!(view.get::<IntegerArray<i32>>(1), Some(33));
        assert_eq!(view.get::<IntegerArray<i32>>(2), None);

        // Slicing the view produces the correct sub-window
        let sub = view.slice(1, 1);
        assert_eq!(sub.len(), 1);
        assert_eq!(sub.get::<IntegerArray<i32>>(0), Some(33));
        assert_eq!(sub.get::<IntegerArray<i32>>(1), None);
    }

    #[test]
    fn test_array_view_null_count_and_cache() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        arr.push(3);
        arr.push(4);

        // Null mask: only index 2 is null
        let mut mask = Bitmask::new_set_all(4, true);
        mask.set(2, false);
        arr.null_mask = Some(mask);

        let array = Array::NumericArray(NumericArray::Int32(Arc::new(arr)));

        let view = ArrayV::new(array, 0, 4);
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
    fn test_array_view_with_supplied_null_count() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(5);
        arr.push(6);

        let array = Array::NumericArray(NumericArray::Int32(Arc::new(arr)));
        let view = ArrayV::with_null_count(array, 0, 2, 99);
        // Should always report the supplied cached value
        assert_eq!(view.null_count(), 99);
        // Trying to set again should fail since it's already initialized
        assert!(view.set_null_count(101).is_err());
        // Still returns original value
        assert_eq!(view.null_count(), 99);
    }

    #[test]
    fn test_array_view_to_array_and_as_tuple() {
        let mut arr = IntegerArray::<i32>::default();
        for v in 10..20 {
            arr.push(v);
        }
        let array = Array::NumericArray(NumericArray::Int32(Arc::new(arr)));
        let view = ArrayV::new(array.clone(), 4, 3);
        let arr2 = view.to_array();
        // Copy should be [14, 15, 16]
        if let Array::NumericArray(NumericArray::Int32(a2)) = arr2 {
            assert_eq!(a2.data, vec64![14, 15, 16]);
        } else {
            panic!("Unexpected variant");
        }

        // as_tuple returns correct metadata
        let tup = view.as_tuple();
        assert_eq!(&tup.0, &array);
        assert_eq!(tup.1, 4);
        assert_eq!(tup.2, 3);
    }

    #[test]
    fn test_array_view_null_mask_view() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(2);
        arr.push(4);
        arr.push(6);

        let mut mask = Bitmask::new_set_all(3, true);
        mask.set(0, false);
        arr.null_mask = Some(mask);

        let array = Array::NumericArray(NumericArray::Int32(Arc::new(arr)));
        let view = ArrayV::new(array.clone(), 1, 2);
        let mask_view = view.null_mask_view().expect("Should have mask");
        assert_eq!(mask_view.len(), 2);
        // Should map to bits 1 and 2 of original mask
        assert!(mask_view.get(0));
        assert!(mask_view.get(1));
    }
}
