use std::cell::Cell;
use std::fmt::{self, Debug, Display, Formatter};

use crate::traits::print::MAX_PREVIEW;
use crate::{Array, BitmaskV, FieldArray, MaskedArray, TextArray};

/// Logical, windowed view over an `Array`.
///
/// This is used to return an indexable view over a subset of the array.
/// Additionally, it can be used to cache null counts for those regions,
/// which can be used to speed up calculations.
///
/// ### Behaviour
/// - Indices are always relative to the window.
/// - Holds a reference to the original `Array` and window bounds.
/// - Windowing uses an arc clone
/// - All access (get/index, etc.) is offset-correct and bounds-checked.
/// - Null count is computed once (on demand or at creation) and cached for subsequent use.
#[derive(Clone, PartialEq)]
pub struct ArrayV {
    pub array: Array, // contains Arc<inner>
    pub offset: usize,
    len: usize,
    null_count: Cell<Option<usize>>
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
            null_count: Cell::new(None)
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
        Self {
            array,
            offset,
            len,
            null_count: Cell::new(Some(null_count))
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
            _ => None
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
            _ => None
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
            null_count: Cell::new(None)
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

    /// Returns the underlying window as a tuple: (&Array, offset, len).
    #[inline]
    pub fn as_tuple(&self) -> (Array, usize, usize) {
        (self.array.clone(), self.offset, self.len) // arc clone
    }

    /// Returns the null count in the window, caching the result after first calculation.
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

    /// Returns a windowed view over the underlying null mask, if any.
    #[inline]
    pub fn null_mask_view(&self) -> Option<BitmaskV> {
        self.array.null_mask().map(|mask| mask.to_window(self.offset, self.len))
    }

    /// Set the cached null count (advanced use only; not thread-safe if mutated after use).
    #[inline]
    pub fn set_null_count(&self, count: usize) {
        self.null_count.set(Some(count));
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
            null_count: Cell::new(None)
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
            null_count: Cell::new(None)
        }
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
            null_count: self.null_count.clone(),
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
        view.set_null_count(101);
        assert_eq!(view.null_count(), 101);
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
