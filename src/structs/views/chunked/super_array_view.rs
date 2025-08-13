//! # SuperArrayView Module
//!
//! `SuperArrayV` is a **borrowed, chunked view** over a single logical array,
//! exposing an arbitrary `[offset .. offset + len)` window that may span
//! multiple underlying chunks. It presents those chunks as one continuous
//! logical range without copying the underlying memory.
//!
//! ## Role
//! - Represents a *mini-batch* window for one `Array` (or one entry inside a
//!   `SuperArray`), useful for streaming, batching, or region-wise processing.
//! - Similar in shape to `SuperArray`, but semantically a **view** over a single
//!   column’s region rather than a bag of heterogeneous columns.
//!
//! ## Interop
//! - Constructed by higher-level chunked containers (`SuperArray::slice`).
//! - Can be materialised to a contiguous `Array` via [`SuperArrayV::copy_to_array`].
//! - Preserves schema via an `Arc<Field>`; no schema cloning per row.
//!
//! ## Features
//! - Zero-copy iteration over chunks: [`chunks`](SuperArrayV::chunks) / [`iter`](SuperArrayV::iter).
//! - Row-wise logical iteration: [`iter_rows`](SuperArrayV::iter_rows) and
//!   random access via [`row_slice`](SuperArrayV::row_slice) / [`get_value`](SuperArrayV::get_value).
//! - Sub-windowing: [`slice`](SuperArrayV::slice) returns another borrowed view.
//!
//! ## Performance Notes
//! - Iterating rows may touch non-contiguous memory if the window crosses chunk
//!   boundaries. For hot loops, prefer contiguous runs or materialise with
//!   `copy_to_array()`.
//!
//! ## Invariants
//! - `len` is the logical row count of this view.
//! - `slices` are ordered, non-overlapping, and cover at most `len` rows.
//! - `field` is the schema for the underlying array and is shared by all slices.
use std::sync::Arc;

use crate::{Array, ArrayV, ArrayVT, Field, SuperArray};

/// # SuperArrayView
/// 
/// Borrowed view over an arbitrary `[offset .. offset+len)` window of a `ChunkedArray`.
/// The window may span multiple internal chunks, presenting them as a unified logical view.
///
/// ## Purpose
/// A mini-batch of **one** array (or one `SuperArray` entry). Handy when you’ve
/// cached null counts / stats over regions and want to operate on those regions
/// without materialising the whole column.
///
/// ## Fields
/// - `slices`: constituent `ArrayView` pieces spanning the window.
/// - `len`: total logical row count for this view.
/// - `field`: schema field associated with the array (shared).
///
/// ## Notes
/// - Use [`chunks`](Self::chunks) / [`iter`](Self::iter) to walk chunk pieces,
///   or [`iter_rows`](Self::iter_rows) to traverse row-by-row across chunks.
/// - For hot paths, prefer contiguous (memory) windows; otherwise consider
///   [`copy_to_array`](Self::copy_to_array) to materialise a single buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct SuperArrayV {
    pub slices: Vec<ArrayV>,
    pub len: usize,
    pub field: Arc<Field>
}

impl SuperArrayV {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn n_slices(&self) -> usize {
        self.slices.len()
    }

    /// Materialise one contiguous `Array`
    pub fn copy_to_array(&self) -> Array {
        SuperArray::from_slices(&self.slices, self.field.clone()).copy_to_array()
    }

    /// Iterator over the underlying `ArraySlice`s
    #[inline]
    pub fn chunks(&self) -> impl Iterator<Item = &ArrayV> {
        self.slices.iter()
    }

    /// Returns a sub-window of this chunked array view over `[offset .. offset+len)`.
    ///
    /// Produces a new `ChunkedArrayView` with updated slice metadata.
    /// The field metadata is preserved as-is. Underlying data is not cloned.
    pub fn slice(&self, mut offset: usize, mut len: usize) -> Self {
        assert!(offset + len <= self.len, "slice out of bounds");

        let mut slices = Vec::new();
        for array_view in &self.slices {
            let base_len = array_view.len();
            let base_offset = array_view.offset;
            if offset >= base_len {
                offset -= base_len;
                continue;
            }

            let take = (base_len - offset).min(len);
            slices.push(ArrayV::new(array_view.array.clone(), base_offset + offset, take));

            len -= take;
            if len == 0 {
                break;
            }
            offset = 0;
        }

        Self {
            slices,
            len: self.len,
            field: self.field.clone()
        }
    }

    /// Returns the 1-element `Array` value at the logical index.
    pub fn get_value(&self, mut idx: usize) -> Array {
        for slice in &self.slices {
            if idx < slice.len() {
                return slice.array.slice_clone(slice.offset + idx, 1);
            }
            idx -= slice.len();
        }
        panic!("index out of bounds");
    }

    /// Iterate over all ArraySliceâ€™s in this slice.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = ArrayV> + '_ {
        self.slices.iter().cloned()
    }

    /// Returns an iterator over all rows as 1-element `ArrayView`s.
    ///
    /// Allows walking across potentially chunked memory logically row-by-row.
    #[inline]
    pub fn iter_rows(&self) -> impl Iterator<Item = ArrayVT> + '_ {
        self.slices
            .iter()
            .flat_map(|slice| {
                let base_offset = slice.offset;
                (0..slice.len()).map(move |i| (&slice.array, base_offset + i, 1))
            })
            .take(self.len)
    }

    /// Maps a logical row index into the corresponding (slice_index, intra_row_offset) pair.
    #[inline]
    fn locate(&self, row: usize) -> (usize, usize) {
        assert!(row < self.len, "row out of bounds");
        let mut acc = 0;
        for (chunk_idx, slice) in self.slices.iter().enumerate() {
            if row < acc + slice.len() {
                return (chunk_idx, row - acc);
            }
            acc += slice.len();
        }
        unreachable!()
    }

    /// Returns a zero-copy 1-row `ArrayWindow` at the given logical row index.
    pub fn row_slice(&self, row: usize) -> ArrayV {
        let (ci, ri) = self.locate(row);
        let (array, base_offset, _) = self.slices[ci].as_tuple();
        ArrayV::new(array, base_offset + ri, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{FieldArray, NumericArray};

    // Test helper - creates a FieldArray for i32
    fn fa(name: &str, vals: &[i32]) -> FieldArray {
        let arr = Array::from_int32(crate::IntegerArray::<i32>::from_slice(vals));
        let field = Field::new(name, ArrowType::Int32, false, None);
        FieldArray::new(field, arr)
    }

    #[test]
    fn test_is_empty_and_n_pieces() {
        let f = Arc::new(Field::new("col", ArrowType::Int32, false, None));
        let empty = SuperArrayV {
            slices: Vec::new(),
            len: 0,
            field: f.clone()
        };
        assert!(empty.is_empty());
        assert_eq!(empty.n_slices(), 0);

        let arr = Array::from_int32(crate::IntegerArray::<i32>::from_slice(&[1, 2, 3]));
        let non_empty = SuperArrayV {
            slices: Vec::from(vec![ArrayV::new(arr, 0, 3)]),
            len: 3,
            field: f.clone()
        };
        assert!(!non_empty.is_empty());
        assert_eq!(non_empty.n_slices(), 1);
    }

    #[test]
    fn test_to_array_materialises_correctly() {
        let fa1 = fa("x", &[1, 2, 3]);
        let fa2 = fa("x", &[4, 5]);
        let ca = SuperArray::from_chunks(Vec::from(vec![fa1.clone(), fa2.clone()]));
        let slice = ca.slice(0, 5);

        let arr = slice.copy_to_array();
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
            assert_eq!(ints.data.as_slice(), &[1, 2, 3, 4, 5]);
        } else {
            panic!("unexpected type");
        }
    }

    #[test]
    fn test_slice_subslice() {
        let fa1 = fa("x", &[1, 2, 3]);
        let fa2 = fa("x", &[4, 5, 6, 7]);
        let ca = SuperArray::from_chunks(Vec::from(vec![fa1.clone(), fa2.clone()]));
        let slice = ca.slice(1, 5); // [2,3,4,5,6]
        let sub = slice.slice(1, 3); // [3,4,5]
        let arr = sub.copy_to_array();
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
            assert_eq!(ints.data.as_slice(), &[3, 4, 5]);
        } else {
            panic!("unexpected type");
        }
    }

    #[test]
    fn test_chunks_and_iter() {
        let fa1 = fa("y", &[10, 20]);
        let fa2 = fa("y", &[30]);
        let ca = SuperArray::from_chunks(Vec::from(vec![fa1.clone(), fa2.clone()]));
        let slice = ca.slice(0, 3);

        let collected: Vec<_> = slice.chunks().map(|c| c.as_tuple()).collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].2, 2);
        assert_eq!(collected[1].2, 1);

        let collected2: Vec<_> = slice.iter().map(|c| c.as_tuple()).collect();
        assert_eq!(collected2.len(), 2);
        assert_eq!(collected2[0].2, 2);
    }

    #[test]
    fn test_get_array_and_row_slice() {
        let fa1 = fa("z", &[7, 8]);
        let fa2 = fa("z", &[9]);
        let ca = SuperArray::from_chunks(Vec::from(vec![fa1.clone(), fa2.clone()]));
        let slice = ca.slice(0, 3);

        let arr = slice.get_value(1); // Should be 8
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
            assert_eq!(ints.data.as_slice(), &[8]);
        } else {
            panic!("unexpected type");
        }

        let arr2 = slice.get_value(2); // Should be 9
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr2 {
            assert_eq!(ints.data.as_slice(), &[9]);
        } else {
            panic!("unexpected type");
        }

        let row = slice.row_slice(2).as_tuple();
        assert_eq!(row.2, 1);
        let arr3 = row.0.slice_clone(row.1, row.2);
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr3 {
            assert_eq!(ints.data.as_slice(), &[9]);
        } else {
            panic!("unexpected type");
        }
    }

    #[test]
    fn test_iter_rows_unified() {
        let fa1 = fa("w", &[1, 2]);
        let fa2 = fa("w", &[3]);
        let ca = SuperArray::from_chunks(Vec::from(vec![fa1.clone(), fa2.clone()]));
        let slice = ca.slice(0, 3);

        let rows: Vec<_> = slice.iter_rows().collect();
        assert_eq!(rows.len(), 3);

        let vals: Vec<i32> = rows
            .iter()
            .map(|s| {
                let s = s;
                if let Array::NumericArray(NumericArray::Int32(ints)) = s.0.slice_clone(s.1, s.2) {
                    ints.data[0]
                } else {
                    panic!("not i32")
                }
            })
            .collect();
        assert_eq!(vals, vec![1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_get_array_oob_panics() {
        let fa1 = fa("a", &[1]);
        let ca = SuperArray::from_chunks(Vec::from(vec![fa1.clone()]));
        let slice = ca.slice(0, 1);
        // Should panic
        slice.get_value(5);
    }

    #[test]
    fn test_field_propagation() {
        let fa1 = fa("field", &[1, 2, 3]);
        let ca = SuperArray::from_chunks(Vec::from(vec![fa1.clone()]));
        let slice = ca.slice(0, 3);
        assert_eq!(slice.field.name, "field");
        let subslice = slice.slice(1, 2);
        assert_eq!(subslice.field.name, "field");
    }
}
