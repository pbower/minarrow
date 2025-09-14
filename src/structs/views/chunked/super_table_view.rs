//! # **SuperTableView Module** - *Chunked, Windowed View over 1:M Tables*
//!
//! `SuperTableV` is a **borrowed, zero-copy view** over a row range
//! `[offset .. offset + len)` from a [`SuperTable`], potentially spanning
//! multiple underlying `Table` batches.
//!
//! ## Role
//! - Similar in structure to [`SuperTable`], but semantically a *windowed view*
//!   over one **logical table** instead of an owned multi-batch table.
//! - Represents a *mini-batch* within one `Table` (or one batch of a `SuperTable`).
//! - Useful when cached statistics (e.g. null counts) are tied to batch or region
//!   boundaries and should be preserved while slicing.
//!
//! ## Interop
//! - Built by higher-level slicing APIs (`SuperTable::view` / `SuperTable::slice`).
//! - Can be materialised into a single contiguous [`Table`] with
//!   [`to_table`](SuperTableV::to_table).
//!
//! ## Features
//! - Zero-copy traversal of the constituent [`TableV`] slices via
//!   [`chunks`](SuperTableV::chunks) / [`iter`](SuperTableV::iter).
//! - Random access to a single row: [`row`](SuperTableV::row) or
//!   [`row_slice`](SuperTableV::row_slice).
//! - Sub-windowing into another borrowed view with [`slice`](SuperTableV::slice).
//!
//! ## Performance Notes
//! - Iterating across slices is efficient but may span non-contiguous memory.
//!   For dense, contiguous runs, consider materialising with [`to_table`](SuperTableV::to_table).
//!
//! ## Invariants
//! - `len` is the logical number of rows in this view.
//! - `slices` are ordered, non-overlapping, and each covers a contiguous region
//!   within its underlying table batch.

use crate::structs::chunked::super_table::SuperTable;
use crate::traits::shape::Shape;
use crate::enums::shape_dim::ShapeDim;
use crate::{Table, TableV};

/// # SuperTableView
///
/// Borrowed view over `[offset .. offset + len)` rows from a [`SuperTable`],
/// potentially spanning multiple table batches (`TableV` slices).
///
/// ## Purpose
/// - Mini-batch representation of a single logical `Table` or `SuperTable` batch.
/// - Allows preserving cached metadata (e.g. null counts) across multiple regions
///   without merging or cloning data.
///
/// ## Fields
/// - `slices`: constituent [`TableV`] slices covering the window.
/// - `len`: logical row count of this view.
///
/// ## Notes
/// - Use [`chunks`](Self::chunks) / [`iter`](Self::iter) for slice-level access.
/// - Random row access via [`row`](Self::row) / [`row_slice`](Self::row_slice).
/// - Create smaller windows via [`slice`](Self::slice) without data copies.
#[derive(Debug, Clone)]
pub struct SuperTableV {
    pub slices: Vec<TableV>,
    pub len: usize
}

impl SuperTableV {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn n_slices(&self) -> usize {
        self.slices.len()
    }

    /// Iterator over slice‐level `TableSlice`s.
    #[inline]
    pub fn chunks(&self) -> impl Iterator<Item = &TableV> {
        self.slices.iter()
    }

    /// Copy-owning materialisation into one contiguous `Table`.
    pub fn to_table(&self, name: Option<&str>) -> Table {
        SuperTable::from_views(&self.slices, "".to_string()).to_table(name)
    }

    /// Returns a sub-window of this chunked slice object as a new ChunkedTableSlice.
    #[inline]
    pub fn slice(&self, mut offset: usize, mut len: usize) -> Self {
        assert!(offset + len <= self.len, "slice out of bounds");
        let mut slices = Vec::new();
        let requested_len = len;
        for slice in &self.slices {
            if offset >= slice.len {
                offset -= slice.len;
                continue;
            }
            let take = (slice.len - offset).min(len);
            slices.push(slice.from_self(offset, take));
            len -= take;
            if len == 0 {
                break;
            }
            offset = 0;
        }
        SuperTableV { slices: slices, len: requested_len }
    }

    /// Random-access a single row (as a zero-copy TableSlice of length 1).
    #[inline]
    pub fn row(&self, mut row_idx: usize) -> TableV {
        for slice in &self.slices {
            if row_idx < slice.len {
                return slice.from_self(row_idx, 1);
            }
            row_idx -= slice.len;
        }
        panic!("row index out of bounds");
    }

    /// Iterate through all `TableSlice`s in this slice.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = TableV> + '_ {
        self.slices.iter().cloned()
    }

    /// Locate (chunked_idx,row_inside_chunked) for a global row index.
    #[inline]
    fn locate(&self, row: usize) -> (usize, usize) {
        assert!(row < self.len, "row out of bounds");
        let mut acc = 0;
        for (ci, p) in self.slices.iter().enumerate() {
            if row < acc + p.len {
                return (ci, row - acc);
            }
            acc += p.len;
        }
        unreachable!()
    }

    /// Return a zero-copy TableSlice of exactly one row by global row index.
    #[inline]
    pub fn row_slice(&self, row: usize) -> TableV {
        let (ci, ri) = self.locate(row);
        self.slices[ci].from_self(ri, 1)
    }
 
    /// Returns the total number of rows in the Super table across all chunks
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.len
    }

    /// Returns the number of columns in the Super table.
    /// 
    /// Assumes that every chunk has the same column schema as per
    /// the semantic requirement.
    #[inline]
    pub fn n_cols(&self) -> usize {
        let n_batches = self.slices.len();
        if n_batches > 0 {
           self.slices[0].fields.len()
        } else {
            0
        }
    }
}

impl Shape for SuperTableV {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank2 { rows: self.n_rows(), cols: self.n_cols() }
    }
}

#[cfg(feature = "views")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{Array, Field, FieldArray, IntegerArray, NumericArray, Table};

    /// Build a `FieldArray` containing an Int32 column with the given values
    fn fa_i32(name: &str, vals: &[i32]) -> FieldArray {
        let arr = Array::from_int32(IntegerArray::<i32>::from_slice(vals));
        let field = Field::new(name, ArrowType::Int32, false, None);
        FieldArray::new(field, arr)
    }

    /// One-column `Table` with Int32 data
    fn table(name: &str, vals: &[i32]) -> Table {
        Table {
            cols: Vec::from(vec![fa_i32(name, vals)]),
            n_rows: vals.len(),
            name: name.to_string()
        }
    }
    /// Handy lens into the first column of a 1-column table
    fn col_vals(t: &Table) -> Vec<i32> {
        if let Array::NumericArray(NumericArray::Int32(a)) = &t.cols[0].array {
            a.data.as_slice().to_vec()
        } else {
            unreachable!()
        }
    }

    #[test]
    fn slice_basic_properties() {
        // batch 1: 2 rows; batch 2: 3 rows
        let b1 = table("t", &[1, 2]);
        let b2 = table("t", &[3, 4, 5]);

        let big_slice = SuperTableV {
            slices: vec![TableV::from_table(b1, 0, 2), TableV::from_table(b2, 0, 3)],
            len: 5
        };

        assert!(!big_slice.is_empty());
        assert_eq!(big_slice.n_slices(), 2);
        assert_eq!(big_slice.len, 5);

        // materialise
        let tbl = big_slice.to_table(Some("merged"));
        assert_eq!(tbl.n_rows, 5);
        assert_eq!(col_vals(&tbl), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn subslice_and_row_access() {
        let big = table("x", &[10, 11, 12, 13, 14]);
        let full = SuperTableV {
            slices: vec![TableV::from_table(big, 0, 5)],
            len: 5
        };

        // Sub-slice  [1 .. 4)  =>  rows 11,12,13
        let sub = full.slice(1, 3);
        assert_eq!(sub.len, 3); // The slice window length, not parent len
        let sub_tbl = sub.to_table(None);
        assert_eq!(col_vals(&sub_tbl), vec![11, 12, 13]);

        // row() — convert row TableSlice to Table, then inspect row 0
        let row_ts = sub.row(1);
        assert_eq!(row_ts.n_rows(), 1);
        let row_tbl = row_ts.to_table();
        assert_eq!(col_vals(&row_tbl)[0], 12);

        // row_slice() — convert to Table, then inspect row 0
        let single = sub.row_slice(2);
        assert_eq!(single.len, 1);
        let single_tbl = single.to_table();
        assert_eq!(col_vals(&single_tbl)[0], 13);
    }

    #[test]
    fn iterate_slices_and_iters() {
        let b1 = table("c", &[0, 1]);
        let b2 = table("c", &[2]);
        let slice = SuperTableV {
            slices: vec![TableV::from_table(b1, 0, 2), TableV::from_table(b2, 0, 1)],
            len: 3
        };

        // chunks()
        let pcs: Vec<_> = slice.chunks().collect();
        assert_eq!(pcs.len(), 2);
        assert_eq!(pcs[0].n_rows(), 2);
        assert_eq!(pcs[1].n_rows(), 1);

        // iter()
        let collected: Vec<_> = slice.iter().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].offset, 0);
    }

    #[test]
    #[should_panic(expected = "row index out of bounds")]
    fn row_oob_panics() {
        let t = table("q", &[1, 2]);
        let slice = SuperTableV {
            slices: vec![TableV::from_table(t, 0, 2)],
            len: 2
        };
        // This should panic
        let _ = slice.row(5);
    }

    #[test]
    #[should_panic(expected = "slice out of bounds")]
    fn subslice_oob_panics() {
        let t = table("p", &[1, 2, 3]);
        let slice = SuperTableV {
            slices: vec![TableV::from_table(t, 0, 3)],
            len: 3
        };
        // slice end exceeds original
        let _ = slice.slice(2, 5);
    }
}
