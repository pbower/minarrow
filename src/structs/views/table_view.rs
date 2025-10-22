//! # **TableV Module** - *Windowed View over a Table*
//!
//! `TableV` is a **row-aligned, zero-copy view** over a `Table`, holding the
//! window `[offset .. offset+len)` across all columns.
//!
//! ## Purpose
//! - Stream or batch rows without copying full columns.
//! - Do per-partition / per-window work with stable schema metadata (`Field`s).
//! - Glue for chunked processing alongside `SuperTableV` and `ArrayV`, but focused
//! on windowed zero-copy access within a single Table batch.
//!
//! ## Behaviour
//! - Columns are exposed as `ArrayV` windows with the same `(offset, len)`.
//! - Schema (`Field`s) is retained and shared; column data is not copied.
//! - Slicing a `TableV` (`from_self`) is O(1) — metadata-only.
//! - `to_table()` materialises an owned `Table` copy of the window.
//!
//! ## When to use
//! - Sliding windows, micro-batching, streaming sinks/sources.
//! - Thread-local views where each worker consumes a row range.
//! - Building higher-level chunked tables (`SuperTableV`) from base tables.
//!
//! ## Example
//! ```rust
//! # use minarrow::{Table, TableV, Array, IntegerArray, FieldArray};
//! # use std::sync::Arc;
//! // Build a simple 2-column table
//! let a = FieldArray::from_arr("a", Array::from_int32(IntegerArray::<i32>::from_slice(&[1,2,3,4,5])));
//! let b = FieldArray::from_arr("b", Array::from_int32(IntegerArray::<i32>::from_slice(&[10,20,30,40,50])));
//! let mut tbl = Table::new("T".to_string(), vec![a,b].into());
//!
//! // View rows 1..4 (3 rows total)
//! let tv = TableV::from_table(tbl, 1, 3);
//! assert_eq!(tv.n_rows(), 3);
//! assert_eq!(tv.n_cols(), 2);
//! // Access a column window
//! let col0 = tv.col(0).unwrap();
//! assert_eq!(col0.get::<minarrow::IntegerArray<i32>>(0), Some(2));
//!
//! // Materialise the window as an owned Table copy
//! let owned = tv.to_table();
//! assert_eq!(owned.n_rows, 3);
//! ```

use std::fmt::{Display, Formatter};
use std::sync::Arc;

#[cfg(feature = "views")]
use crate::ArrayV;
use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::traits::concatenate::Concatenate;
use crate::traits::print::MAX_PREVIEW;
use crate::traits::shape::Shape;
use crate::{Field, FieldArray, Table};

/// # TableView
///
/// Row-aligned view into a `Table` over `[offset .. offset+len)`.
///
/// ## Fields
/// - `name`: table name - propagated from the parent `Table`.
/// - `fields`: shared schema - `Arc<Field>` per column.
/// - `cols`: column windows as `ArrayV`, all with the same `(offset, len)`.
/// - `offset`: starting row, relative to parent table.
/// - `len`: number of rows in the window.
///
/// ## Notes
/// - Construction from a `Table`/`Arc<Table>` is zero-copy for column data.
/// - Use `from_self` to take sub-windows cheaply.
/// - Use `to_table()` to materialise an owned copy of just this window.
/// - Column helpers (`col`, `col_by_name`, `col_window`) provide ergonomic access.
#[derive(Debug, Clone, PartialEq)]
pub struct TableV {
    /// Table name
    pub name: String,
    /// Fields
    pub fields: Vec<Arc<Field>>,
    /// Column slices (field metadata + windowed array)
    pub cols: Vec<ArrayV>,
    /// Row offset from start of parent table
    pub offset: usize,
    /// Length of slice (in rows)
    pub len: usize,
}

impl TableV {
    /// Creates a new `TableView` over `table[offset .. offset+len)`.
    /// Provides a non-owning view into a subrange of the table.
    #[inline]
    pub fn from_table(table: Table, offset: usize, len: usize) -> Self {
        let mut fields = Vec::with_capacity(table.cols.len());
        let mut cols = Vec::with_capacity(table.cols.len());

        for fa in &table.cols {
            fields.push(fa.field.clone());
            cols.push(ArrayV::new(fa.array.clone(), offset, len));
        }

        Self {
            name: table.name.clone(),
            fields,
            cols,
            offset,
            len,
        }
    }

    /// Creates a new `TableView` over `table[offset .. offset+len)`.
    /// Provides a non-owning view into a subrange of the table.
    #[inline]
    pub fn from_arc_table(table: Arc<Table>, offset: usize, len: usize) -> Self {
        let mut fields = Vec::with_capacity(table.cols.len());
        let mut cols = Vec::with_capacity(table.cols.len());

        for fa in &table.cols {
            fields.push(fa.field.clone());
            cols.push(ArrayV::new(fa.array.clone(), offset, len));
        }

        Self {
            name: table.name.clone(),
            fields,
            cols,
            offset,
            len,
        }
    }

    /// Derives a subwindow from this `TableView`, adjusted by `offset` and `len`.
    #[inline]
    pub fn from_self(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= self.len,
            "TableView::from_self: slice out of bounds"
        );

        let mut fields = Vec::with_capacity(self.cols.len());
        let mut cols = Vec::with_capacity(self.cols.len());

        for (field, array_window) in self.fields.iter().zip(self.cols.iter()) {
            let w = array_window.as_tuple();
            fields.push(field.clone());
            cols.push(ArrayV::new(
                w.0,          // &Array
                w.1 + offset, // adjusted offset
                len,          // subwindow length
            ));
        }

        TableV {
            name: self.name.clone(),
            fields,
            cols,
            offset: self.offset + offset,
            len,
        }
    }

    /// Returns true if the window contains no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the exclusive end row index of the window.
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Returns the number of columns in the table window.
    #[inline]
    pub fn n_cols(&self) -> usize {
        self.cols.len()
    }

    /// Returns the number of rows in the window.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.len
    }

    /// Returns the name of the table.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the column window at the given index.
    #[inline]
    pub fn col(&self, idx: usize) -> Option<&ArrayV> {
        self.cols.get(idx).map(|v| &*v)
    }

    /// Returns a slice of all column windows.
    #[inline]
    pub fn cols(&self) -> Vec<&ArrayV> {
        self.cols.iter().map(|av| &*av).collect()
    }

    /// Returns an iterator over all column names.
    #[inline]
    pub fn col_names(&self) -> impl Iterator<Item = &str> {
        self.fields.iter().map(|f| f.name.as_str())
    }

    /// Returns the index of a column by name.
    #[inline]
    pub fn col_index(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    /// Returns the window tuple of the column at the given index.
    #[inline]
    pub fn col_window(&self, idx: usize) -> Option<ArrayV> {
        self.cols.get(idx).map(|av| {
            let (array, offset, len) = &av.as_tuple();
            ArrayV::new(array.clone(), *offset, *len)
        })
    }

    /// Returns the name of the column at the given index.
    #[inline]
    pub fn col_name(&self, idx: usize) -> Option<&str> {
        self.fields.get(idx).map(|f| f.name.as_str())
    }

    /// Returns a `Vec<bool>` indicating which columns are nullable.
    #[inline]
    pub fn cols_nullable(&self) -> Vec<bool> {
        self.fields.iter().map(|f| f.nullable).collect()
    }

    /// Returns a reference to the column window by column name.
    #[inline]
    pub fn col_by_name(&self, name: &str) -> Option<&ArrayV> {
        self.col_index(name).map(|i| &self.cols[i])
    }

    /// Consumes the TableView, producing an owned Table with the sliced data.
    /// Copies the data.
    pub fn to_table(&self) -> Table {
        let cols: Vec<_> = self
            .fields
            .iter()
            .zip(self.cols.iter())
            .map(|(field, window)| {
                let w = window.as_tuple();
                let sliced = w.0.slice_clone(w.1, w.2);
                let null_count = sliced.null_count();
                FieldArray {
                    field: field.clone(),
                    array: sliced,
                    null_count,
                }
            })
            .collect();

        Table {
            cols,
            n_rows: self.len,
            name: self.name.clone(),
        }
    }

    /// Converts a column window into an owned `FieldArray`, slicing the array and copying data.
    /// Copies the data.
    pub fn extract_column(field: &Field, window: &ArrayV) -> FieldArray {
        let w = window.as_tuple();
        let sliced = w.0.slice_clone(w.1, w.2);
        let null_count = sliced.null_count();
        FieldArray {
            field: field.clone().into(),
            array: sliced,
            null_count,
        }
    }
}

impl Display for TableV {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let n_rows = self.n_rows();
        let n_cols = self.n_cols();
        let col_names: Vec<&str> = self.col_names().collect();

        writeln!(
            f,
            "TableView '{}' [{} rows × {} cols]",
            self.name, n_rows, n_cols
        )?;

        // Header
        write!(f, "  ")?;
        for name in &col_names {
            write!(f, "{:<16}", name)?;
        }
        writeln!(f)?;

        // Rows
        let display_rows = n_rows.min(MAX_PREVIEW);
        for row_idx in 0..display_rows {
            write!(f, "  ")?;
            for col in &self.cols {
                match col.get_str(row_idx) {
                    Some(s) => write!(f, "{:<16}", s)?,
                    None => write!(f, "{:<16}", "·")?,
                }
            }
            writeln!(f)?;
        }

        if n_rows > MAX_PREVIEW {
            writeln!(f, "  ... ({} more rows)", n_rows - MAX_PREVIEW)?;
        }

        Ok(())
    }
}

impl Shape for TableV {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank2 {
            rows: self.n_rows(),
            cols: self.n_cols(),
        }
    }
}

impl Concatenate for TableV {
    /// Concatenates two table views by materializing both to owned tables,
    /// concatenating them, and wrapping the result back in a view.
    ///
    /// # Notes
    /// - This operation copies data from both views to create owned tables.
    /// - The resulting view has offset=0 and length equal to the combined length.
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Materialize both views to owned tables
        let self_table = self.to_table();
        let other_table = other.to_table();

        // Concatenate the owned tables
        let concatenated = self_table.concat(other_table)?;

        // Wrap the result in a new view
        Ok(TableV::from(concatenated))
    }
}

// From implementations for conversion between Table and TableV

/// Table -> TableV conversion
impl From<Table> for TableV {
    fn from(table: Table) -> Self {
        let fields: Vec<Arc<Field>> = table.cols.iter().map(|fa| fa.field.clone()).collect();

        let cols: Vec<ArrayV> = table.cols.into_iter().map(|fa| ArrayV::from(fa)).collect();

        TableV {
            name: table.name,
            fields,
            cols,
            offset: 0,
            len: table.n_rows,
        }
    }
}

/// TableV -> Table conversion
impl From<TableV> for Table {
    fn from(view: TableV) -> Self {
        let field_arrays: Vec<FieldArray> = view
            .cols
            .into_iter()
            .enumerate()
            .map(|(i, array_v)| {
                let field = if i < view.fields.len() {
                    (*view.fields[i]).clone()
                } else {
                    Field::new(format!("col_{}", i), array_v.array.arrow_type(), true, None)
                };

                // If the view is windowed, we need to materialize the slice
                let array = if view.offset > 0 || view.len < array_v.len() {
                    // Need to slice the array - use the existing slice method
                    array_v.slice(0, view.len).array
                } else {
                    array_v.array
                };

                FieldArray {
                    field: Arc::new(field),
                    array: array.clone(),
                    null_count: array.null_count(),
                }
            })
            .collect();

        Table::new(view.name, Some(field_arrays))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::structs::field::Field;
    use crate::structs::field_array::FieldArray;
    use crate::structs::table::Table;
    use crate::{Array, IntegerArray};

    #[test]
    fn test_table_slice_from_table_and_access() {
        let field_a = Field::new("a", ArrowType::Int32, false, None);
        let arr_a = Array::from_int32(IntegerArray::from_slice(&[1, 2, 3, 4, 5]));
        let fa_a = FieldArray::new(field_a, arr_a);

        let field_b = Field::new("b", ArrowType::Int32, false, None);
        let arr_b = Array::from_int32(IntegerArray::from_slice(&[10, 20, 30, 40, 50]));
        let fa_b = FieldArray::new(field_b, arr_b);

        let mut tbl = Table::new_empty();
        tbl.add_col(fa_a);
        tbl.add_col(fa_b);
        tbl.name = "TestTable".to_string();

        // Slice rows 1..4 (offset 1, len 3)
        let slice = TableV::from_table(tbl, 1, 3);

        assert_eq!(slice.offset, 1);
        assert_eq!(slice.len, 3);
        assert_eq!(slice.n_rows(), 3);
        assert_eq!(slice.n_cols(), 2);
        assert_eq!(slice.name(), "TestTable");
        assert_eq!(slice.fields[0].name, "a");
        assert_eq!(slice.fields[1].name, "b");
        assert_eq!(slice.col_names().collect::<Vec<_>>(), vec!["a", "b"]);
        assert_eq!(slice.col_index("b"), Some(1));
        assert!(slice.col_by_name("a").is_some());
        assert!(slice.col_by_name("nonexistent").is_none());
        assert!(!slice.is_empty());
        assert_eq!(slice.end(), 4);
    }

    #[test]
    fn test_table_slice_empty() {
        let tbl = Table::new_empty();
        let slice = TableV::from_table(tbl, 0, 0);
        assert_eq!(slice.n_cols(), 0);
        assert_eq!(slice.n_rows(), 0);
        assert!(slice.is_empty());
    }
}
