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
//! - Slicing a `TableV` (`from_self`) is O(1) - metadata-only.
//! - `to_table()` materialises an owned `Table` copy of the window.
//!
//! ## When to use
//! - Sliding windows, micro-batching, streaming sinks/sources.
//! - Thread-local views where each worker consumes a row range.
//! - Building higher-level chunked tables (`SuperTableV`) from base tables.
//!
//! ## Example
//! ```rust
//! use minarrow::{Table, TableV, Array, IntegerArray, FieldArray};
//! use minarrow::ColumnSelection;  // Re-exported at crate root
//! use std::sync::Arc;
//!
//! // Build a simple 2-column table
//! let a = FieldArray::from_arr("a", Array::from_int32(IntegerArray::<i32>::from_slice(&[1,2,3,4,5])));
//! let b = FieldArray::from_arr("b", Array::from_int32(IntegerArray::<i32>::from_slice(&[10,20,30,40,50])));
//! let mut tbl = Table::new("T".to_string(), vec![a,b].into());
//!
//! // View rows 1..4 (3 rows total)
//! let tv = TableV::from_table(tbl, 1, 3);
//! assert_eq!(tv.n_rows(), 3);
//! assert_eq!(tv.n_cols(), 2);
//! // Access a column window via ColumnSelection trait
//! let col0 = tv.col_ix(0).unwrap();
//! assert_eq!(col0.get::<minarrow::IntegerArray<i32>>(0), Some(2));
//!
//! // Materialise the window as an owned Table copy
//! let owned = tv.to_table();
//! assert_eq!(owned.n_rows, 3);
//! ```

use std::fmt::{Display, Formatter};
use std::sync::Arc;

#[cfg(feature = "select")]
use crate::Array;
#[cfg(feature = "views")]
use crate::ArrayV;
use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::traits::concatenate::Concatenate;
use crate::traits::print::{
    MAX_PREVIEW, print_ellipsis_row, print_header_row, print_rule, value_to_string,
};
#[cfg(feature = "select")]
use crate::traits::selection::{ColumnSelection, DataSelector, FieldSelector, RowSelection};
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
/// - `active_col_selection`: Optional column selection indices (for pandas-style `.c[...]` syntax).
/// - `active_row_selection`: Optional row selection indices (for pandas-style `.r[...]` syntax).
///
/// ## Notes
/// - Construction from a `Table`/`Arc<Table>` is zero-copy for column data.
/// - Use `from_self` to take sub-windows cheaply.
/// - Use `to_table()` to materialise an owned copy of just this window.
/// - Column helpers (`col`, `col_ix`, `col_vec`) provide ergonomic access.
/// - Active selections are transparently applied to all access methods.
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
        self.n_rows() == 0
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

    /// Returns the number of rows in the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the name of the table.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns an iterator over all column names.
    #[inline]
    pub fn col_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    /// Returns the index of a column by name.
    #[inline]
    pub fn col_name_index(&self, name: &str) -> Option<usize> {
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
    pub fn nullable_cols(&self) -> Vec<bool> {
        self.fields.iter().map(|f| f.nullable).collect()
    }

    /// Consumes the TableView, producing an owned Table with the sliced data.
    /// Copies the data.
    pub fn to_table(&self) -> Table {
        let col_indices: Vec<usize> = (0..self.cols.len()).collect();
        let cols: Vec<_> = col_indices
            .iter()
            .filter_map(|&col_idx| {
                let field = self.fields.get(col_idx)?;
                let window = self.cols.get(col_idx)?;
                let w = window.as_tuple();

                // The ArrayV window already contains the correct offset/len
                let sliced = w.0.slice_clone(w.1, w.2);
                let null_count = sliced.null_count();

                Some(FieldArray {
                    field: field.clone(),
                    array: sliced,
                    null_count,
                })
            })
            .collect();

        let n_rows = self.len;
        Table::build(cols, n_rows, self.name.clone())
    }

    /// Gather specific rows from an ArrayV window
    #[cfg(feature = "select")]
    fn gather_rows_from_window(&self, window: &ArrayV, row_indices: &[usize]) -> Option<Array> {
        use crate::{
            Array, BooleanArray, CategoricalArray, FloatArray, IntegerArray, MaskedArray,
            NumericArray, StringArray, TextArray,
        };
        #[cfg(feature = "datetime")]
        use crate::{DatetimeArray, TemporalArray};

        let result = match &window.array {
            Array::Null => return None,
            Array::NumericArray(num_arr) => match num_arr {
                NumericArray::Int32(_) => {
                    let mut new_arr = IntegerArray::<i32>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<i32>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_int32(new_arr)
                }
                NumericArray::Int64(_) => {
                    let mut new_arr = IntegerArray::<i64>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<i64>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_int64(new_arr)
                }
                NumericArray::UInt32(_) => {
                    let mut new_arr = IntegerArray::<u32>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<u32>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_uint32(new_arr)
                }
                NumericArray::UInt64(_) => {
                    let mut new_arr = IntegerArray::<u64>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<u64>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_uint64(new_arr)
                }
                NumericArray::Float32(_) => {
                    let mut new_arr = FloatArray::<f32>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<FloatArray<f32>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_float32(new_arr)
                }
                NumericArray::Float64(_) => {
                    let mut new_arr = FloatArray::<f64>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<FloatArray<f64>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_float64(new_arr)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(_) => {
                    let mut new_arr = IntegerArray::<i8>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<i8>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_int8(new_arr)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(_) => {
                    let mut new_arr = IntegerArray::<i16>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<i16>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_int16(new_arr)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(_) => {
                    let mut new_arr = IntegerArray::<u8>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<u8>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_uint8(new_arr)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(_) => {
                    let mut new_arr = IntegerArray::<u16>::with_capacity(row_indices.len(), true);
                    for &idx in row_indices {
                        if let Some(val) = window.get::<IntegerArray<u16>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_uint16(new_arr)
                }
                NumericArray::Null => return None,
            },
            Array::TextArray(text_arr) => match text_arr {
                TextArray::String32(_) => {
                    let mut new_arr = StringArray::<u32>::default();
                    for &idx in row_indices {
                        if let Some(val) = window.get_str(idx) {
                            new_arr.push(val.to_string());
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_string32(new_arr)
                }
                #[cfg(feature = "large_string")]
                TextArray::String64(_) => {
                    let mut new_arr = StringArray::<u64>::default();
                    for &idx in row_indices {
                        if let Some(val) = window.get_str(idx) {
                            new_arr.push(val.to_string());
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_string64(new_arr)
                }
                TextArray::Categorical32(_) => {
                    use crate::{Bitmask, Vec64};
                    use std::collections::HashMap;

                    let mut codes = Vec64::<u32>::with_capacity(row_indices.len());
                    let mut value_map = HashMap::<String, u32>::new();
                    let mut mask = Bitmask::new_set_all(row_indices.len(), true);

                    for (i, &idx) in row_indices.iter().enumerate() {
                        if let Some(val) = window.get_str(idx) {
                            let code = if let Some(&existing_code) = value_map.get(val) {
                                existing_code
                            } else {
                                let new_code = value_map.len() as u32;
                                value_map.insert(val.to_string(), new_code);
                                new_code
                            };
                            codes.push(code);
                        } else {
                            codes.push(0);
                            mask.set_false(i);
                        }
                    }

                    let mut unique_values = Vec64::<String>::with_capacity(value_map.len());
                    for (val, code) in value_map {
                        unique_values[code as usize] = val;
                    }

                    let null_mask = if mask.all_set() { None } else { Some(mask) };

                    let new_arr = CategoricalArray::<u32>::new(codes, unique_values, null_mask);
                    Array::from_categorical32(new_arr)
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical8(_) => {
                    use crate::{Bitmask, Vec64};
                    use std::collections::HashMap;

                    let mut codes = Vec64::<u8>::with_capacity(row_indices.len());
                    let mut value_map = HashMap::<String, u8>::new();
                    let mut mask = Bitmask::new_set_all(row_indices.len(), true);

                    for (i, &idx) in row_indices.iter().enumerate() {
                        if let Some(val) = window.get_str(idx) {
                            let code = if let Some(&existing_code) = value_map.get(val) {
                                existing_code
                            } else {
                                let new_code = value_map.len() as u8;
                                value_map.insert(val.to_string(), new_code);
                                new_code
                            };
                            codes.push(code);
                        } else {
                            codes.push(0);
                            mask.set_false(i);
                        }
                    }

                    let mut unique_values = Vec64::<String>::with_capacity(value_map.len());
                    for (val, code) in value_map {
                        unique_values[code as usize] = val;
                    }

                    let null_mask = if mask.all_set() { None } else { Some(mask) };

                    let new_arr = CategoricalArray::<u8>::new(codes, unique_values, null_mask);
                    Array::from_categorical8(new_arr)
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(_) => {
                    use crate::{Bitmask, Vec64};
                    use std::collections::HashMap;

                    let mut codes = Vec64::<u16>::with_capacity(row_indices.len());
                    let mut value_map = HashMap::<String, u16>::new();
                    let mut mask = Bitmask::new_set_all(row_indices.len(), true);

                    for (i, &idx) in row_indices.iter().enumerate() {
                        if let Some(val) = window.get_str(idx) {
                            let code = if let Some(&existing_code) = value_map.get(val) {
                                existing_code
                            } else {
                                let new_code = value_map.len() as u16;
                                value_map.insert(val.to_string(), new_code);
                                new_code
                            };
                            codes.push(code);
                        } else {
                            codes.push(0);
                            mask.set_false(i);
                        }
                    }

                    let mut unique_values = Vec64::<String>::with_capacity(value_map.len());
                    for (val, code) in value_map {
                        unique_values[code as usize] = val;
                    }

                    let null_mask = if mask.all_set() { None } else { Some(mask) };

                    let new_arr = CategoricalArray::<u16>::new(codes, unique_values, null_mask);
                    Array::from_categorical16(new_arr)
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(_) => {
                    use crate::{Bitmask, Vec64};
                    use std::collections::HashMap;

                    let mut codes = Vec64::<u64>::with_capacity(row_indices.len());
                    let mut value_map = HashMap::<String, u64>::new();
                    let mut mask = Bitmask::new_set_all(row_indices.len(), true);

                    for (i, &idx) in row_indices.iter().enumerate() {
                        if let Some(val) = window.get_str(idx) {
                            let code = if let Some(&existing_code) = value_map.get(val) {
                                existing_code
                            } else {
                                let new_code = value_map.len() as u64;
                                value_map.insert(val.to_string(), new_code);
                                new_code
                            };
                            codes.push(code);
                        } else {
                            codes.push(0);
                            mask.set_false(i);
                        }
                    }

                    let mut unique_values = Vec64::<String>::with_capacity(value_map.len());
                    for (val, code) in value_map {
                        unique_values[code as usize] = val;
                    }

                    let null_mask = if mask.all_set() { None } else { Some(mask) };

                    let new_arr = CategoricalArray::<u64>::new(codes, unique_values, null_mask);
                    Array::from_categorical64(new_arr)
                }
                TextArray::Null => return None,
            },
            Array::BooleanArray(_) => {
                let mut new_arr = BooleanArray::with_capacity(row_indices.len(), true);
                for &idx in row_indices {
                    if let Some(val) = window.get::<BooleanArray<()>>(idx) {
                        new_arr.push(val);
                    } else {
                        new_arr.push_null();
                    }
                }
                Array::from_bool(new_arr)
            }
            #[cfg(feature = "datetime")]
            Array::TemporalArray(temp_arr) => match temp_arr {
                TemporalArray::Datetime32(arr) => {
                    let mut new_arr = DatetimeArray::<i32>::with_capacity(
                        row_indices.len(),
                        true,
                        Some(arr.time_unit),
                    );
                    for &idx in row_indices {
                        if let Some(val) = window.get::<DatetimeArray<i32>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_datetime_i32(new_arr)
                }
                TemporalArray::Datetime64(arr) => {
                    let mut new_arr = DatetimeArray::<i64>::with_capacity(
                        row_indices.len(),
                        true,
                        Some(arr.time_unit),
                    );
                    for &idx in row_indices {
                        if let Some(val) = window.get::<DatetimeArray<i64>>(idx) {
                            new_arr.push(val);
                        } else {
                            new_arr.push_null();
                        }
                    }
                    Array::from_datetime_i64(new_arr)
                }
                TemporalArray::Null => return None,
            },
        };

        Some(result)
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

    /// Gather specific row indices from this view into a materialised Table.
    /// Indices are relative to this view's window.
    #[cfg(feature = "select")]
    pub fn gather_rows(&self, indices: &[usize]) -> Table {
        if indices.is_empty() {
            return Table::new(self.name.clone(), Some(vec![]));
        }

        let cols: Vec<_> = self
            .fields
            .iter()
            .zip(&self.cols)
            .filter_map(|(field, window)| {
                let gathered_array = self.gather_rows_from_window(window, indices)?;
                let null_count = gathered_array.null_count();

                Some(FieldArray {
                    field: field.clone(),
                    array: gathered_array,
                    null_count,
                })
            })
            .collect();

        Table::build(cols, indices.len(), self.name.clone())
    }
}

impl Display for TableV {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let n_rows = self.n_rows();
        let n_cols = self.n_cols();

        if n_cols == 0 {
            return writeln!(f, "TableView \"{}\" [0 rows × 0 cols] – empty", self.name);
        }

        // Determine which rows to display
        let row_indices: Vec<usize> = if n_rows <= MAX_PREVIEW {
            (0..n_rows).collect()
        } else {
            let mut idx = (0..10).collect::<Vec<_>>();
            idx.extend((n_rows - 10)..n_rows);
            idx
        };

        // Build column headers and track widths
        let mut headers: Vec<String> = Vec::with_capacity(n_cols);
        let mut widths: Vec<usize> = Vec::with_capacity(n_cols);

        for col_idx in 0..n_cols {
            if let Some(_col_view) = self.cols.get(col_idx) {
                let hdr = if let Some(f) = self.fields.get(col_idx) {
                    format!("{}:{:?}", f.name, f.dtype)
                } else {
                    "unknown".to_string()
                };
                widths.push(hdr.len());
                headers.push(hdr);
            }
        }

        // Build matrix of cell strings
        let mut rows: Vec<Vec<String>> = Vec::with_capacity(row_indices.len());

        for &row_idx in &row_indices {
            let mut row: Vec<String> = Vec::with_capacity(n_cols);

            for col_idx in 0..n_cols {
                if let Some(col_view) = self.cols.get(col_idx) {
                    let val = value_to_string(&col_view.array, row_idx);
                    widths[col_idx] = widths[col_idx].max(val.len());
                    row.push(val);
                } else {
                    row.push("·".to_string());
                }
            }
            rows.push(row);
        }

        // Calculate idx column width
        let max_idx = n_rows.saturating_sub(1);
        let idx_width = usize::max(
            3, // "idx"
            (max_idx as f64).log10().floor() as usize + 1,
        );

        // Render header
        writeln!(
            f,
            "TableView \"{}\" [{} rows × {} cols]",
            self.name, n_rows, n_cols
        )?;
        print_rule(f, idx_width, &widths)?;
        print_header_row(f, idx_width, &headers, &widths)?;
        print_rule(f, idx_width, &widths)?;

        // Render body
        for (i, cells) in rows.iter().enumerate() {
            let row_idx = row_indices[i];

            write!(f, "| {idx:>w$} |", idx = row_idx, w = idx_width)?;
            for (col_idx, cell) in cells.iter().enumerate() {
                write!(f, " {val:^w$} |", val = cell, w = widths[col_idx])?;
            }
            writeln!(f)?;
            if i == 9 && n_rows > MAX_PREVIEW {
                print_ellipsis_row(f, idx_width, &widths)?;
            }
        }
        print_rule(f, idx_width, &widths)
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
    /// Concatenates two table views by materialising both to owned tables,
    /// concatenating them, and wrapping the result back in a view.
    ///
    /// # Notes
    /// - This operation copies data from both views to create owned tables.
    /// - The resulting view has offset=0 and length equal to the combined length.
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Materialise both views to owned tables
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

                // If the view is windowed, we need to materialise the slice
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

// ===== Selection Trait Implementations =====

#[cfg(feature = "select")]
impl ColumnSelection for TableV {
    type View = TableV;
    type ColView = ArrayV;

    fn c<S: FieldSelector>(&self, selection: S) -> TableV {
        // Resolve selector to field indices
        let indices = selection.resolve_fields(&self.fields);

        // Filter fields and cols to only selected ones
        let selected_fields: Vec<Arc<Field>> = indices
            .iter()
            .filter_map(|&i| self.fields.get(i).cloned())
            .collect();
        let selected_cols: Vec<ArrayV> = indices
            .iter()
            .filter_map(|&i| self.cols.get(i).cloned())
            .collect();

        TableV {
            name: self.name.clone(),
            fields: selected_fields,
            cols: selected_cols,
            offset: self.offset,
            len: self.len,
        }
    }

    fn col_ix(&self, idx: usize) -> Option<ArrayV> {
        self.cols.get(idx).cloned()
    }

    fn col_vec(&self) -> Vec<ArrayV> {
        self.cols.clone()
    }

    fn get_cols(&self) -> Vec<Arc<Field>> {
        self.fields.clone()
    }
}

#[cfg(feature = "select")]
impl RowSelection for TableV {
    type View = TableV;

    fn r<S: DataSelector>(&self, selection: S) -> TableV {
        if selection.is_contiguous() {
            // Contiguous selection (ranges): adjust offset and len
            let indices = selection.resolve_indices(self.len);
            if indices.is_empty() {
                return TableV {
                    name: self.name.clone(),
                    fields: self.fields.clone(),
                    cols: self.cols.clone(),
                    offset: self.offset,
                    len: 0,
                };
            }
            let new_offset = self.offset + indices[0];
            let new_len = indices.len();
            TableV {
                name: self.name.clone(),
                fields: self.fields.clone(),
                cols: self.cols.clone(),
                offset: new_offset,
                len: new_len,
            }
        } else {
            // Non-contiguous selection (index arrays): materialise into Table
            let indices = selection.resolve_indices(self.len);
            let materialised_table = self.gather_rows(&indices);
            TableV::from(materialised_table)
        }
    }

    fn get_row_count(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::structs::field::Field;
    use crate::structs::field_array::FieldArray;
    use crate::structs::table::Table;
    #[cfg(feature = "select")]
    use crate::traits::selection::ColumnSelection;
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
        assert_eq!(slice.col_names(), vec!["a", "b"]);
        assert_eq!(slice.col_name_index("b"), Some(1));
        assert!(!slice.is_empty());
        assert_eq!(slice.end(), 4);
    }

    #[cfg(feature = "select")]
    #[test]
    fn test_table_view_selection_trait() {
        let field_a = Field::new("a", ArrowType::Int32, false, None);
        let arr_a = Array::from_int32(IntegerArray::from_slice(&[1, 2, 3, 4, 5]));
        let fa_a = FieldArray::new(field_a, arr_a);

        let mut tbl = Table::new_empty();
        tbl.add_col(fa_a);

        let slice = TableV::from_table(tbl, 0, 5);

        // Test ColumnSelection trait methods
        assert_eq!(slice.col("a").cols.len(), 1); // col returns TableV, check it has 1 column
        assert_eq!(slice.col("nonexistent").cols.len(), 0); // Not found = 0 columns
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
