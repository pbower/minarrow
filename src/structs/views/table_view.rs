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
use crate::traits::print::{MAX_PREVIEW, value_to_string, print_rule, print_header_row, print_ellipsis_row};
#[cfg(feature = "select")]
use crate::traits::selection::{FieldSelector, DataSelector, FieldSelection, DataSelection};
use crate::traits::shape::Shape;
use crate::{Field, FieldArray, Table};
#[cfg(feature = "select")]
use crate::Array;

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
/// - Column helpers (`col`, `col_by_name`, `col_window`) provide ergonomic access.
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
    /// Active column selection indices (None = all columns)
    #[cfg(feature = "select")]
    pub active_col_selection: Option<Vec<usize>>,
    /// Active row selection indices (None = all rows in window)
    #[cfg(feature = "select")]
    pub active_row_selection: Option<Vec<usize>>,
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
            #[cfg(feature = "select")]
            active_col_selection: None,
            #[cfg(feature = "select")]
            active_row_selection: None,
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
            #[cfg(feature = "select")]
            active_col_selection: None,
            #[cfg(feature = "select")]
            active_row_selection: None,
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
            #[cfg(feature = "select")]
            active_col_selection: self.active_col_selection.clone(),
            #[cfg(feature = "select")]
            active_row_selection: self.active_row_selection.clone(),
        }
    }

    /// Returns the effective column indices based on active_col_selection.
    /// If None, returns all column indices (0..n_cols).
    #[inline]
    #[cfg(feature = "select")]
    fn effective_col_indices(&self) -> Vec<usize> {
        match &self.active_col_selection {
            Some(indices) => indices.clone(),
            None => (0..self.cols.len()).collect(),
        }
    }

    /// Maps a logical column index (within selection) to physical column index.
    /// Returns None if idx is out of bounds of the selection.
    #[inline]
    #[cfg(feature = "select")]
    fn map_col_idx(&self, idx: usize) -> Option<usize> {
        match &self.active_col_selection {
            Some(indices) => indices.get(idx).copied(),
            None => if idx < self.cols.len() { Some(idx) } else { None },
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
    /// Respects active_col_selection if present.
    #[inline]
    pub fn n_cols(&self) -> usize {
        #[cfg(feature = "select")]
        if let Some(indices) = &self.active_col_selection {
            return indices.len();
        }
        self.cols.len()
    }

    /// Returns the number of rows in the window.
    /// Respects active_row_selection if present.
    #[inline]
    pub fn n_rows(&self) -> usize {
        #[cfg(feature = "select")]
        if let Some(indices) = &self.active_row_selection {
            return indices.len();
        }
        self.len
    }

    /// Returns the number of rows in the window.
    /// Respects active_row_selection if present.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_rows()
    }

    /// Returns the name of the table.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the column window at the given index.
    /// The index is relative to the active column selection if present.
    #[inline]
    pub fn col(&self, idx: usize) -> Option<&ArrayV> {
        #[cfg(feature = "select")]
        {
            self.map_col_idx(idx).and_then(|physical_idx| self.cols.get(physical_idx))
        }
        #[cfg(not(feature = "select"))]
        {
            self.cols.get(idx)
        }
    }

    /// Returns a slice of all column windows.
    /// Respects active_col_selection if present.
    #[inline]
    pub fn cols(&self) -> Vec<&ArrayV> {
        #[cfg(feature = "select")]
        {
            let indices = self.effective_col_indices();
            indices.iter().filter_map(|&i| self.cols.get(i)).collect()
        }
        #[cfg(not(feature = "select"))]
        {
            self.cols.iter().collect()
        }
    }

    /// Returns an iterator over all column names.
    /// Respects active_col_selection if present.
    #[inline]
    pub fn col_names(&self) -> Vec<&str> {
        #[cfg(feature = "select")]
        {
            let indices = self.effective_col_indices();
            indices.iter().filter_map(|&i| self.fields.get(i).map(|f| f.name.as_str())).collect()
        }
        #[cfg(not(feature = "select"))]
        {
            self.fields.iter().map(|f| f.name.as_str()).collect()
        }
    }

    /// Returns the index of a column by name.
    /// Returns the logical index within the active selection if present.
    #[inline]
    pub fn col_index(&self, name: &str) -> Option<usize> {
        // First find the physical index
        let physical_idx = self.fields.iter().position(|f| f.name == name)?;

        // Then map to logical index within selection
        #[cfg(feature = "select")]
        {
            match &self.active_col_selection {
                Some(indices) => indices.iter().position(|&i| i == physical_idx),
                None => Some(physical_idx),
            }
        }
        #[cfg(not(feature = "select"))]
        {
            Some(physical_idx)
        }
    }

    /// Returns the window tuple of the column at the given index.
    /// The index is relative to the active column selection if present.
    #[inline]
    pub fn col_window(&self, idx: usize) -> Option<ArrayV> {
        #[cfg(feature = "select")]
        {
            self.map_col_idx(idx).and_then(|physical_idx| {
                self.cols.get(physical_idx).map(|av| {
                    let (array, offset, len) = &av.as_tuple();
                    ArrayV::new(array.clone(), *offset, *len)
                })
            })
        }
        #[cfg(not(feature = "select"))]
        {
            self.cols.get(idx).map(|av| {
                let (array, offset, len) = &av.as_tuple();
                ArrayV::new(array.clone(), *offset, *len)
            })
        }
    }

    /// Returns the name of the column at the given index.
    /// The index is relative to the active column selection if present.
    #[inline]
    pub fn col_name(&self, idx: usize) -> Option<&str> {
        #[cfg(feature = "select")]
        {
            self.map_col_idx(idx).and_then(|physical_idx| {
                self.fields.get(physical_idx).map(|f| f.name.as_str())
            })
        }
        #[cfg(not(feature = "select"))]
        {
            self.fields.get(idx).map(|f| f.name.as_str())
        }
    }

    /// Returns a `Vec<bool>` indicating which columns are nullable.
    /// Respects active_col_selection if present.
    #[inline]
    pub fn cols_nullable(&self) -> Vec<bool> {
        #[cfg(feature = "select")]
        {
            let indices = self.effective_col_indices();
            indices.iter().filter_map(|&i| self.fields.get(i).map(|f| f.nullable)).collect()
        }
        #[cfg(not(feature = "select"))]
        {
            self.fields.iter().map(|f| f.nullable).collect()
        }
    }

    /// Returns a reference to the column window by column name.
    /// Returns None if the column is not in the active selection.
    #[inline]
    pub fn col_by_name(&self, name: &str) -> Option<&ArrayV> {
        self.col_index(name).and_then(|logical_idx| self.col(logical_idx))
    }

    /// Refine column selection on this view (table-specific convenience method).
    ///
    /// This method delegates to `.f()` from the `Selection` trait.
    /// For compatibility across dimensions, prefer using `.f()`, `.fields()`, or `.y()`.
    ///
    /// # Example
    /// ```ignore
    /// use minarrow::TableV;
    ///
    /// // Refine column selection
    /// let view2 = view.c(&["A", "B"]);
    /// ```
    #[cfg(feature = "select")]
    pub fn c<S: FieldSelector>(&self, selection: S) -> TableV {
        self.f(selection)
    }

    /// Refine row selection on this view (table-specific convenience method).
    ///
    /// This method delegates to `.d()` from the `Selection` trait.
    /// For compatibility across dimensions, prefer using `.d()`, `.data()`, or `.x()`.
    ///
    /// # Example
    /// ```ignore
    /// use minarrow::TableV;
    ///
    /// // Refine row selection
    /// let view2 = view.r(5..10);
    /// ```
    #[cfg(feature = "select")]
    pub fn r<S: DataSelector>(&self, selection: S) -> TableV {
        self.d(selection)
    }

    /// Consumes the TableView, producing an owned Table with the sliced data.
    /// Copies the data.
    /// Respects both active_col_selection and active_row_selection.
    ///
    /// Gathers specific rows if active_row_selection is set, supporting both
    /// contiguous and non-contiguous row selections.
    pub fn to_table(&self) -> Table {
        #[cfg(feature = "select")]
        let col_indices = self.effective_col_indices();
        #[cfg(not(feature = "select"))]
        let col_indices: Vec<usize> = (0..self.cols.len()).collect();

        // Handle row selection by gathering specific rows
        #[cfg(feature = "select")]
        if let Some(row_sel) = &self.active_row_selection {
            if row_sel.is_empty() {
                return Table::new(self.name.clone(), Some(vec![]));
            }

            let cols: Vec<_> = col_indices
                .iter()
                .filter_map(|&col_idx| {
                    let field = self.fields.get(col_idx)?;
                    let window = self.cols.get(col_idx)?;

                    // Gather rows from the window based on selected indices
                    let gathered_array = self.gather_rows_from_window(window, row_sel)?;
                    let null_count = gathered_array.null_count();

                    Some(FieldArray {
                        field: field.clone(),
                        array: gathered_array,
                        null_count,
                    })
                })
                .collect();

            return Table {
                cols,
                n_rows: row_sel.len(),
                name: self.name.clone(),
            };
        }

        // No row selection - standard slicing
        let cols: Vec<_> = col_indices
            .iter()
            .filter_map(|&col_idx| {
                let field = self.fields.get(col_idx)?;
                let window = self.cols.get(col_idx)?;
                let w = window.as_tuple();

                let sliced = w.0.slice_clone(w.1, w.2);
                let null_count = sliced.null_count();

                Some(FieldArray {
                    field: field.clone(),
                    array: sliced,
                    null_count,
                })
            })
            .collect();

        let n_rows = if cols.is_empty() { 0 } else { cols[0].len() };

        Table {
            cols,
            n_rows,
            name: self.name.clone(),
        }
    }

    // TODO: Replace this - horribly inefficient

    /// Helper method to gather specific rows from an ArrayV window
    #[cfg(feature = "select")]
    fn gather_rows_from_window(&self, window: &ArrayV, row_indices: &[usize]) -> Option<Array> {
        use crate::{Array, BooleanArray, CategoricalArray, FloatArray, IntegerArray, MaskedArray, NumericArray, StringArray, TextArray};
        #[cfg(feature = "datetime")]
        use crate::{DatetimeArray, TemporalArray};

        let result = match &window.array {
            Array::Null => return None, // Cannot gather from null array
            Array::NumericArray(num_arr) => {
                match num_arr {
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
                }
            }
            Array::TextArray(text_arr) => {
                match text_arr {
                    TextArray::String32(_) => {
                        // Collect strings first, then create array
                        let mut values: Vec<&str> = Vec::with_capacity(row_indices.len());
                        for &idx in row_indices {
                            if let Some(val) = window.get_str(idx) {
                                values.push(val);
                            } else {
                                values.push(""); // Placeholder for null
                            }
                        }
                        let mut new_arr = StringArray::<u32>::from_vec(values, None);
                        // Mark nulls appropriately
                        for (i, &idx) in row_indices.iter().enumerate() {
                            if window.get_str(idx).is_none() {
                                new_arr.set_null(i);
                            }
                        }
                        Array::from_string32(new_arr)
                    }
                    #[cfg(feature = "large_string")]
                    TextArray::String64(_) => {
                        let mut values: Vec<&str> = Vec::with_capacity(row_indices.len());
                        for &idx in row_indices {
                            if let Some(val) = window.get_str(idx) {
                                values.push(val);
                            } else {
                                values.push("");
                            }
                        }
                        let mut new_arr = StringArray::<u64>::from_vec(values, None);
                        for (i, &idx) in row_indices.iter().enumerate() {
                            if window.get_str(idx).is_none() {
                                new_arr.set_null(i);
                            }
                        }
                        Array::from_string64(new_arr)
                    }
                    TextArray::Categorical32(_) => {
                        // Similar approach - collect strings and build categorical
                        let mut values: Vec<&str> = Vec::with_capacity(row_indices.len());
                        for &idx in row_indices {
                            if let Some(val) = window.get_str(idx) {
                                values.push(val);
                            } else {
                                values.push("");
                            }
                        }
                        let mut new_arr = CategoricalArray::<u32>::from_vec(values, None);
                        for (i, &idx) in row_indices.iter().enumerate() {
                            if window.get_str(idx).is_none() {
                                new_arr.set_null(i);
                            }
                        }
                        Array::from_categorical32(new_arr)
                    }
                    #[cfg(feature = "extended_categorical")]
                    TextArray::Categorical8(_) => {
                        let mut values: Vec<&str> = Vec::with_capacity(row_indices.len());
                        for &idx in row_indices {
                            if let Some(val) = window.get_str(idx) {
                                values.push(val);
                            } else {
                                values.push("");
                            }
                        }
                        let mut new_arr = CategoricalArray::<u8>::from_vec(values, None);
                        for (i, &idx) in row_indices.iter().enumerate() {
                            if window.get_str(idx).is_none() {
                                new_arr.set_null(i);
                            }
                        }
                        Array::from_categorical8(new_arr)
                    }
                    #[cfg(feature = "extended_categorical")]
                    TextArray::Categorical16(_) => {
                        let mut values: Vec<&str> = Vec::with_capacity(row_indices.len());
                        for &idx in row_indices {
                            if let Some(val) = window.get_str(idx) {
                                values.push(val);
                            } else {
                                values.push("");
                            }
                        }
                        let mut new_arr = CategoricalArray::<u16>::from_vec(values, None);
                        for (i, &idx) in row_indices.iter().enumerate() {
                            if window.get_str(idx).is_none() {
                                new_arr.set_null(i);
                            }
                        }
                        Array::from_categorical16(new_arr)
                    }
                    #[cfg(feature = "extended_categorical")]
                    TextArray::Categorical64(_) => {
                        let mut values: Vec<&str> = Vec::with_capacity(row_indices.len());
                        for &idx in row_indices {
                            if let Some(val) = window.get_str(idx) {
                                values.push(val);
                            } else {
                                values.push("");
                            }
                        }
                        let mut new_arr = CategoricalArray::<u64>::from_vec(values, None);
                        for (i, &idx) in row_indices.iter().enumerate() {
                            if window.get_str(idx).is_none() {
                                new_arr.set_null(i);
                            }
                        }
                        Array::from_categorical64(new_arr)
                    }
                    TextArray::Null => return None,
                }
            }
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
            Array::TemporalArray(temp_arr) => {
                match temp_arr {
                    TemporalArray::Datetime32(arr) => {
                        let mut new_arr = DatetimeArray::<i32>::with_capacity(row_indices.len(), true, Some(arr.time_unit));
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
                        let mut new_arr = DatetimeArray::<i64>::with_capacity(row_indices.len(), true, Some(arr.time_unit));
                        for &idx in row_indices {
                            if let Some(val) = window.get::<DatetimeArray<i64>>(idx) {
                                new_arr.push(val);
                            } else {
                                new_arr.push_null();
                            }
                        }
                        Array::from_datetime_i64(new_arr)
                    }
                    TemporalArray::Null => return None, // Cannot gather from null array
                }
            }
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

        // Get fields respecting active_col_selection
        #[cfg(feature = "select")]
        let fields: Vec<Arc<Field>> = match &self.active_col_selection {
            Some(indices) => indices.iter()
                .filter_map(|&i| self.fields.get(i).cloned())
                .collect(),
            None => self.fields.clone(),
        };
        #[cfg(not(feature = "select"))]
        let fields = &self.fields;

        for logical_col_idx in 0..n_cols {
            if let Some(_col_view) = self.col(logical_col_idx) {
                let hdr = if let Some(f) = fields.get(logical_col_idx) {
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

        for &logical_row_idx in &row_indices {
            let mut row: Vec<String> = Vec::with_capacity(n_cols);

            for logical_col_idx in 0..n_cols {
                if let Some(col_view) = self.col(logical_col_idx) {
                    // Map logical row to physical row for active_row_selection
                    #[cfg(feature = "select")]
                    let physical_row_idx = if let Some(ref row_sel) = self.active_row_selection {
                        row_sel.get(logical_row_idx).copied().unwrap_or(logical_row_idx)
                    } else {
                        logical_row_idx
                    };
                    #[cfg(not(feature = "select"))]
                    let physical_row_idx = logical_row_idx;

                    let val = value_to_string(&col_view.array, physical_row_idx);
                    widths[logical_col_idx] = widths[logical_col_idx].max(val.len());
                    row.push(val);
                } else {
                    row.push("·".to_string());
                }
            }
            rows.push(row);
        }

        // Calculate idx column width
        #[cfg(feature = "select")]
        let max_physical_idx = if let Some(ref row_sel) = self.active_row_selection {
            row_sel.iter().max().copied().unwrap_or(0)
        } else {
            n_rows.saturating_sub(1)
        };
        #[cfg(not(feature = "select"))]
        let max_physical_idx = n_rows.saturating_sub(1);

        let idx_width = usize::max(
            3, // "idx"
            (max_physical_idx as f64).log10().floor() as usize + 1,
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
            let logical_row = row_indices[i];

            // Get physical row index for display
            #[cfg(feature = "select")]
            let physical_row = if let Some(ref row_sel) = self.active_row_selection {
                row_sel.get(logical_row).copied().unwrap_or(logical_row)
            } else {
                logical_row
            };
            #[cfg(not(feature = "select"))]
            let physical_row = logical_row;

            write!(f, "| {idx:>w$} |", idx = physical_row, w = idx_width)?;
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
            #[cfg(feature = "select")]
            active_col_selection: None,
            #[cfg(feature = "select")]
            active_row_selection: None,
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

// ===== Selection Trait Implementations =====

#[cfg(feature = "select")]
impl FieldSelection for TableV {
    type View = TableV;

    fn f<S: FieldSelector>(&self, selection: S) -> TableV {
        // Get current fields (filtered by active_col_selection)
        let current_fields = self.get_fields();

        // Resolve selection within current view's fields
        let logical_indices = selection.resolve_fields(&current_fields);

        // Map logical indices back to physical indices
        let physical_indices: Vec<usize> = match &self.active_col_selection {
            Some(existing) => logical_indices.iter()
                .filter_map(|&i| existing.get(i).copied())
                .collect(),
            None => logical_indices,
        };

        let mut result = self.clone();
        result.active_col_selection = Some(physical_indices);
        result
    }

    fn get_fields(&self) -> Vec<Arc<Field>> {
        // Return fields filtered by active_col_selection
        match &self.active_col_selection {
            Some(indices) => indices.iter()
                .filter_map(|&i| self.fields.get(i).cloned())
                .collect(),
            None => self.fields.clone(),
        }
    }
}

#[cfg(feature = "select")]
impl DataSelection for TableV {
    type View = TableV;

    fn d<S: DataSelector>(&self, selection: S) -> TableV {
        // Resolve to logical indices within current view
        let logical_n_rows = self.n_rows();
        let logical_indices = selection.resolve_indices(logical_n_rows);

        // Map logical indices back to physical indices
        let physical_indices: Vec<usize> = match &self.active_row_selection {
            Some(existing) => logical_indices.iter()
                .filter_map(|&i| existing.get(i).copied())
                .collect(),
            None => logical_indices,
        };

        let mut result = self.clone();
        result.active_row_selection = Some(physical_indices);
        result
    }

    fn get_data_count(&self) -> usize {
        self.n_rows()
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
        assert_eq!(slice.col_names(), vec!["a", "b"]);
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
