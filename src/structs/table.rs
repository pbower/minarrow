//! # **Table Module** - *Standard Table ("RecordBatch") for Columnar Analytics and Data Engineering*
//!
//! Columnar data container pairing a fixed-length set of rows
//! with named, typed `FieldArray` columns.
//!
//! Equivalent in role to Apache Arrow’s `RecordBatch`, with
//! guaranteed column length consistency and optional table name.
//!
//! Great for in-memory analytics, transformation pipelines,
//! and zero-copy FFI interchange.
//!
//! Cast into *Polars* dataframe via `.to_polars()` or *Apache Arrow* RecordBatch via `.to_apache_arrow()`,
//! zero-copy, via the `cast_polars` and `cast_arrow` features.

use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "cast_arrow")]
use arrow::array::RecordBatch;
#[cfg(feature = "cast_polars")]
use polars::frame::DataFrame;
#[cfg(feature = "cast_polars")]
use polars::prelude::Column;
#[cfg(feature = "parallel_proc")]
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator};

use super::field_array::FieldArray;
#[cfg(all(feature = "views", feature = "select"))]
use crate::ArrayV;
use crate::Field;
#[cfg(feature = "chunked")]
use crate::SuperTable;
#[cfg(feature = "views")]
use crate::TableV;
use crate::enums::{error::MinarrowError, shape_dim::ShapeDim};
#[cfg(all(feature = "views", feature = "select"))]
use crate::traits::selection::{ColumnSelection, DataSelector, FieldSelector, RowSelection};
use crate::traits::{
    concatenate::Concatenate,
    print::{MAX_PREVIEW, print_ellipsis_row, print_header_row, print_rule, value_to_string},
    shape::Shape,
};

// Global counter for unnamed table instances
static UNNAMED_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// # Table
///
/// # Description
/// - Standard columnar table with named columns (`FieldArray`),
/// a fixed number of rows, and an optional logical table name.
/// - All columns are required to be equal length and have consistent schema.
/// - Supports zero-copy slicing, efficient iteration, and bulk operations.
/// - Equivalent to the `RecordBatch` in *Apache Arrow*.
///
/// # Structure
/// - `cols`: A vector of `FieldArray`, each representing a column with metadata and data.
/// - `n_rows`: The logical number of rows (guaranteed equal for all columns).
/// - `name`: Optional logical name or alias for this table instance.
///
/// # Usage
/// - Use `Table` as a general-purpose, in-memory columnar data container.
/// - Good for analytics, and transformation pipelines.
/// - For batched/partitioned tables, see [`SuperTable`] or windowed/chunked abstractions.
/// - Cast into *Polars* dataframe via `.to_polars()` or *Apache Arrow* via `.to_apache_arrow()`
/// - FFI-compatible
///
/// # Notes
/// - Table instances are typically lightweight to clone and pass by value.
/// - For mutation, construct a new table or replace individual columns as needed.
/// - There is an alias `RecordBatch` under [crate::aliases::RecordBatch]
///
/// # Example
/// ```rust
/// use minarrow::{FieldArray, Print, Table, arr_i32, arr_str32, vec64};
///
/// let col1 = FieldArray::from_arr("numbers", arr_i32![1, 2, 3]);
/// let col2 = FieldArray::from_arr("letters", arr_str32!["x", "y", "z"]);
///
/// let mut tbl = Table::new("Demo".into(), vec![col1, col2].into());
/// tbl.print();
/// ```
#[repr(C, align(64))]
#[derive(Default, PartialEq, Clone, Debug)]
pub struct Table {
    /// FieldArrays representing named columns.
    pub cols: Vec<FieldArray>,
    /// Number of rows in the table.
    pub n_rows: usize,
    /// Table name
    pub name: String,
}

impl Table {
    /// Constructs a new Table with a specified name and optional columns.
    /// If `cols` is provided, the number of rows will be inferred from the first column.
    pub fn new(name: String, cols: Option<Vec<FieldArray>>) -> Self {
        let cols = cols.unwrap_or_else(Vec::new);
        let n_rows = cols.first().map(|col| col.len()).unwrap_or(0);

        let name = if name.trim().is_empty() {
            let id = UNNAMED_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("UnnamedTable{}", id)
        } else {
            name
        };

        Self { cols, n_rows, name }
    }

    /// Constructs a new, empty Table with a globally unique name.
    pub fn new_empty() -> Self {
        let id = UNNAMED_COUNTER.fetch_add(1, Ordering::Relaxed);
        let name = format!("UnnamedTable{}", id);

        Self {
            cols: Vec::new(),
            n_rows: 0,
            name,
        }
    }

    /// Adds a column with a name.
    pub fn add_col(&mut self, field_array: FieldArray) {
        let array_len = field_array.len();
        if self.cols.is_empty() {
            self.n_rows = array_len;
        } else {
            assert!(self.n_rows == array_len, "Column length mismatch");
        }
        self.cols.push(field_array);
    }

    /// Builds a schema via the underlying field arrays
    pub fn schema(&self) -> Vec<Arc<Field>> {
        let mut vec = Vec::new();
        for fa in &self.cols {
            vec.push(fa.field.clone())
        }
        vec
    }

    /// Returns the number of columns.
    pub fn n_cols(&self) -> usize {
        self.cols.len()
    }

    /// Returns the number of rows.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Returns true if the table is empty (no columns or no rows).
    pub fn is_empty(&self) -> bool {
        self.n_cols() == 0 || self.n_rows == 0
    }

    /// Returns the list of column names.
    pub fn col_names(&self) -> Vec<&str> {
        self.cols.iter().map(|fa| fa.field.name.as_str()).collect()
    }

    /// Returns the index of a column by name.
    pub fn col_name_index(&self, name: &str) -> Option<usize> {
        self.cols.iter().position(|fa| fa.field.name == name)
    }

    /// Removes a column by name.
    pub fn remove_col(&mut self, name: &str) -> bool {
        if let Some(idx) = self.col_name_index(name) {
            self.cols.remove(idx);
            self.recalc_n_rows();
            true
        } else {
            false
        }
    }

    /// Removes a column by index.
    pub fn remove_col_at(&mut self, idx: usize) -> bool {
        if idx < self.cols.len() {
            self.cols.remove(idx);
            self.recalc_n_rows();
            true
        } else {
            false
        }
    }

    /// Clears all columns and resets row count.
    pub fn clear(&mut self) {
        self.cols.clear();
        self.n_rows = 0;
    }

    /// Checks if a column with the given name exists.
    pub fn has_col(&self, name: &str) -> bool {
        self.col_name_index(name).is_some()
    }

    /// Returns all columns as a slice.
    pub fn cols(&self) -> &[FieldArray] {
        &self.cols
    }

    /// Returns mutable reference to all columns.
    pub fn cols_mut(&mut self) -> &mut [FieldArray] {
        &mut self.cols
    }

    // Keeps total rows cache up to date
    fn recalc_n_rows(&mut self) {
        if let Some(col) = self.cols.first() {
            self.n_rows = col.len();
        } else {
            self.n_rows = 0;
        }
    }

    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, FieldArray> {
        self.cols.iter()
    }
    #[inline]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, FieldArray> {
        self.cols.iter_mut()
    }

    #[inline]
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n_rows()
    }

    /// Returns a new owned `Table` containing rows `[offset, offset+len)`.
    ///
    /// All columns are deeply copied, but only for the affected row(s).
    pub fn slice_clone(&self, offset: usize, len: usize) -> Self {
        assert!(offset <= self.n_rows, "offset out of bounds");
        assert!(offset + len <= self.n_rows, "slice window out of bounds");
        let cols: Vec<FieldArray> = self
            .cols
            .iter()
            .map(|fa| fa.slice_clone(offset, len))
            .collect();
        let name = format!("{}[{}, {})", self.name, offset, offset + len);
        Table {
            cols,
            n_rows: len,
            name,
        }
    }

    /// Returns a zero-copy view over rows `[offset, offset+len)`.
    /// This view borrows from the parent table and does not copy data.
    #[cfg(feature = "views")]
    pub fn slice(&self, offset: usize, len: usize) -> TableV {
        assert!(offset <= self.n_rows, "offset out of bounds");
        assert!(offset + len <= self.n_rows, "slice window out of bounds");
        TableV::from_table(self.clone(), offset, len)
    }

    /// Maps a function over a single column by name, returning the result.
    /// Returns None if the column doesn't exist.
    pub fn map_col<T, F>(&self, col_name: &str, func: F) -> Option<T>
    where
        F: FnOnce(&FieldArray) -> T,
    {
        self.cols
            .iter()
            .find(|c| c.field.name == col_name)
            .map(func)
    }

    /// Maps a function over multiple columns by name, returning a Vec of results.
    /// Warns if any requested columns are missing.
    pub fn map_cols_by_name<T, F>(&self, col_names: &[&str], mut func: F) -> Vec<T>
    where
        F: FnMut(&FieldArray) -> T,
    {
        let mut results = Vec::with_capacity(col_names.len());
        for name in col_names {
            match self.cols.iter().find(|c| c.field.name == *name) {
                Some(col) => results.push(func(col)),
                None => {
                    eprintln!(
                        "Warning: Column '{}' not found in table '{}'",
                        name, self.name
                    );
                }
            }
        }
        results
    }

    /// Maps a function over multiple columns by index, returning a Vec of results.
    /// Warns if any requested indices are out of bounds.
    pub fn map_cols_by_index<T, F>(&self, indices: &[usize], mut func: F) -> Vec<T>
    where
        F: FnMut(&FieldArray) -> T,
    {
        let mut results = Vec::with_capacity(indices.len());
        for &idx in indices {
            match self.cols.get(idx) {
                Some(col) => results.push(func(col)),
                None => {
                    eprintln!(
                        "Warning: Column index {} out of bounds in table '{}' (has {} columns)",
                        idx,
                        self.name,
                        self.n_cols()
                    );
                }
            }
        }
        results
    }

    /// Maps a function over all columns, returning a Vec of results.
    pub fn map_all_cols<T, F>(&self, func: F) -> Vec<T>
    where
        F: FnMut(&FieldArray) -> T,
    {
        self.cols.iter().map(func).collect()
    }

    /// Inserts rows from another table at the specified index.
    ///
    /// This is an **O(n)** operation where n is the number of rows after the insertion point.
    ///
    /// # Arguments
    /// * `index` - Position before which to insert (0 = prepend, n_rows = append)
    /// * `other` - Table to insert
    ///
    /// # Requirements
    /// - Both tables must have the same number of columns
    /// - Column names, types, and nullability must match in order
    /// - `index` must be <= `self.n_rows()`
    ///
    /// # Errors
    /// - `IndexError` if index > n_rows
    /// - `IncompatibleTypeError` if column schemas don't match
    pub fn insert_rows(&mut self, index: usize, other: &Self) -> Result<(), MinarrowError> {
        // Validate index
        if index > self.n_rows {
            return Err(MinarrowError::IndexError(format!(
                "Index {} out of bounds for table with {} rows",
                index, self.n_rows
            )));
        }

        // Check column count
        if self.n_cols() != other.n_cols() {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "Table",
                to: "Table",
                message: Some(format!(
                    "Cannot insert tables with different column counts: {} vs {}",
                    self.n_cols(),
                    other.n_cols()
                )),
            });
        }

        // If both tables are empty, nothing to do
        if self.n_cols() == 0 {
            return Ok(());
        }

        // Validate column schemas and insert into each column
        for (col_idx, (self_col, other_col)) in
            self.cols.iter_mut().zip(other.cols.iter()).enumerate()
        {
            // Check field compatibility
            if self_col.field.name != other_col.field.name {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "Table",
                    to: "Table",
                    message: Some(format!(
                        "Column {} name mismatch: '{}' vs '{}'",
                        col_idx, self_col.field.name, other_col.field.name
                    )),
                });
            }

            if self_col.field.dtype != other_col.field.dtype {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "Table",
                    to: "Table",
                    message: Some(format!(
                        "Column '{}' type mismatch: {:?} vs {:?}",
                        self_col.field.name, self_col.field.dtype, other_col.field.dtype
                    )),
                });
            }

            if self_col.field.nullable != other_col.field.nullable {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "Table",
                    to: "Table",
                    message: Some(format!(
                        "Column '{}' nullable mismatch: {} vs {}",
                        self_col.field.name, self_col.field.nullable, other_col.field.nullable
                    )),
                });
            }

            // Insert into this column's array
            self_col.array.insert_rows(index, &other_col.array)?;

            // Update null count
            self_col.null_count = self_col.array.null_count();
        }

        // Update row count
        self.n_rows += other.n_rows;

        Ok(())
    }

    /// Splits the Table at the specified row index, consuming self and returning a SuperTable
    /// with two Table batches.
    ///
    /// Splits the underlying buffers, allocating new storage for the second half.
    #[cfg(feature = "chunked")]
    pub fn split(self, index: usize) -> Result<SuperTable, MinarrowError> {
        if index == 0 || index >= self.n_rows {
            return Err(MinarrowError::IndexError(format!(
                "Split index {} out of valid range (0, {})",
                index, self.n_rows
            )));
        }

        // Split each column
        let mut left_cols = Vec::with_capacity(self.cols.len());
        let mut right_cols = Vec::with_capacity(self.cols.len());

        for col in self.cols {
            let split_result = col.array.split(index, &col.field)?;

            // Extract the two arrays from the SuperArray
            let mut left_arrays = split_result.into_arrays();
            let mut right_arrays = left_arrays.split_off(1);

            // Should have one array each after split
            let left_field = left_arrays.remove(0);
            let right_field = right_arrays.remove(0);

            left_cols.push(left_field);
            right_cols.push(right_field);
        }

        let left_table = Table {
            cols: left_cols,
            n_rows: index,
            name: format!("{}_left", self.name),
        };

        let right_table = Table {
            cols: right_cols,
            n_rows: self.n_rows - index,
            name: format!("{}_right", self.name),
        };

        Ok(SuperTable::from_batches(
            vec![Arc::new(left_table), Arc::new(right_table)],
            Some(self.name),
        ))
    }
}

impl Table {
    #[cfg(feature = "parallel_proc")]
    #[inline]
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, FieldArray> {
        self.cols.par_iter()
    }

    #[cfg(feature = "parallel_proc")]
    #[inline]
    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, FieldArray> {
        self.cols.par_iter_mut()
    }

    /// Export each column to arrow-rs `ArrayRef` and build a `RecordBatch`.
    ///
    /// The Arrow schema is derived from the imported array dtypes while
    /// preserving the original field names and nullability flags.
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn to_apache_arrow(&self) -> RecordBatch {
        use arrow::array::ArrayRef;
        assert!(
            !self.cols.is_empty(),
            "Cannot build RecordBatch from an empty Table"
        );

        // Convert columns
        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(self.cols.len());
        for col in &self.cols {
            arrays.push(col.to_apache_arrow());
        }

        // Build Arrow Schema using field names + imported dtypes
        let mut fields = Vec::with_capacity(self.cols.len());
        for (i, col) in self.cols.iter().enumerate() {
            let dt = arrays[i].data_type().clone();
            fields.push(arrow_schema::Field::new(
                col.field.name.clone(),
                dt,
                col.field.nullable,
            ));
        }
        let schema = Arc::new(arrow_schema::Schema::new(fields));

        // Assemble batch
        RecordBatch::try_new(schema, arrays).expect("Failed to build RecordBatch from Table")
    }

    // ** The below polars function is tested tests/polars.rs **

    /// Casts the table to a Polars DataFrame
    #[cfg(feature = "cast_polars")]
    pub fn to_polars(&self) -> DataFrame {
        let cols = self
            .cols
            .iter()
            .map(|fa| Column::new(fa.field.name.clone().into(), fa.to_polars()))
            .collect::<Vec<_>>();
        DataFrame::new(cols).expect("DataFrame build failed")
    }
}

impl<'a> IntoIterator for &'a Table {
    type Item = &'a FieldArray;
    type IntoIter = std::slice::Iter<'a, FieldArray>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.cols.iter()
    }
}

impl<'a> IntoIterator for &'a mut Table {
    type Item = &'a mut FieldArray;
    type IntoIter = std::slice::IterMut<'a, FieldArray>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.cols.iter_mut()
    }
}

impl IntoIterator for Table {
    type Item = FieldArray;
    type IntoIter = <Vec<FieldArray> as IntoIterator>::IntoIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.cols.into_iter()
    }
}

impl Shape for Table {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank2 {
            rows: self.n_rows(),
            cols: self.n_cols(),
        }
    }
}

impl Concatenate for Table {
    /// Concatenates two tables vertically (row-wise).
    ///
    /// # Requirements
    /// - Both tables must have the same number of columns
    /// - Column names, types, and nullability must match in order
    ///
    /// # Returns
    /// A new Table with rows from `self` followed by rows from `other`
    ///
    /// # Errors
    /// - `IncompatibleTypeError` if column schemas don't match
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Check column count
        if self.n_cols() != other.n_cols() {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "Table",
                to: "Table",
                message: Some(format!(
                    "Cannot concatenate tables with different column counts: {} vs {}",
                    self.n_cols(),
                    other.n_cols()
                )),
            });
        }

        // If both tables are empty, return empty table
        if self.n_cols() == 0 {
            return Ok(Table::new(format!("{}+{}", self.name, other.name), None));
        }

        // Validate column schemas match and concatenate arrays
        let mut result_cols = Vec::with_capacity(self.n_cols());

        for (col_idx, (self_col, other_col)) in self
            .cols
            .into_iter()
            .zip(other.cols.into_iter())
            .enumerate()
        {
            // Check field compatibility
            if self_col.field.name != other_col.field.name {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "Table",
                    to: "Table",
                    message: Some(format!(
                        "Column {} name mismatch: '{}' vs '{}'",
                        col_idx, self_col.field.name, other_col.field.name
                    )),
                });
            }

            if self_col.field.dtype != other_col.field.dtype {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "Table",
                    to: "Table",
                    message: Some(format!(
                        "Column '{}' type mismatch: {:?} vs {:?}",
                        self_col.field.name, self_col.field.dtype, other_col.field.dtype
                    )),
                });
            }

            if self_col.field.nullable != other_col.field.nullable {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "Table",
                    to: "Table",
                    message: Some(format!(
                        "Column '{}' nullable mismatch: {} vs {}",
                        self_col.field.name, self_col.field.nullable, other_col.field.nullable
                    )),
                });
            }

            // Concatenate arrays
            let concatenated_array = self_col.array.concat(other_col.array)?;
            let null_count = concatenated_array.null_count();

            // Create new FieldArray with concatenated data
            result_cols.push(FieldArray {
                field: self_col.field.clone(),
                array: concatenated_array,
                null_count,
            });
        }

        // Create result table
        let n_rows = result_cols.first().map(|c| c.len()).unwrap_or(0);
        let name = format!("{}+{}", self.name, other.name);

        Ok(Table {
            cols: result_cols,
            n_rows,
            name,
        })
    }
}

impl Display for Table {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.cols.is_empty() {
            return writeln!(f, "Table  \"{}\" [0 rows × 0 cols] – empty", self.name);
        }

        // Gather column metadata & cell strings (with null handling)
        let row_indices: Vec<usize> = if self.n_rows <= MAX_PREVIEW {
            (0..self.n_rows).collect()
        } else {
            let mut idx = (0..10).collect::<Vec<_>>();
            idx.extend((self.n_rows - 10)..self.n_rows);
            idx
        };

        // column header strings and tracked widths
        let mut headers: Vec<String> = Vec::with_capacity(self.cols.len());
        let mut widths: Vec<usize> = Vec::with_capacity(self.cols.len());

        for col in &self.cols {
            let hdr = format!("{}:{:?}", col.field.name, col.field.dtype);
            widths.push(hdr.len());
            headers.push(hdr);
        }

        // matrix of cell strings
        let mut rows: Vec<Vec<String>> = Vec::with_capacity(row_indices.len());

        for &row_idx in &row_indices {
            let mut row: Vec<String> = Vec::with_capacity(self.cols.len());

            for (col_idx, col) in self.cols.iter().enumerate() {
                let val = value_to_string(&col.array, row_idx);
                widths[col_idx] = widths[col_idx].max(val.len());
                row.push(val);
            }
            rows.push(row);
        }

        // row-index column (“idx”)
        let idx_width = usize::max(
            3, // “idx”
            ((self.n_rows - 1) as f64).log10().floor() as usize + 1,
        );

        // Render header
        writeln!(
            f,
            "Table \"{}\" [{} rows × {} cols]",
            self.name,
            self.n_rows,
            self.cols.len()
        )?;
        print_rule(f, idx_width, &widths)?;
        print_header_row(f, idx_width, &headers, &widths)?;
        print_rule(f, idx_width, &widths)?;

        // Render body
        for (logical_row, cells) in rows.iter().enumerate() {
            let physical_row = row_indices[logical_row];
            write!(f, "| {idx:>w$} |", idx = physical_row, w = idx_width)?;
            for (col_idx, cell) in cells.iter().enumerate() {
                write!(f, " {val:^w$} |", val = cell, w = widths[col_idx])?;
            }
            writeln!(f)?;
            if logical_row == 9 && self.n_rows > MAX_PREVIEW {
                print_ellipsis_row(f, idx_width, &widths)?;
            }
        }
        print_rule(f, idx_width, &widths)
    }
}

// ===== Selection Trait Implementations =====

#[cfg(all(feature = "views", feature = "select"))]
impl ColumnSelection for Table {
    type View = TableV;
    type ColView = ArrayV;

    fn c<S: FieldSelector>(&self, selection: S) -> TableV {
        let fields = self
            .cols
            .iter()
            .map(|fa| fa.field.clone())
            .collect::<Vec<_>>();
        let col_indices = selection.resolve_fields(&fields);

        // Filter to selected columns
        let selected_fields: Vec<Arc<Field>> = col_indices
            .iter()
            .filter_map(|&i| self.cols.get(i).map(|fa| fa.field.clone()))
            .collect();
        let selected_cols: Vec<ArrayV> = col_indices
            .iter()
            .filter_map(|&i| self.cols.get(i).map(|fa| ArrayV::from(fa.clone())))
            .collect();

        TableV {
            name: self.name.clone(),
            fields: selected_fields,
            cols: selected_cols,
            offset: 0,
            len: self.n_rows,
        }
    }

    fn col_ix(&self, idx: usize) -> Option<ArrayV> {
        self.cols.get(idx).map(|fa| ArrayV::from(fa.clone()))
    }

    fn col_vec(&self) -> Vec<ArrayV> {
        self.cols
            .iter()
            .map(|fa| ArrayV::from(fa.clone()))
            .collect()
    }

    fn get_cols(&self) -> Vec<Arc<Field>> {
        self.cols.iter().map(|fa| fa.field.clone()).collect()
    }
}

#[cfg(all(feature = "views", feature = "select"))]
impl RowSelection for Table {
    type View = TableV;

    fn r<S: DataSelector>(&self, selection: S) -> TableV {
        let table_v = TableV::from(self.clone());

        if selection.is_contiguous() {
            // Contiguous selection (ranges): adjust offset and len
            let indices = selection.resolve_indices(self.n_rows);
            if indices.is_empty() {
                return TableV {
                    name: self.name.clone(),
                    fields: table_v.fields,
                    cols: table_v.cols,
                    offset: 0,
                    len: 0,
                };
            }
            let new_offset = indices[0];
            let new_len = indices.len();
            TableV {
                name: self.name.clone(),
                fields: table_v.fields,
                cols: table_v.cols,
                offset: new_offset,
                len: new_len,
            }
        } else {
            // Non-contiguous selection (index arrays): materialise
            let indices = selection.resolve_indices(self.n_rows);
            let materialised_table = table_v.gather_rows(&indices);
            TableV::from(materialised_table)
        }
    }

    fn get_row_count(&self) -> usize {
        self.n_rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::field_array::field_array;
    use crate::traits::masked_array::MaskedArray;
    #[cfg(all(feature = "views", feature = "select"))]
    use crate::traits::selection::ColumnSelection;
    use crate::{Array, BooleanArray, IntegerArray, NumericArray};

    #[test]
    fn test_new_table() {
        let t = Table::new_empty();
        assert_eq!(t.n_cols(), 0);
        assert_eq!(t.n_rows(), 0);
        assert!(t.is_empty());
    }

    #[test]
    fn test_add_and_get_columns() {
        let mut t = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        col1.push(2);
        let mut col2 = BooleanArray::default();
        col2.push(true);
        col2.push(false);

        t.add_col(field_array("ints", Array::from_int32(col1)));
        t.add_col(field_array("bools", Array::from_bool(col2)));

        assert_eq!(t.n_cols(), 2);
        assert_eq!(t.n_rows(), 2);
        assert!(!t.is_empty());

        // Test column access via cols()
        assert!(t.cols().get(0).is_some());
        assert!(t.cols().get(1).is_some());
        assert!(t.cols().get(2).is_none());
        assert_eq!(t.col_names(), vec!["ints", "bools"]);

        // Test column by name via col_name_index
        let idx = t.col_name_index("ints").unwrap();
        let col = t.cols().get(idx).unwrap();
        match &col.array {
            Array::NumericArray(NumericArray::Int32(a)) => assert_eq!(a.len(), 2),
            _ => panic!("ints column type mismatch"),
        }
    }

    #[cfg(all(feature = "views", feature = "select"))]
    #[test]
    fn test_column_selection_trait() {
        let mut t = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        col1.push(2);
        let mut col2 = BooleanArray::default();
        col2.push(true);
        col2.push(false);

        t.add_col(field_array("ints", Array::from_int32(col1)));
        t.add_col(field_array("bools", Array::from_bool(col2)));

        // Test ColumnSelection trait methods
        assert!(t.col_ix(0).is_some());
        assert!(t.col_ix(1).is_some());
        assert!(t.col_ix(2).is_none());

        // col() returns TableV, col_ix(0) gets the single column as ArrayV
        let col_view = t.col("ints");
        assert_eq!(col_view.cols.len(), 1); // Column found
        let av = col_view.col_ix(0).unwrap();
        assert_eq!(col_view.fields[0].name, "ints");
        match &av.array {
            Array::NumericArray(NumericArray::Int32(a)) => assert_eq!(a.len(), 2),
            _ => panic!("ints column type mismatch"),
        }
    }

    #[test]
    #[should_panic(expected = "Column length mismatch")]
    fn test_column_length_mismatch_panics() {
        let mut t = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        let mut col2 = BooleanArray::default();
        col2.push(true);
        col2.push(false);
        t.add_col(field_array("ints", Array::from_int32(col1)));
        // This should panic due to mismatched row count
        t.add_col(field_array("bools", Array::from_bool(col2)));
    }

    #[test]
    fn test_column_index_and_has_column() {
        let mut t = Table::new_empty();
        let col = IntegerArray::<i64>::default();
        t.add_col(field_array("foo", Array::from_int64(col)));
        assert_eq!(t.col_name_index("foo"), Some(0));
        assert_eq!(t.col_name_index("bar"), None);
        assert!(t.has_col("foo"));
        assert!(!t.has_col("bar"));
    }

    #[test]
    fn test_remove_column_by_name_and_index() {
        let mut t = Table::new_empty();
        let mut col1 = IntegerArray::<u32>::default();
        col1.push(10);
        let mut col2 = BooleanArray::default();
        col2.push(true);

        t.add_col(field_array("a", Array::from_uint32(col1)));
        t.add_col(field_array("b", Array::from_bool(col2)));

        assert!(t.remove_col("a"));
        assert!(!t.has_col("a"));
        assert_eq!(t.n_cols(), 1);

        assert!(t.remove_col_at(0));
        assert_eq!(t.n_cols(), 0);
        assert_eq!(t.n_rows(), 0);

        // Removing non-existent column
        assert!(!t.remove_col("not_there"));
        assert!(!t.remove_col_at(5));
    }

    #[test]
    fn test_clear() {
        let mut t = Table::new_empty();
        let mut col = IntegerArray::<i32>::default();
        col.push(42); // Ensure at least one row
        t.add_col(field_array("x", Array::from_int32(col)));
        assert!(!t.is_empty());
        t.clear();
        assert!(t.is_empty());
        assert_eq!(t.n_cols(), 0);
        assert_eq!(t.n_rows(), 0);
    }

    #[test]
    fn test_columns() {
        let mut t = Table::new_empty();
        let mut col = IntegerArray::<i32>::default();
        col.push(7);
        t.add_col(field_array("c", Array::from_int32(col)));
        {
            let cols = t.cols();
            assert_eq!(cols.len(), 1);
        }
    }

    #[test]
    fn test_table_iter() {
        let mut t = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        t.add_col(field_array("a", Array::from_int32(col1)));
        let mut col2 = BooleanArray::default();
        col2.push(true);
        t.add_col(field_array("b", Array::from_bool(col2)));

        let names: Vec<_> = t.iter().map(|fa| fa.field.name.as_str()).collect();
        assert_eq!(names, ["a", "b"]);

        let names2: Vec<_> = (&t).into_iter().map(|fa| fa.field.name.as_str()).collect();
        assert_eq!(names2, ["a", "b"]);
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_table_slice_and_slice() {
        use crate::structs::field_array::field_array;
        use crate::{Array, BooleanArray, IntegerArray};

        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        col1.push(2);
        col1.push(3);
        let mut col2 = BooleanArray::default();
        col2.push(true);
        col2.push(false);
        col2.push(true);

        let mut t = Table::new("foo".into(), None);
        t.add_col(field_array("ints", Array::from_int32(col1)));
        t.add_col(field_array("bools", Array::from_bool(col2)));

        let sliced = t.slice_clone(1, 2);
        assert_eq!(sliced.n_rows(), 2);
        // Access column by name index
        let idx = sliced.col_name_index("ints").unwrap();
        assert_eq!(sliced.cols().get(idx).unwrap().array.len(), 2);

        let view = t.slice(1, 2);
        assert_eq!(view.n_rows(), 2);
        // TableV is a zero-copy view - underlying array still has full length
        // The view's logical length is accessed via n_rows()
        assert!(view.col_name_index("bools").is_some());

        // // Zero-copy: view.table == &t via the underlying arrays
        // for (orig, sliced) in t.cols.iter().zip(view.cols.iter()) {
        //     use std::sync::Arc;

        //     assert!(Arc::ptr_eq(&orig.field, &sliced.field), "FieldArc pointer mismatch");
        // }
    }

    #[test]
    fn test_map_cols_by_name() {
        let mut t = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        col1.push(2);
        let mut col2 = IntegerArray::<i32>::default();
        col2.push(3);
        col2.push(4);

        t.add_col(field_array("a", Array::from_int32(col1)));
        t.add_col(field_array("b", Array::from_int32(col2)));

        // Test with all valid names
        let results = t.map_cols_by_name(&["a", "b"], |fa| fa.field.name.clone());
        assert_eq!(results, vec!["a", "b"]);

        // Test with missing column (will warn but skip)
        let results = t.map_cols_by_name(&["a", "missing", "b"], |fa| fa.field.name.clone());
        assert_eq!(results, vec!["a", "b"]);
    }

    #[test]
    fn test_map_cols_by_index() {
        let mut t = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        col1.push(2);
        let mut col2 = IntegerArray::<i32>::default();
        col2.push(3);
        col2.push(4);

        t.add_col(field_array("a", Array::from_int32(col1)));
        t.add_col(field_array("b", Array::from_int32(col2)));

        // Test with all valid indices
        let results = t.map_cols_by_index(&[0, 1], |fa| fa.field.name.clone());
        assert_eq!(results, vec!["a", "b"]);

        // Test with out-of-bounds index (will warn but skip)
        let results = t.map_cols_by_index(&[0, 5, 1], |fa| fa.field.name.clone());
        assert_eq!(results, vec!["a", "b"]);
    }

    #[test]
    fn test_table_insert_rows_prepend() {
        let mut t1 = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        col1.push(2);
        let mut col2 = IntegerArray::<i32>::default();
        col2.push(10);
        col2.push(20);
        t1.add_col(field_array("a", Array::from_int32(col1)));
        t1.add_col(field_array("b", Array::from_int32(col2)));

        let mut t2 = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(99);
        let mut col2 = IntegerArray::<i32>::default();
        col2.push(88);
        t2.add_col(field_array("a", Array::from_int32(col1)));
        t2.add_col(field_array("b", Array::from_int32(col2)));

        t1.insert_rows(0, &t2).unwrap();

        assert_eq!(t1.n_rows(), 3);
        match &t1.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[99, 1, 2]);
            }
            _ => panic!("wrong type"),
        }
        match &t1.cols[1].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[88, 10, 20]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_table_insert_rows_middle() {
        let mut t1 = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        col1.push(2);
        col1.push(3);
        let mut col2 = IntegerArray::<i32>::default();
        col2.push(10);
        col2.push(20);
        col2.push(30);
        t1.add_col(field_array("a", Array::from_int32(col1)));
        t1.add_col(field_array("b", Array::from_int32(col2)));

        let mut t2 = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(99);
        col1.push(88);
        let mut col2 = IntegerArray::<i32>::default();
        col2.push(77);
        col2.push(66);
        t2.add_col(field_array("a", Array::from_int32(col1)));
        t2.add_col(field_array("b", Array::from_int32(col2)));

        t1.insert_rows(1, &t2).unwrap();

        assert_eq!(t1.n_rows(), 5);
        match &t1.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 99, 88, 2, 3]);
            }
            _ => panic!("wrong type"),
        }
        match &t1.cols[1].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[10, 77, 66, 20, 30]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_table_insert_rows_append() {
        let mut t1 = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        col1.push(2);
        t1.add_col(field_array("a", Array::from_int32(col1)));

        let mut t2 = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(3);
        col1.push(4);
        t2.add_col(field_array("a", Array::from_int32(col1)));

        t1.insert_rows(2, &t2).unwrap();

        assert_eq!(t1.n_rows(), 4);
        match &t1.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 2, 3, 4]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_table_insert_rows_schema_mismatch() {
        let mut t1 = Table::new_empty();
        let col1 = IntegerArray::<i32>::default();
        t1.add_col(field_array("a", Array::from_int32(col1)));

        let mut t2 = Table::new_empty();
        let col1 = IntegerArray::<i32>::default();
        t2.add_col(field_array("b", Array::from_int32(col1)));

        let result = t1.insert_rows(0, &t2);
        assert!(result.is_err());
    }

    #[test]
    fn test_table_insert_rows_out_of_bounds() {
        let mut t1 = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        t1.add_col(field_array("a", Array::from_int32(col1)));

        let t2 = Table::new_empty();
        let result = t1.insert_rows(10, &t2);
        assert!(result.is_err());
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_table_split_basic() {
        let mut t = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        col1.push(2);
        col1.push(3);
        col1.push(4);
        let mut col2 = IntegerArray::<i32>::default();
        col2.push(10);
        col2.push(20);
        col2.push(30);
        col2.push(40);
        t.add_col(field_array("a", Array::from_int32(col1)));
        t.add_col(field_array("b", Array::from_int32(col2)));

        let super_table = t.split(2).unwrap();

        assert_eq!(super_table.n_batches(), 2);
        assert_eq!(super_table.batches[0].n_rows(), 2);
        assert_eq!(super_table.batches[1].n_rows(), 2);

        match &super_table.batches[0].cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 2]);
            }
            _ => panic!("wrong type"),
        }

        match &super_table.batches[1].cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[3, 4]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[cfg(feature = "chunked")]
    #[test]
    fn test_table_split_invalid_index() {
        let mut t1 = Table::new_empty();
        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        col1.push(2);
        t1.add_col(field_array("a", Array::from_int32(col1.clone())));
        assert!(t1.split(0).is_err());

        let mut t2 = Table::new_empty();
        t2.add_col(field_array("a", Array::from_int32(col1.clone())));
        assert!(t2.split(2).is_err());

        let mut t3 = Table::new_empty();
        t3.add_col(field_array("a", Array::from_int32(col1)));
        assert!(t3.split(10).is_err());
    }
}

#[cfg(test)]
#[cfg(feature = "parallel_proc")]
mod parallel_column_tests {
    use rayon::prelude::*;

    use super::*;
    use crate::structs::field_array::field_array;
    use crate::{Array, BooleanArray, IntegerArray, MaskedArray};

    #[test]
    fn test_table_par_iter_column_names() {
        let mut table = Table::new_empty();

        let mut col1 = IntegerArray::<i32>::default();
        col1.push(1);
        let mut col2 = BooleanArray::default();
        col2.push(true);

        table.add_col(field_array("id", Array::from_int32(col1)));
        table.add_col(field_array("flag", Array::from_bool(col2)));

        let mut names: Vec<&str> = table.par_iter().map(|fa| fa.field.name.as_str()).collect();
        names.sort_unstable(); // Ensure deterministic order for assert
        assert_eq!(names, vec!["flag", "id"]);
    }
}
