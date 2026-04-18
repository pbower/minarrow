// Copyright 2025 Peter Garfield Bower
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # **Matrix Module** - *De-facto Matrix Memory Layout for BLAS/LAPACK ecosystem compatibility*
//!
//! Dense column-major matrix type for high-performance linear algebra.
//! BLAS/LAPACK compatible with built-inconversions from `Table` data.

use std::fmt;
use std::sync::Arc;

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::structs::buffer::Buffer;
use crate::structs::shared_buffer::SharedBuffer;
use crate::traits::{concatenate::Concatenate, shape::Shape};
use crate::{Array, Field, FieldArray, FloatArray, NumericArray, Table, Vec64};

/// # Matrix
///
/// Column-major dense matrix.
///
/// ### Description
/// This struct is compatible with Arrow, LAPACK, BLAS, and all
/// column-major numeric routines.
///
/// **This is an optional extra enabled by the `matrix` feature,
/// and is not part of the *`Apache Arrow`* framework**.
///
/// ### Properties
/// - `n_rows`: Logical number of rows.
/// - `n_cols`: Number of columns.
/// - `stride`: Physical elements per column in the buffer. Padded to 8-element
///   (64-byte) boundaries so every column starts SIMD-aligned. This is the
///   BLAS leading dimension (lda). Always `>= n_rows`.
/// - `data`: Flat buffer in column-major order with stride padding.
/// - `name`: Optional matrix name for diagnostics.
///
/// ### Null handling
/// - It is dense - nulls can be represented through `f64::NAN`
/// - However this is not always reliable, as a single *NaN* can affect vectorised
/// calculations when integrating with various frameworks.
#[repr(C, align(64))]
#[derive(Clone, PartialEq)]
pub struct Matrix {
    pub n_rows: usize,
    pub n_cols: usize,
    /// Physical column stride in elements, padded so each column is 64-byte aligned.
    pub stride: usize,
    pub data: Vec64<f64>,
    pub name: Option<String>,
}

/// Number of f64 elements per 64-byte alignment boundary.
const ALIGN_ELEMS: usize = 64 / std::mem::size_of::<f64>(); // 8

/// Round up to next multiple of ALIGN_ELEMS for 64-byte column alignment.
#[inline]
const fn aligned_stride(n_rows: usize) -> usize {
    (n_rows + ALIGN_ELEMS - 1) & !(ALIGN_ELEMS - 1)
}

impl Matrix {
    /// Constructs a new dense Matrix with shape and optional name.
    /// Data buffer is zeroed. Columns are padded to 64-byte alignment.
    pub fn new(n_rows: usize, n_cols: usize, name: Option<String>) -> Self {
        let stride = aligned_stride(n_rows);
        let len = stride * n_cols;
        let mut data = Vec64::with_capacity(len);
        data.0.resize(len, 0.0);
        Matrix { n_rows, n_cols, stride, data, name }
    }

    /// Constructs a Matrix from a pre-padded Vec64 buffer.
    /// The buffer must already have `stride * n_cols` elements with the correct
    /// stride layout. Use `from_f64_unaligned` if your data is unpadded.
    pub fn from_f64_aligned(data: Vec64<f64>, n_rows: usize, n_cols: usize, name: Option<String>) -> Self {
        let stride = aligned_stride(n_rows);
        assert_eq!(
            data.len(),
            stride * n_cols,
            "Matrix: padded buffer length does not match stride * n_cols"
        );
        Matrix { n_rows, n_cols, stride, data, name }
    }

    /// Constructs a Matrix from a flat column-major buffer without stride padding.
    /// The data is re-laid out with 64-byte aligned column padding.
    pub fn from_f64_unaligned(src: &[f64], n_rows: usize, n_cols: usize, name: Option<String>) -> Self {
        assert_eq!(
            src.len(),
            n_rows * n_cols,
            "Matrix shape does not match buffer length"
        );
        let stride = aligned_stride(n_rows);
        if stride == n_rows {
            // No padding needed, take ownership directly
            let data = Vec64::from(src);
            return Matrix { n_rows, n_cols, stride, data, name };
        }
        // Re-layout with padding between columns
        let mut data = Vec64::with_capacity(stride * n_cols);
        data.0.resize(stride * n_cols, 0.0);
        for col in 0..n_cols {
            let src_start = col * n_rows;
            let dst_start = col * stride;
            data.as_mut_slice()[dst_start..dst_start + n_rows]
                .copy_from_slice(&src[src_start..src_start + n_rows]);
        }
        Matrix { n_rows, n_cols, stride, data, name }
    }

    /// Constructs a Matrix from a slice of `FloatArray<f64>` columns. Each
    /// column becomes one Matrix column; null masks are rejected so Matrix
    /// stays dense.
    ///
    /// `impl AsRef<[FloatArray<f64>]>` accepts any contiguous layout at the
    /// call-site: `&[FloatArray<f64>]`, `&Vec<FloatArray<f64>>`,
    /// `[FloatArray<f64>; N]`, etc. 
    ///
    /// All columns must have the same length. Columns are copied into a
    /// 64-byte-aligned column-major buffer with stride padding.
    ///
    /// For existing `Table` / `TableV` containers prefer `Matrix::try_from`.
    /// This constructor targets direct construction from already-built column
    /// arrays.
    pub fn try_from_cols(
        cols: impl AsRef<[FloatArray<f64>]>,
        name: Option<String>,
    ) -> Result<Self, MinarrowError> {
        let columns = cols.as_ref();
        if columns.is_empty() {
            return Err(MinarrowError::ShapeError {
                message: "Matrix::try_from_cols requires at least one column".into(),
            });
        }

        let n_rows = columns[0].data.len();
        for (i, col) in columns.iter().enumerate() {
            if col.data.len() != n_rows {
                return Err(MinarrowError::ColumnLengthMismatch {
                    col: i,
                    expected: n_rows,
                    found: col.data.len(),
                });
            }
            // `None` short-circuits without touching the mask. 
            // A present-but-empty mask pays a single popcount.
            if col.null_mask.as_ref().map_or(false, |m| m.has_nulls()) {
                return Err(MinarrowError::NullError {
                    message: Some(format!(
                        "Matrix::try_from_cols: column {i} contains null values; Matrix requires dense data"
                    )),
                });
            }
        }

        let n_cols = columns.len();
        let stride = aligned_stride(n_rows);
        let pad = stride - n_rows;
        let mut data = Vec64::with_capacity(stride * n_cols);

        // Each column is a guaranteed-dense slice, so
        // extend_from_slice is a straight memcpy into the column-major buffer.
        for col in columns {
            let col_slice: &[f64] = col.data.as_slice();
            data.extend_from_slice(col_slice);
            if pad > 0 {
                data.extend_from_slice(&[0.0; ALIGN_ELEMS][..pad]);
            }
        }
        Ok(Matrix { n_rows, n_cols, stride, data, name })
    }

    /// Returns the value at (row, col) (0-based). Panics if out of bounds.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < self.n_rows, "Row out of bounds");
        debug_assert!(col < self.n_cols, "Col out of bounds");
        self.data[col * self.stride + row]
    }

    /// Sets the value at (row, col) (0-based). Panics if out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        debug_assert!(row < self.n_rows, "Row out of bounds");
        debug_assert!(col < self.n_cols, "Col out of bounds");
        self.data[col * self.stride + row] = value;
    }

    /// Returns true if the matrix is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_rows == 0 || self.n_cols == 0
    }

    /// Returns the logical number of elements (n_rows * n_cols), not including padding.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_rows * self.n_cols
    }

    /// Returns an immutable reference to the full flat buffer including padding.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Returns a mutable reference to the full flat buffer including padding.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Returns a view of the matrix as a slice of columns (logical rows only, no padding).
    pub fn columns(&self) -> Vec<&[f64]> {
        (0..self.n_cols)
            .map(|col| &self.data[(col * self.stride)..(col * self.stride + self.n_rows)])
            .collect()
    }

    /// Returns a vector of mutable slices, each corresponding to a column.
    pub fn columns_mut(&mut self) -> Vec<&mut [f64]> {
        let n_rows = self.n_rows;
        let stride = self.stride;
        let n_cols = self.n_cols;
        let ptr = self.data.as_mut_slice().as_mut_ptr();
        let mut result = Vec::with_capacity(n_cols);

        for col in 0..n_cols {
            let start = col * stride;
            // SAFETY: each slice is within bounds and non-overlapping,
            // we have exclusive &mut access to self.
            unsafe {
                let col_ptr = ptr.add(start);
                let slice = std::slice::from_raw_parts_mut(col_ptr, n_rows);
                result.push(slice);
            }
        }
        result
    }

    /// Returns a single column as a slice, panics if col out of bounds.
    #[inline]
    pub fn col(&self, col: usize) -> &[f64] {
        debug_assert!(col < self.n_cols, "Col out of bounds");
        &self.data[(col * self.stride)..(col * self.stride + self.n_rows)]
    }

    /// Returns a single column as a mutable slice, panics if col out of bounds.
    #[inline]
    pub fn col_mut(&mut self, col: usize) -> &mut [f64] {
        debug_assert!(col < self.n_cols, "Col out of bounds");
        let start = col * self.stride;
        &mut self.data[start..start + self.n_rows]
    }

    /// Returns a single row as an owned Vec.
    #[inline]
    pub fn row(&self, row: usize) -> Vec<f64> {
        debug_assert!(row < self.n_rows, "Row out of bounds");
        (0..self.n_cols).map(|col| self.data[col * self.stride + row]).collect()
    }

    /// Transpose this matrix, returning a new Matrix with rows and columns swapped.
    /// The result has stride-aligned columns for SIMD access.
    pub fn transpose(&self) -> Self {
        let mut dst = Matrix::new(self.n_cols, self.n_rows, self.name.clone());
        for j in 0..self.n_cols {
            for i in 0..self.n_rows {
                dst.data[i * dst.stride + j] = self.data[j * self.stride + i];
            }
        }
        dst
    }

    /// Extract rows by index into a new Matrix.
    /// Column stride alignment is maintained in the result.
    pub fn extract_rows(&self, indices: &[usize]) -> Self {
        let n_new = indices.len();
        let mut dst = Matrix::new(n_new, self.n_cols, self.name.clone());
        for j in 0..self.n_cols {
            let src_col = self.col(j);
            let dst_col = dst.col_mut(j);
            for (k, &idx) in indices.iter().enumerate() {
                dst_col[k] = src_col[idx];
            }
        }
        dst
    }

    /// Sets the matrix name.
    #[inline]
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Returns the number of columns.
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Returns the number of rows.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    // ********************** BLAS/LAPACK Compatibility **************

    /// Number of rows as i32 for BLAS parameter passing.
    #[inline]
    pub fn m(&self) -> i32 {
        self.n_rows as i32
    }

    /// Number of columns as i32 for BLAS parameter passing.
    #[inline]
    pub fn n(&self) -> i32 {
        self.n_cols as i32
    }

    /// Leading dimension for BLAS - equals stride, which is n_rows padded to
    /// 64-byte alignment. Pass this as the `lda` parameter to all BLAS/LAPACK calls.
    #[inline]
    pub fn lda(&self) -> i32 {
        self.stride as i32
    }

    // ********************** Table conversion **********************

    /// Convert this Matrix into a Table with zero-copy column sharing.
    ///
    /// The matrix data buffer is frozen into a SharedBuffer, and each column
    /// becomes a FloatArray backed by a window into that shared allocation.
    /// No data is copied.
    ///
    /// `fields` must have exactly `n_cols` entries, providing the name and
    /// metadata for each column.
    pub fn to_table(self, fields: Vec<Field>) -> Result<Table, MinarrowError> {
        if fields.len() != self.n_cols {
            return Err(MinarrowError::ShapeError {
                message: format!(
                    "to_table: expected {} fields for {} columns, got {}",
                    self.n_cols, self.n_cols, fields.len()
                ),
            });
        }

        let n_rows = self.n_rows;
        let n_cols = self.n_cols;
        let stride = self.stride;

        // Freeze the Vec64<f64> into a SharedBuffer (zero-copy, refcounted)
        // SAFETY: f64 is plain data with no drop logic
        let shared = unsafe { SharedBuffer::from_vec64_typed(self.data) };

        let mut cols = Vec::with_capacity(n_cols);
        for (i, field) in fields.into_iter().enumerate() {
            // Each column starts at i * stride elements, which is 64-byte aligned
            let col_offset = i * stride;
            let buf: Buffer<f64> = Buffer::from_shared_column(shared.clone(), col_offset, n_rows);
            let float_arr = FloatArray::new(buf, None);
            let array = Array::NumericArray(NumericArray::Float64(Arc::new(float_arr)));
            cols.push(FieldArray::new(field, array));
        }

        Ok(Table::new(self.name.unwrap_or_default(), Some(cols)))
    }

    /// Convert this Matrix into a Table using auto-generated column names
    /// (col_0, col_1, ...).
    pub fn to_table_gen(self) -> Table {
        let n_cols = self.n_cols;
        let fields: Vec<Field> = (0..n_cols)
            .map(|i| Field::new(format!("col_{}", i), crate::ffi::arrow_dtype::ArrowType::Float64, false, None))
            .collect();
        self.to_table(fields).unwrap()
    }
}

impl Shape for Matrix {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank2 {
            rows: self.n_rows(),
            cols: self.n_cols(),
        }
    }
}

impl Concatenate for Matrix {
    /// Concatenates two matrices vertically (i.e., row-wise stacking).
    ///
    /// # Requirements
    /// - Both matrices must have the same number of columns
    ///
    /// # Returns
    /// A new Matrix with rows from `self` followed by rows from `other`
    ///
    /// # Errors
    /// - `IncompatibleTypeError` if column counts don't match
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Check column count
        if self.n_cols != other.n_cols {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "Matrix",
                to: "Matrix",
                message: Some(format!(
                    "Cannot concatenate matrices with different column counts: {} vs {}",
                    self.n_cols, other.n_cols
                )),
            });
        }

        // Handle empty matrices
        if self.is_empty() && other.is_empty() {
            return Ok(Matrix::new(
                0,
                0,
                None,
            ));
        }

        let result_n_rows = self.n_rows + other.n_rows;
        let result_n_cols = self.n_cols;
        let result_stride = aligned_stride(result_n_rows);
        let pad = result_stride - result_n_rows;
        let mut result_data = Vec64::with_capacity(result_stride * result_n_cols);

        for col in 0..result_n_cols {
            result_data.extend_from_slice(self.col(col));
            result_data.extend_from_slice(other.col(col));
            if pad > 0 {
                result_data.extend_from_slice(&[0.0; ALIGN_ELEMS][..pad]);
            }
        }

        Ok(Matrix {
            n_rows: result_n_rows,
            n_cols: result_n_cols,
            stride: result_stride,
            data: result_data,
            name: None,
        })
    }
}

// Pretty print
impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Matrix{}: {} × {} [col-major]",
            self.name.as_deref().map_or(String::new(), |n| format!(" '{}'", n)),
            self.n_rows, self.n_cols
        )?;
        for row in 0..self.n_rows.min(6) {
            // Print up to 6 rows
            write!(f, "\n[")?;
            for col in 0..self.n_cols.min(8) {
                // Print up to 8 cols
                write!(f, " {:8.4}", self.get(row, col))?;
                if col != self.n_cols - 1 {
                    write!(f, ",")?;
                }
            }
            if self.n_cols > 8 {
                write!(f, " ...")?;
            }
            write!(f, " ]")?;
        }
        if self.n_rows > 6 {
            write!(f, "\n...")?;
        }
        Ok(())
    }
}

// From Vec<FloatArray<f64>> to unnamed Matrix
impl From<Vec<FloatArray<f64>>> for Matrix {
    fn from(columns: Vec<FloatArray<f64>>) -> Self {
        let n_cols = columns.len();
        let n_rows = columns.first().map(|c| c.data.len()).unwrap_or(0);
        let stride = aligned_stride(n_rows);
        let pad = stride - n_rows;
        for col in &columns {
            assert_eq!(col.data.len(), n_rows, "Column length mismatch");
        }
        let mut data = Vec64::with_capacity(stride * n_cols);
        for col in &columns {
            data.extend_from_slice(&col.data);
            if pad > 0 {
                data.extend_from_slice(&[0.0; ALIGN_ELEMS][..pad]);
            }
        }
        Matrix { n_rows, n_cols, stride, data, name: None }
    }
}

// From (Vec<FloatArray<f64>>, String) to named Matrix
impl From<(Vec<FloatArray<f64>>, String)> for Matrix {
    fn from((columns, name): (Vec<FloatArray<f64>>, String)) -> Self {
        let mut mat = Matrix::from(columns);
        mat.name = Some(name);
        mat
    }
}

// From &[FloatArray<f64>] to unnamed Matrix
impl From<&[FloatArray<f64>]> for Matrix {
    fn from(columns: &[FloatArray<f64>]) -> Self {
        let n_cols = columns.len();
        let n_rows = columns.first().map(|c| c.data.len()).unwrap_or(0);
        let stride = aligned_stride(n_rows);
        let pad = stride - n_rows;
        for col in columns {
            assert_eq!(col.data.len(), n_rows, "Column length mismatch");
        }
        let mut data = Vec64::with_capacity(stride * n_cols);
        for col in columns {
            data.extend_from_slice(&col.data);
            if pad > 0 {
                data.extend_from_slice(&[0.0; ALIGN_ELEMS][..pad]);
            }
        }
        Matrix { n_rows, n_cols, stride, data, name: None }
    }
}

impl TryFrom<&Table> for Matrix {
    type Error = MinarrowError;

    fn try_from(table: &Table) -> Result<Self, Self::Error> {
        let name = if table.name.is_empty() { None } else { Some(table.name.clone()) };
        let n_cols = table.n_cols();
        let n_rows = table.n_rows;
        let stride = aligned_stride(n_rows);
        let pad = stride - n_rows;

        let mut data = Vec64::with_capacity(stride * n_cols);
        for (col_idx, fa) in table.cols.iter().enumerate() {
            let numeric = fa.array.num_ref().map_err(|_| MinarrowError::TypeError {
                from: "non-numeric",
                to: "Float64",
                message: Some(format!("column {} is not numeric", col_idx)),
            })?;
            let f64_arr = numeric.clone().f64()?;
            if f64_arr.data.len() != n_rows {
                return Err(MinarrowError::ColumnLengthMismatch {
                    col: col_idx,
                    expected: n_rows,
                    found: f64_arr.data.len(),
                });
            }
            data.extend_from_slice(f64_arr.data.as_slice());
            if pad > 0 {
                data.extend_from_slice(&[0.0; ALIGN_ELEMS][..pad]);
            }
        }

        Ok(Matrix { n_rows, n_cols, stride, data, name })
    }
}

impl TryFrom<Table> for Matrix {
    type Error = MinarrowError;

    fn try_from(table: Table) -> Result<Self, Self::Error> {
        Matrix::try_from(&table)
    }
}

// From &[Vec<f64>] (Vec-of-cols) to unnamed Matrix
impl From<&[Vec<f64>]> for Matrix {
    fn from(columns: &[Vec<f64>]) -> Self {
        let n_cols = columns.len();
        let n_rows = columns.first().map(|c| c.len()).unwrap_or(0);
        let stride = aligned_stride(n_rows);
        let pad = stride - n_rows;
        for col in columns {
            assert_eq!(col.len(), n_rows, "Column length mismatch");
        }
        let mut data = Vec64::with_capacity(stride * n_cols);
        for col in columns {
            data.extend_from_slice(col);
            if pad > 0 {
                data.extend_from_slice(&[0.0; ALIGN_ELEMS][..pad]);
            }
        }
        Matrix { n_rows, n_cols, stride, data, name: None }
    }
}

// From flat unpadded slice with shape - re-lays out with stride padding
impl<'a> From<(&'a [f64], usize, usize, Option<String>)> for Matrix {
    fn from((slice, n_rows, n_cols, name): (&'a [f64], usize, usize, Option<String>)) -> Self {
        assert_eq!(slice.len(), n_rows * n_cols, "Slice shape mismatch");
        Matrix::from_f64_unaligned(slice, n_rows, n_cols, name)
    }
}

// ********************** Iterators ***********************

impl<'a> IntoIterator for &'a Matrix {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a> IntoIterator for &'a mut Matrix {
    type Item = &'a mut f64;
    type IntoIter = std::slice::IterMut<'a, f64>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl IntoIterator for Matrix {
    type Item = f64;
    type IntoIter = <Vec64<f64> as IntoIterator>::IntoIter;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}
