//! # **Matrix Module** - *De-facto Matrix Memory Layout for BLAS/LAPACK ecosystem compatibility*
//!
//! Dense column-major matrix type for high-performance linear algebra.
//! BLAS/LAPACK compatible with built-inconversions from `Table` data.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::traits::{concatenate::Concatenate, shape::Shape};
use crate::{FloatArray, Vec64};

// Global counter for unnamed matrix instances
static UNNAMED_MATRIX_COUNTER: AtomicUsize = AtomicUsize::new(1);

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
/// - `n_rows`: Number of rows.
/// - `n_cols`: Number of columns.
/// - `data`: Flat buffer in column-major order.
/// - `name`: Optional matrix name (used for debugging, diagnostics, or pretty printing).
///
/// ### Null handling
/// - It is dense - nulls can be represented through `f64::NAN`
/// - However this is not always reliable, as a single *NaN* can affect vectorised
/// calculations when integrating with various frameworks.
///
/// ### Under Development
/// ⚠️ **Unstable API and WIP: expect future development. Breaking changes will be minimised,
/// but avoid using this in production unless you are ready to wear API adjustments**.
/// Specifically, we are considering whether to make a 'logical columns' matrix for easy
/// access, but backed by a single buffer. This would provide the look/feel of a regular table
/// whilst keeping the implementation efficient and consistent with established frameworks,
/// at the cost of immutability. Consider this change likely.
#[repr(C, align(64))]
#[derive(Clone, PartialEq)]
pub struct Matrix {
    pub n_rows: usize,
    pub n_cols: usize,
    pub data: Vec64<f64>,
    pub name: String,
}

impl Matrix {
    /// Constructs a new dense Matrix with shape and optional name.
    /// Data buffer is zeroed.
    pub fn new(n_rows: usize, n_cols: usize, name: Option<String>) -> Self {
        let len = n_rows * n_cols;
        let mut data = Vec64::with_capacity(len);
        data.0.resize(len, 0.0);
        let name = name.unwrap_or_else(|| {
            let id = UNNAMED_MATRIX_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("UnnamedMatrix{}", id)
        });
        Matrix {
            n_rows,
            n_cols,
            data,
            name,
        }
    }

    /// Constructs a Matrix from a flat buffer (must be column-major order).
    /// Panics if data length does not match shape.
    pub fn from_flat(data: Vec64<f64>, n_rows: usize, n_cols: usize, name: Option<String>) -> Self {
        assert_eq!(
            data.len(),
            n_rows * n_cols,
            "Matrix shape does not match buffer length"
        );
        let name = name.unwrap_or_else(|| {
            let id = UNNAMED_MATRIX_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("UnnamedMatrix{}", id)
        });
        Matrix {
            n_rows,
            n_cols,
            data,
            name,
        }
    }

    /// Returns the value at (row, col) (0-based). Panics if out of bounds.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < self.n_rows, "Row out of bounds");
        debug_assert!(col < self.n_cols, "Col out of bounds");
        self.data[col * self.n_rows + row]
    }

    /// Sets the value at (row, col) (0-based). Panics if out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        debug_assert!(row < self.n_rows, "Row out of bounds");
        debug_assert!(col < self.n_cols, "Col out of bounds");
        self.data[col * self.n_rows + row] = value;
    }

    /// Returns true if the matrix is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_rows == 0 || self.n_cols == 0
    }

    /// Returns the total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns an immutable reference to the flat buffer.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// Returns a mutable reference to the flat buffer.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Returns a view of the matrix as a slice of columns.
    pub fn columns(&self) -> Vec<&[f64]> {
        (0..self.n_cols)
            .map(|col| &self.data[(col * self.n_rows)..((col + 1) * self.n_rows)])
            .collect()
    }

    /// Returns a vector of mutable slices, each corresponding to a column of the matrix.
    pub fn columns_mut(&mut self) -> Vec<&mut [f64]> {
        let n_rows = self.n_rows;
        let n_cols = self.n_cols;
        let ptr = self.data.as_mut_slice().as_mut_ptr();
        let mut result = Vec::with_capacity(n_cols);

        for col in 0..n_cols {
            let start = col * n_rows;
            // SAFETY:
            // - Each slice is within bounds and non-overlapping,
            // - We have exclusive &mut access to self.
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
        &self.data[(col * self.n_rows)..((col + 1) * self.n_rows)]
    }

    /// Returns a single column as a mutable slice, panics if col out of bounds.
    #[inline]
    pub fn col_mut(&mut self, col: usize) -> &mut [f64] {
        debug_assert!(col < self.n_cols, "Col out of bounds");
        &mut self.data[(col * self.n_rows)..((col + 1) * self.n_rows)]
    }

    /// Returns a single row as an owned Vec.
    #[inline]
    pub fn row(&self, row: usize) -> Vec<f64> {
        debug_assert!(row < self.n_rows, "Row out of bounds");
        (0..self.n_cols).map(|col| self.get(row, col)).collect()
    }

    /// Renames the matrix
    #[inline]
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
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
    /// Concatenates two matrices vertically (row-wise stacking).
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
                Some(format!("{}+{}", self.name, other.name)),
            ));
        }

        let result_n_rows = self.n_rows + other.n_rows;
        let result_n_cols = self.n_cols;
        let mut result_data = Vec64::with_capacity(result_n_rows * result_n_cols);

        // For each column, concatenate self's column with other's column
        // Since data is stored column-major, each column is contiguous
        for col in 0..result_n_cols {
            // Copy self's column
            let self_col_start = col * self.n_rows;
            let self_col_end = self_col_start + self.n_rows;
            result_data.extend_from_slice(&self.data[self_col_start..self_col_end]);

            // Copy other's column
            let other_col_start = col * other.n_rows;
            let other_col_end = other_col_start + other.n_rows;
            result_data.extend_from_slice(&other.data[other_col_start..other_col_end]);
        }

        Ok(Matrix {
            n_rows: result_n_rows,
            n_cols: result_n_cols,
            data: result_data,
            name: format!("{}+{}", self.name, other.name),
        })
    }
}

// Pretty print
impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Matrix '{}': {} × {} [col-major]",
            self.name, self.n_rows, self.n_cols
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

// From Vec<FloatArray<f64>> to Matrix (all cols must match length)
impl From<(Vec<FloatArray<f64>>, String)> for Matrix {
    fn from((columns, name): (Vec<FloatArray<f64>>, String)) -> Self {
        let n_cols = columns.len();
        let n_rows = columns.first().map(|c| c.data.len()).unwrap_or(0);
        for col in &columns {
            assert_eq!(col.data.len(), n_rows, "Column length mismatch");
        }
        let mut data = Vec64::with_capacity(n_rows * n_cols);
        for col in &columns {
            data.extend_from_slice(&col.data);
        }
        Matrix {
            n_rows,
            n_cols,
            data,
            name,
        }
    }
}

// From &[FloatArray<f64>] to Matrix
impl From<&[FloatArray<f64>]> for Matrix {
    fn from(columns: &[FloatArray<f64>]) -> Self {
        let n_cols = columns.len();
        let n_rows = columns.first().map(|c| c.data.len()).unwrap_or(0);
        for col in columns {
            assert_eq!(col.data.len(), n_rows, "Column length mismatch");
        }
        let mut data = Vec64::with_capacity(n_rows * n_cols);
        for col in columns {
            data.extend_from_slice(&col.data);
        }
        let name = {
            let id = UNNAMED_MATRIX_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("UnnamedMatrix{}", id)
        };
        Matrix {
            n_rows,
            n_cols,
            data,
            name,
        }
    }
}

// TODO: Fix
// impl TryFrom<&Table> for Matrix {
//     type Error = MinarrowError;

//     fn try_from(table: &Table) -> Result<Self, Self::Error> {
//         let name = table.name.clone();
//         let n_cols = table.n_cols();
//         let n_rows = table.n_rows();

//         // Collect and check columns
//         let mut float_columns = Vec::with_capacity(n_cols);
//         for fa in &table.cols {
//             let numeric_array = fa.array.num();
//             let arr: FloatArray<f64> = numeric_array.f64()?;
//             float_columns.push(arr);
//         }

//         // Ensure all columns are the correct length
//         for (col_idx, col) in float_columns.iter().enumerate() {
//             if col.data.len() != n_rows {
//                 return Err(MinarrowError::ColumnLengthMismatch {
//                     col: col_idx,
//                     expected: n_rows,
//                     found: col.data.len()
//                 });
//             }
//         }

//         // Flatten into single column-major Vec64<f64>
//         let mut data = Vec64::with_capacity(n_rows * n_cols);
//         for col in &float_columns {
//             data.0.extend_from_slice(&col.data);
//         }

//         Ok(Matrix { n_rows, n_cols, data, name })
//     }
// }

// From Vec<Vec<f64>> (Vec-of-cols) to Matrix (anonymous name)
impl From<&[Vec<f64>]> for Matrix {
    fn from(columns: &[Vec<f64>]) -> Self {
        let n_cols = columns.len();
        let n_rows = columns.first().map(|c| c.len()).unwrap_or(0);
        for col in columns {
            assert_eq!(col.len(), n_rows, "Column length mismatch");
        }
        let mut data = Vec64::with_capacity(n_rows * n_cols);
        for col in columns {
            data.extend_from_slice(col);
        }
        let name = {
            let id = UNNAMED_MATRIX_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("UnnamedMatrix{}", id)
        };
        Matrix {
            n_rows,
            n_cols,
            data,
            name,
        }
    }
}

// From flat slice with shape
impl<'a> From<(&'a [f64], usize, usize, Option<String>)> for Matrix {
    fn from((slice, n_rows, n_cols, name): (&'a [f64], usize, usize, Option<String>)) -> Self {
        assert_eq!(slice.len(), n_rows * n_cols, "Slice shape mismatch");
        let data = Vec64::from(slice);
        let name = name.unwrap_or_else(|| {
            let id = UNNAMED_MATRIX_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("UnnamedMatrix{}", id)
        });
        Matrix {
            n_rows,
            n_cols,
            data,
            name,
        }
    }
}

// ===================== Iterators ======================

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
