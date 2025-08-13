//! # Matrix Module - *De-facto Matrix Memory Layout for BLAS/LAPACK ecosystem compatibility*
//! 
//! Dense column-major matrix type for high-performance linear algebra.
//! BLAS/LAPACK compatible with built-inconversions from `Table` data.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::Table;
use crate::enums::error::MinarrowError;
use crate::{FloatArray, Vec64};

// TODO: Leave out of the prod version

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
/// - `nrows`: Number of rows.
/// - `ncols`: Number of columns.
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
    pub nrows: usize,
    pub ncols: usize,
    pub data: Vec64<f64>,
    pub name: String
}

impl Matrix {
    /// Constructs a new dense Matrix with shape and optional name.
    /// Data buffer is zeroed.
    pub fn new(nrows: usize, ncols: usize, name: Option<String>) -> Self {
        let len = nrows * ncols;
        let mut data = Vec64::with_capacity(len);
        data.0.resize(len, 0.0);
        let name = name.unwrap_or_else(|| {
            let id = UNNAMED_MATRIX_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("UnnamedMatrix{}", id)
        });
        Matrix { nrows, ncols, data, name }
    }

    /// Constructs a Matrix from a flat buffer (must be column-major order).
    /// Panics if data length does not match shape.
    pub fn from_flat(data: Vec64<f64>, nrows: usize, ncols: usize, name: Option<String>) -> Self {
        assert_eq!(data.len(), nrows * ncols, "Matrix shape does not match buffer length");
        let name = name.unwrap_or_else(|| {
            let id = UNNAMED_MATRIX_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("UnnamedMatrix{}", id)
        });
        Matrix { nrows, ncols, data, name }
    }

    /// Returns the value at (row, col) (0-based). Panics if out of bounds.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < self.nrows, "Row out of bounds");
        debug_assert!(col < self.ncols, "Col out of bounds");
        self.data[col * self.nrows + row]
    }

    /// Sets the value at (row, col) (0-based). Panics if out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        debug_assert!(row < self.nrows, "Row out of bounds");
        debug_assert!(col < self.ncols, "Col out of bounds");
        self.data[col * self.nrows + row] = value;
    }

    /// Returns true if the matrix is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nrows == 0 || self.ncols == 0
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
        (0..self.ncols)
            .map(|col| &self.data[(col * self.nrows)..((col + 1) * self.nrows)])
            .collect()
    }

    /// Returns a vector of mutable slices, each corresponding to a column of the matrix.
    pub fn columns_mut(&mut self) -> Vec<&mut [f64]> {
        let nrows = self.nrows;
        let ncols = self.ncols;
        let ptr = self.data.as_mut_slice().as_mut_ptr();
        let mut result = Vec::with_capacity(ncols);

        for col in 0..ncols {
            let start = col * nrows;
            // SAFETY:
            // - Each slice is within bounds and non-overlapping,
            // - We have exclusive &mut access to self.
            unsafe {
                let col_ptr = ptr.add(start);
                let slice = std::slice::from_raw_parts_mut(col_ptr, nrows);
                result.push(slice);
            }
        }
        result
    }

    /// Returns a single column as a slice, panics if col out of bounds.
    #[inline]
    pub fn col(&self, col: usize) -> &[f64] {
        debug_assert!(col < self.ncols, "Col out of bounds");
        &self.data[(col * self.nrows)..((col + 1) * self.nrows)]
    }

    /// Returns a single column as a mutable slice, panics if col out of bounds.
    #[inline]
    pub fn col_mut(&mut self, col: usize) -> &mut [f64] {
        debug_assert!(col < self.ncols, "Col out of bounds");
        &mut self.data[(col * self.nrows)..((col + 1) * self.nrows)]
    }

    /// Returns a single row as an owned Vec.
    #[inline]
    pub fn row(&self, row: usize) -> Vec<f64> {
        debug_assert!(row < self.nrows, "Row out of bounds");
        (0..self.ncols).map(|col| self.get(row, col)).collect()
    }

    /// Renames the matrix
    #[inline]
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }
}

// Pretty print
impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Matrix '{}': {} × {} [col-major]", self.name, self.nrows, self.ncols)?;
        for row in 0..self.nrows.min(6) {
            // Print up to 6 rows
            write!(f, "\n[")?;
            for col in 0..self.ncols.min(8) {
                // Print up to 8 cols
                write!(f, " {:8.4}", self.get(row, col))?;
                if col != self.ncols - 1 {
                    write!(f, ",")?;
                }
            }
            if self.ncols > 8 {
                write!(f, " ...")?;
            }
            write!(f, " ]")?;
        }
        if self.nrows > 6 {
            write!(f, "\n...")?;
        }
        Ok(())
    }
}

// From Vec<FloatArray<f64>> to Matrix (all cols must match length)
impl From<(Vec<FloatArray<f64>>, String)> for Matrix {
    fn from((columns, name): (Vec<FloatArray<f64>>, String)) -> Self {
        let ncols = columns.len();
        let nrows = columns.first().map(|c| c.data.len()).unwrap_or(0);
        for col in &columns {
            assert_eq!(col.data.len(), nrows, "Column length mismatch");
        }
        let mut data = Vec64::with_capacity(nrows * ncols);
        for col in &columns {
            data.extend_from_slice(&col.data);
        }
        Matrix { nrows, ncols, data, name }
    }
}

// From &[FloatArray<f64>] to Matrix
impl From<&[FloatArray<f64>]> for Matrix {
    fn from(columns: &[FloatArray<f64>]) -> Self {
        let ncols = columns.len();
        let nrows = columns.first().map(|c| c.data.len()).unwrap_or(0);
        for col in columns {
            assert_eq!(col.data.len(), nrows, "Column length mismatch");
        }
        let mut data = Vec64::with_capacity(nrows * ncols);
        for col in columns {
            data.extend_from_slice(&col.data);
        }
        let name = {
            let id = UNNAMED_MATRIX_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("UnnamedMatrix{}", id)
        };
        Matrix { nrows, ncols, data, name }
    }
}

// TODO: Fix
// impl TryFrom<&Table> for Matrix {
//     type Error = MinarrowError;

//     fn try_from(table: &Table) -> Result<Self, Self::Error> {
//         let name = table.name.clone();
//         let ncols = table.n_cols();
//         let nrows = table.n_rows();

//         // Collect and check columns
//         let mut float_columns = Vec::with_capacity(ncols);
//         for fa in &table.cols {
//             let numeric_array = fa.array.num();
//             let arr: FloatArray<f64> = numeric_array.f64()?;
//             float_columns.push(arr);
//         }

//         // Ensure all columns are the correct length
//         for (col_idx, col) in float_columns.iter().enumerate() {
//             if col.data.len() != nrows {
//                 return Err(MinarrowError::ColumnLengthMismatch {
//                     col: col_idx,
//                     expected: nrows,
//                     found: col.data.len()
//                 });
//             }
//         }

//         // Flatten into single column-major Vec64<f64>
//         let mut data = Vec64::with_capacity(nrows * ncols);
//         for col in &float_columns {
//             data.0.extend_from_slice(&col.data);
//         }

//         Ok(Matrix { nrows, ncols, data, name })
//     }
// }

// From Vec<Vec<f64>> (Vec-of-cols) to Matrix (anonymous name)
impl From<&[Vec<f64>]> for Matrix {
    fn from(columns: &[Vec<f64>]) -> Self {
        let ncols = columns.len();
        let nrows = columns.first().map(|c| c.len()).unwrap_or(0);
        for col in columns {
            assert_eq!(col.len(), nrows, "Column length mismatch");
        }
        let mut data = Vec64::with_capacity(nrows * ncols);
        for col in columns {
            data.extend_from_slice(col);
        }
        let name = {
            let id = UNNAMED_MATRIX_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("UnnamedMatrix{}", id)
        };
        Matrix { nrows, ncols, data, name }
    }
}

// From flat slice with shape
impl<'a> From<(&'a [f64], usize, usize, Option<String>)> for Matrix {
    fn from((slice, nrows, ncols, name): (&'a [f64], usize, usize, Option<String>)) -> Self {
        assert_eq!(slice.len(), nrows * ncols, "Slice shape mismatch");
        let data = Vec64::from(slice);
        let name = name.unwrap_or_else(|| {
            let id = UNNAMED_MATRIX_COUNTER.fetch_add(1, Ordering::Relaxed);
            format!("UnnamedMatrix{}", id)
        });
        Matrix { nrows, ncols, data, name }
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
