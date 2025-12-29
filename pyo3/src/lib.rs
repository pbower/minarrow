//! # minarrow-pyo3 - PyO3 Bindings for MinArrow
//!
//! Zero-copy Python bindings for MinArrow via the Arrow C Data Interface.
//!
//! This crate provides transparent wrapper types that enable seamless conversion
//! between MinArrow's Rust types and PyArrow's Python types.
//!
//! ## Features
//!
//! - **Zero-copy conversion** via Arrow C Data Interface where possible
//! - **Transparent wrappers** (`PyArray`, `PyRecordBatch`) implementing PyO3 traits
//! - **Idiomatic Rust API** for building Python extensions
//!
//! ## Type Mappings
//! 
//! Minarrow calls an object with a header, rows and columns a 'Table' favouring broader matter-of-factness.
//! Apache Arrow calls it a 'RecordBatch' in line with the Apache Arrow standard, whereby a 'Table' (at least in PyArrow),
//! is considered a chunked composition of those RecordBatches, for a more highly engineered approach.
//! Below is how they map to one another for the equivalent memory and object layout.
//! 
//! | MinArrow | PyArrow | Wrapper Type |
//! |----------|---------|--------------|
//! | `Array` | `pa.Array` | `PyArray` |
//! | `Table` | `pa.RecordBatch` | `PyRecordBatch` |
//! | `SuperTable` | `pa.Table` | `PyTable` |
//! | `SuperArray` | `pa.ChunkedArray` | `PyChunkedArray` |
//!
//! ## Example
//!
//! ```ignore
//! use minarrow_pyo3::{PyArray, PyRecordBatch};
//! use pyo3::prelude::*;
//!
//! #[pyfunction]
//! fn process_batch(input: PyRecordBatch) -> PyResult<PyRecordBatch> {
//!     let table: minarrow::Table = input.into();
//!     // Process the table using MinArrow...
//!     Ok(PyRecordBatch::from(table))
//! }
//!
//! #[pymodule]
//! fn my_extension(m: &Bound<'_, PyModule>) -> PyResult<()> {
//!     m.add_function(wrap_pyfunction!(process_batch, m)?)?;
//!     Ok(())
//! }
//! ```
//!
//! In Python:
//! ```python
//! import pyarrow as pa
//! import my_extension
//!
//! batch = pa.RecordBatch.from_pydict({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
//! result = my_extension.process_batch(batch)
//! ```

#![feature(allocator_api)]
#![feature(slice_ptr_get)]
#![feature(portable_simd)]

use once_cell::sync::Lazy;
use pyo3::prelude::*;

pub mod error;
pub mod ffi;
pub mod types;

#[cfg(test)]
mod tests;

// Re-export the main types for ease of use
pub use error::{PyMinarrowError, PyMinarrowResult};
pub use types::{PyArray, PyChunkedArray, PyField, PyRecordBatch, PyTable};

// Re-export minarrow types that users might need
pub use minarrow::{Array, Field, FieldArray, MaskedArray, NumericArray, SuperArray, SuperTable, Table, TextArray};

/// Lazily-initialised reference to the pyarrow module.
/// Used internally for efficient module lookup.
#[allow(dead_code)]
pub(crate) static PYARROW: Lazy<Py<PyModule>> = Lazy::new(|| {
    Python::with_gil(|py| {
        PyModule::import(py, "pyarrow")
            .expect("pyarrow must be installed")
            .unbind()
    })
});

/// Echo back a PyArrow array after roundtrip through MinArrow.
/// Used to test that conversion works correctly.
#[pyfunction]
fn echo_array(arr: PyArray) -> PyResult<PyArray> {
    // The array is converted to MinArrow on input and back to PyArrow on output
    Ok(arr)
}

/// Echo back a PyArrow RecordBatch after roundtrip through MinArrow.
/// Used to test that conversion works correctly.
#[pyfunction]
fn echo_batch(batch: PyRecordBatch) -> PyResult<PyRecordBatch> {
    // The batch is converted to MinArrow Table on input and back to PyArrow on output
    Ok(batch)
}

/// Get information about a PyArrow array after converting to MinArrow.
#[pyfunction]
fn array_info(arr: PyArray) -> PyResult<String> {
    let inner = arr.inner();
    Ok(format!(
        "MinArrow Array: len={}, null_count={}",
        inner.len(),
        inner.null_count()
    ))
}

/// Get information about a PyArrow RecordBatch after converting to MinArrow.
#[pyfunction]
fn batch_info(batch: PyRecordBatch) -> PyResult<String> {
    let inner = batch.inner();
    Ok(format!(
        "MinArrow Table: rows={}, cols={}",
        inner.n_rows(),
        inner.n_cols()
    ))
}

/// Echo back a PyArrow Table after roundtrip through MinArrow.
/// Used to test that conversion works correctly.
#[pyfunction]
fn echo_table(table: PyTable) -> PyResult<PyTable> {
    // The table is converted to MinArrow SuperTable on input and back to PyArrow on output
    Ok(table)
}

/// Echo back a PyArrow ChunkedArray after roundtrip through MinArrow.
/// Used to test that conversion works correctly.
#[pyfunction]
fn echo_chunked(arr: PyChunkedArray) -> PyResult<PyChunkedArray> {
    // The array is converted to MinArrow SuperArray on input and back to PyArrow on output
    Ok(arr)
}

/// Get information about a PyArrow Table after converting to MinArrow.
#[pyfunction]
fn table_info(table: PyTable) -> PyResult<String> {
    let inner = table.inner();
    Ok(format!(
        "MinArrow SuperTable: batches={}, rows={}, cols={}",
        inner.batches.len(),
        inner.n_rows,
        inner.schema.len()
    ))
}

/// Get information about a PyArrow ChunkedArray after converting to MinArrow.
#[pyfunction]
fn chunked_info(arr: PyChunkedArray) -> PyResult<String> {
    let inner = arr.inner();
    Ok(format!(
        "MinArrow SuperArray: chunks={}, len={}",
        inner.n_chunks(),
        inner.len()
    ))
}

/// Python module definition for minarrow_pyo3.
///
/// This module primarily provides type conversion capabilities via the
/// `PyArray` and `PyRecordBatch` wrapper types. The actual conversions
/// happen automatically when these types are used as function parameters
/// or return values in PyO3 functions.
#[pymodule]
fn minarrow_pyo3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Module-level docstring
    m.add("__doc__", "PyO3 bindings for MinArrow - zero-copy Arrow interop with Python")?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Add test functions
    m.add_function(wrap_pyfunction!(echo_array, m)?)?;
    m.add_function(wrap_pyfunction!(echo_batch, m)?)?;
    m.add_function(wrap_pyfunction!(echo_table, m)?)?;
    m.add_function(wrap_pyfunction!(echo_chunked, m)?)?;
    m.add_function(wrap_pyfunction!(array_info, m)?)?;
    m.add_function(wrap_pyfunction!(batch_info, m)?)?;
    m.add_function(wrap_pyfunction!(table_info, m)?)?;
    m.add_function(wrap_pyfunction!(chunked_info, m)?)?;

    Ok(())
}
