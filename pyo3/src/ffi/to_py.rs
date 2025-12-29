//! # MinArrow to PyArrow Conversion
//!
//! Converts MinArrow arrays to PyArrow arrays using the Arrow C Data Interface.

use minarrow::ffi::arrow_c_ffi::export_to_c;
use minarrow::ffi::schema::Schema;
use minarrow::{Array, Field, SuperArray, SuperTable, Table};
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::Arc;

use crate::error::PyMinarrowError;

/// Converts a MinArrow Array to a PyArrow Array.
///
/// Uses the Arrow C Data Interface for zero-copy conversion where possible.
///
/// # Arguments
/// * `array` - The MinArrow array to convert (wrapped in Arc)
/// * `field` - Field metadata for the array
/// * `py` - Python interpreter handle
///
/// # Returns
/// * `PyResult<Bound<'py, PyAny>>` - The PyArrow Array
pub fn array_to_py<'py>(
    array: Arc<Array>,
    field: &Field,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;

    // Build schema from field
    let schema = Schema::from(vec![field.clone()]);

    // Export to Arrow C format
    let (array_ptr, schema_ptr) = export_to_c(array, schema);

    // Import into PyArrow via _import_from_c
    // PyArrow takes ownership and will call the release callbacks when done.
    // We must NOT call them ourselves - that would cause a double-free.
    pyarrow.getattr("Array")?.call_method1(
        "_import_from_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    ).map_err(|e| {
        PyMinarrowError::PyArrow(format!("Failed to import array into PyArrow: {}", e)).into()
    })
}

/// Converts a MinArrow Table to a PyArrow RecordBatch.
///
/// Converts each column to a PyArrow array and assembles them into a RecordBatch.
///
/// # Arguments
/// * `table` - The MinArrow Table to convert
/// * `py` - Python interpreter handle
///
/// # Returns
/// * `PyResult<Bound<'py, PyAny>>` - The PyArrow RecordBatch
pub fn table_to_py<'py>(table: &Table, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;

    // Convert each column to a PyArrow array
    let mut py_arrays = Vec::with_capacity(table.n_cols());
    let mut names = Vec::with_capacity(table.n_cols());

    for fa in &table.cols {
        let array = Arc::new(fa.array.clone());
        let py_array = array_to_py(array, &fa.field, py)?;
        py_arrays.push(py_array);
        names.push(fa.field.name.clone());
    }

    // Create PyArrow RecordBatch from arrays
    // PyArrow's RecordBatch.from_arrays(arrays, names) expects:
    // - arrays: list of arrays
    // - names: list of column names
    let py_arrays_list = PyList::new(py, py_arrays)?;
    let py_names_list = PyList::new(py, names)?;

    pyarrow
        .getattr("RecordBatch")?
        .call_method1("from_arrays", (py_arrays_list, py_names_list))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to create PyArrow RecordBatch: {}", e)).into()
        })
}

/// Converts a MinArrow SuperTable to a PyArrow Table.
///
/// Converts each batch (Table) to a PyArrow RecordBatch, then assembles them
/// into a PyArrow Table.
///
/// # Arguments
/// * `super_table` - The MinArrow SuperTable to convert
/// * `py` - Python interpreter handle
///
/// # Returns
/// * `PyResult<Bound<'py, PyAny>>` - The PyArrow Table
pub fn super_table_to_py<'py>(super_table: &SuperTable, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;

    if super_table.batches.is_empty() {
        // Return empty table
        return pyarrow
            .getattr("Table")?
            .call_method0("from_batches")
            .map_err(|e| {
                PyMinarrowError::PyArrow(format!("Failed to create empty PyArrow Table: {}", e)).into()
            });
    }

    // Convert each batch to a PyArrow RecordBatch
    let mut py_batches = Vec::with_capacity(super_table.batches.len());
    for batch in &super_table.batches {
        let py_batch = table_to_py(batch, py)?;
        py_batches.push(py_batch);
    }

    let py_batches_list = PyList::new(py, py_batches)?;

    // Create PyArrow Table from RecordBatches
    pyarrow
        .getattr("Table")?
        .call_method1("from_batches", (py_batches_list,))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to create PyArrow Table: {}", e)).into()
        })
}

/// Converts a MinArrow SuperArray to a PyArrow ChunkedArray.
///
/// Converts each chunk (FieldArray) to a PyArrow Array, then assembles them
/// into a PyArrow ChunkedArray.
///
/// # Arguments
/// * `super_array` - The MinArrow SuperArray to convert
/// * `py` - Python interpreter handle
///
/// # Returns
/// * `PyResult<Bound<'py, PyAny>>` - The PyArrow ChunkedArray
pub fn super_array_to_py<'py>(super_array: &SuperArray, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;

    let chunks = super_array.chunks();
    if chunks.is_empty() {
        // Return empty chunked array - need a type, use int32 as default
        let empty_arr = pyarrow.call_method1("array", (Vec::<i32>::new(),))?;
        return pyarrow
            .call_method1("chunked_array", (vec![empty_arr],))
            .map_err(|e| {
                PyMinarrowError::PyArrow(format!("Failed to create empty PyArrow ChunkedArray: {}", e)).into()
            });
    }

    // Convert each chunk to a PyArrow Array
    let mut py_arrays = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let array = Arc::new(chunk.array.clone());
        let py_array = array_to_py(array, &chunk.field, py)?;
        py_arrays.push(py_array);
    }

    let py_arrays_list = PyList::new(py, py_arrays)?;

    // Create PyArrow ChunkedArray from arrays
    pyarrow
        .call_method1("chunked_array", (py_arrays_list,))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to create PyArrow ChunkedArray: {}", e)).into()
        })
}
