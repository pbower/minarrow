//! # PyArrow to MinArrow Conversion
//!
//! Converts PyArrow arrays to MinArrow arrays using the Arrow C Data Interface.

use minarrow::ffi::arrow_c_ffi::{ArrowArray, ArrowSchema, import_from_c_owned};
use minarrow::{Field, FieldArray, SuperArray, SuperTable};
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use std::sync::Arc;

use crate::error::{PyMinarrowError, PyMinarrowResult};

/// Converts a PyArrow Array to a MinArrow FieldArray.
///
/// Uses the Arrow C Data Interface for zero-copy conversion.
/// The PyArrow array must support the `_export_to_c` method.
///
/// # Zero-Copy Behaviour
/// This function does true zero-copy where possible:
/// - The ArrowArray's release callback is stored and called when the MinArrow array is dropped
/// - PyArrow's buffer memory is held alive via SharedBuffer until all references are gone
/// - Null masks and string offsets are copied due to the smaller overhead.
///
/// # Arguments
/// * `obj` - A PyArrow Array object
///
/// # Returns
/// * `PyMinarrowResult<FieldArray>` - The converted MinArrow FieldArray with preserved type metadata
pub fn array_to_rust(obj: &Bound<PyAny>) -> PyMinarrowResult<FieldArray> {
    // Heap-allocate the C structures to receive the exported data
    // Using Box ensures proper memory management across the FFI boundary
    let array = Box::new(ArrowArray::empty());
    let schema = Box::new(ArrowSchema::empty());

    let array_ptr = Box::into_raw(array);
    let schema_ptr = Box::into_raw(schema);

    // Call PyArrow's _export_to_c method to fill in the C structures
    // PyArrow expects two integer arguments representing the memory addresses
    obj.call_method1(
        "_export_to_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )
    .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to export PyArrow array: {}", e)))?;

    // Reclaim ownership of the boxes
    let array_box = unsafe { Box::from_raw(array_ptr) };
    let schema_box = unsafe { Box::from_raw(schema_ptr) };

    // Import with ownership transfer - MinArrow takes ownership of the ArrowArray
    // and will call the release callback when the imported array is dropped.
    // Returns both the Array and its Field metadata (i.e., preserving exact Arrow type).
    // SAFETY: The boxes contain valid data populated by PyArrow's _export_to_c
    let (array, field) = unsafe { import_from_c_owned(array_box, schema_box) };

    Ok(FieldArray::new(field, (*array).clone()))
}

/// Converts a PyArrow RecordBatch to a MinArrow Table.
///
/// Iterates over the columns of the RecordBatch and converts each to a MinArrow array,
/// then assembles them into a MinArrow Table.
///
/// # Arguments
/// * `obj` - A PyArrow RecordBatch object
///
/// # Returns
/// * `PyMinarrowResult<minarrow::Table>` - The converted MinArrow Table
pub fn record_batch_to_rust(obj: &Bound<PyAny>) -> PyMinarrowResult<minarrow::Table> {
    let _py = obj.py();

    // Get number of columns
    let num_columns: usize = obj
        .getattr("num_columns")
        .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to get num_columns: {}", e)))?
        .extract()
        .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to extract num_columns: {}", e)))?;

    // Get schema for field names
    let schema = obj
        .getattr("schema")
        .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to get schema: {}", e)))?;

    let mut cols = Vec::with_capacity(num_columns);

    for i in 0..num_columns {
        // Get the column array
        let column = obj
            .call_method1("column", (i,))
            .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to get column {}: {}", i, e)))?;

        // Get the field metadata
        let field = schema
            .call_method1("field", (i,))
            .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to get field {}: {}", i, e)))?;

        let name: String = field
            .getattr("name")
            .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to get field name: {}", e)))?
            .extract()
            .map_err(|e| {
                PyMinarrowError::PyArrow(format!("Failed to extract field name: {}", e))
            })?;

        // Convert the array - returns FieldArray with preserved type metadata
        let mut field_array = array_to_rust(&column)?;

        // Use the column name from the schema - overrides any name from PyArrow
        field_array.field = Arc::new(Field::new(
            name,
            field_array.field.dtype.clone(),
            field_array.field.nullable,
            None,
        ));
        cols.push(field_array);
    }

    // Create the Table
    let table = minarrow::Table::new(String::new(), Some(cols));
    Ok(table)
}

/// Converts a PyArrow Table to a MinArrow SuperTable.
///
/// A PyArrow Table is composed of ChunkedArrays (one per column), where each
/// ChunkedArray contains multiple array chunks. We convert this to a SuperTable
/// by converting each chunk group into a RecordBatch/Table.
///
/// # Arguments
/// * `obj` - A PyArrow Table object
///
/// # Returns
/// * `PyMinarrowResult<SuperTable>` - The converted MinArrow SuperTable
pub fn table_to_rust(obj: &Bound<PyAny>) -> PyMinarrowResult<SuperTable> {
    // Get the list of RecordBatches from the PyArrow Table
    let batches = obj
        .call_method0("to_batches")
        .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to get batches from Table: {}", e)))?;

    let batches_list: Vec<Bound<PyAny>> = batches
        .extract()
        .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to extract batches list: {}", e)))?;

    if batches_list.is_empty() {
        return Ok(SuperTable::new(String::new()));
    }

    let mut tables = Vec::with_capacity(batches_list.len());
    for batch in batches_list {
        let table = record_batch_to_rust(&batch)?;
        tables.push(Arc::new(table));
    }

    Ok(SuperTable::from_batches(tables, None))
}

/// Converts a PyArrow ChunkedArray to a MinArrow SuperArray.
///
/// A PyArrow ChunkedArray contains multiple array chunks. We convert each chunk
/// to a MinArrow Array and wrap them in a SuperArray.
///
/// # Arguments
/// * `obj` - A PyArrow ChunkedArray object
///
/// # Returns
/// * `PyMinarrowResult<SuperArray>` - The converted MinArrow SuperArray
pub fn chunked_array_to_rust(obj: &Bound<PyAny>) -> PyMinarrowResult<SuperArray> {
    // Get the chunks from the ChunkedArray
    let chunks = obj
        .getattr("chunks")
        .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to get chunks: {}", e)))?;

    let chunks_list: Vec<Bound<PyAny>> = chunks
        .extract()
        .map_err(|e| PyMinarrowError::PyArrow(format!("Failed to extract chunks list: {}", e)))?;

    if chunks_list.is_empty() {
        return Ok(SuperArray::new());
    }

    // Convert the first chunk to get the canonical field metadata
    let first_fa = array_to_rust(&chunks_list[0])?;
    let field = first_fa.field.clone();

    let mut field_arrays = Vec::with_capacity(chunks_list.len());
    field_arrays.push(first_fa);

    // Convert remaining chunks using the same field metadata for consistency
    for chunk in chunks_list.iter().skip(1) {
        let chunk_fa = array_to_rust(chunk)?;
        // Use the first chunk's field metadata to ensure consistency
        let field_array = FieldArray::new((*field).clone(), chunk_fa.array);
        field_arrays.push(field_array);
    }

    Ok(SuperArray::from_field_array_chunks(field_arrays))
}
