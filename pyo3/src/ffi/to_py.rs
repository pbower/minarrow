//! # MinArrow to Python Conversion
//!
//! Converts MinArrow arrays to PyArrow arrays and PyCapsules using the
//! Arrow C Data Interface and Arrow PyCapsule Interface.
//!
//! ## Export Strategy
//! Two export paths are provided:
//! 1. **PyArrow objects** - convert to PyArrow arrays/tables via `_import_from_c`
//! 2. **PyCapsules** - export as Arrow PyCapsule objects for direct consumption
//!    by any Python library supporting the Arrow PyCapsule Interface

use minarrow::ffi::arrow_c_ffi::{
    export_array_stream, export_record_batch_stream_with_metadata, export_to_c, ArrowArray,
    ArrowArrayStream, ArrowSchema,
};
use minarrow::ffi::schema::Schema;
use minarrow::{Array, Field, SuperArray, SuperTable, Table};
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyList};
use std::sync::Arc;

use crate::error::PyMinarrowError;

/// Key used to store the MinArrow table name in Arrow schema metadata.
pub(crate) const TABLE_NAME_KEY: &str = "minarrow:table_name";

/// Builds an Arrow metadata map containing the table name, if non-empty.
fn table_name_metadata(name: &str) -> Option<std::collections::BTreeMap<String, String>> {
    if name.is_empty() {
        None
    } else {
        let mut m = std::collections::BTreeMap::new();
        m.insert(TABLE_NAME_KEY.to_string(), name.to_string());
        Some(m)
    }
}

// PyArrow conversion - legacy C data interface

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
    pyarrow
        .getattr("Array")?
        .call_method1(
            "_import_from_c",
            (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
        )
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to import array into PyArrow: {}", e)).into()
        })
}

/// Converts a MinArrow Table to a PyArrow RecordBatch.
///
/// Converts each column to a PyArrow array and assembles them into a RecordBatch.
/// If the table has a non-empty name, it is stored in the PyArrow schema metadata
/// under the `minarrow:table_name` key so it can be recovered on import.
pub fn table_to_py<'py>(table: &Table, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;

    let mut py_fields = Vec::with_capacity(table.n_cols());
    let mut py_arrays = Vec::with_capacity(table.n_cols());

    for fa in &table.cols {
        let array = Arc::new(fa.array.clone());
        let py_array = array_to_py(array, &fa.field, py)?;
        // Extract the PyArrow field from the array's type to preserve type metadata
        let py_field = pyarrow.call_method1(
            "field",
            (fa.field.name.clone(), py_array.getattr("type")?),
        )?;
        py_fields.push(py_field);
        py_arrays.push(py_array);
    }

    let py_fields_list = PyList::new(py, &py_fields)?;

    // Build schema, attaching table name as metadata if present
    let mut schema = pyarrow.call_method1("schema", (py_fields_list,))?;
    if !table.name.is_empty() {
        let metadata = [(TABLE_NAME_KEY, &table.name)].into_py_dict(py)?;
        schema = schema.call_method1("with_metadata", (metadata,))?;
    }

    let py_arrays_list = PyList::new(py, py_arrays)?;

    let kwargs = [("schema", schema)].into_py_dict(py)?;
    pyarrow
        .getattr("RecordBatch")?
        .call_method("from_arrays", (py_arrays_list,), Some(&kwargs))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to create PyArrow RecordBatch: {}", e)).into()
        })
}

/// Converts a MinArrow SuperTable to a PyArrow Table.
pub fn super_table_to_py<'py>(
    super_table: &SuperTable,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;

    if super_table.batches.is_empty() {
        return pyarrow
            .getattr("Table")?
            .call_method0("from_batches")
            .map_err(|e| {
                PyMinarrowError::PyArrow(format!(
                    "Failed to create empty PyArrow Table: {}",
                    e
                ))
                .into()
            });
    }

    let mut py_batches = Vec::with_capacity(super_table.batches.len());
    for batch in &super_table.batches {
        let py_batch = table_to_py(batch, py)?;
        py_batches.push(py_batch);
    }

    let py_batches_list = PyList::new(py, py_batches)?;

    let py_table = pyarrow
        .getattr("Table")?
        .call_method1("from_batches", (py_batches_list,))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to create PyArrow Table: {}", e))
        })?;

    // Attach table name as schema metadata if present
    if !super_table.name.is_empty() {
        let metadata = [(TABLE_NAME_KEY, &super_table.name)]
            .into_py_dict(py)?;
        return py_table
            .call_method1("replace_schema_metadata", (metadata,))
            .map_err(|e| {
                PyMinarrowError::PyArrow(format!("Failed to set schema metadata: {}", e)).into()
            });
    }

    Ok(py_table)
}

/// Converts a MinArrow SuperArray to a PyArrow ChunkedArray.
pub fn super_array_to_py<'py>(
    super_array: &SuperArray,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let pyarrow = py.import("pyarrow")?;

    let chunks = super_array.chunks();
    if chunks.is_empty() {
        let empty_arr = pyarrow.call_method1("array", (Vec::<i32>::new(),))?;
        return pyarrow
            .call_method1("chunked_array", (vec![empty_arr],))
            .map_err(|e| {
                PyMinarrowError::PyArrow(format!(
                    "Failed to create empty PyArrow ChunkedArray: {}",
                    e
                ))
                .into()
            });
    }

    let field = super_array.field_ref();
    let mut py_arrays = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let array = Arc::new(chunk.clone());
        let py_array = array_to_py(array, field, py)?;
        py_arrays.push(py_array);
    }

    let py_arrays_list = PyList::new(py, py_arrays)?;

    pyarrow
        .call_method1("chunked_array", (py_arrays_list,))
        .map_err(|e| {
            PyMinarrowError::PyArrow(format!("Failed to create PyArrow ChunkedArray: {}", e))
                .into()
        })
}

// PyCapsule export

/// Capsule destructor for ArrowSchema.
/// Called when the PyCapsule is garbage collected without being consumed.
unsafe extern "C" fn arrow_schema_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        let name = c"arrow_schema";
        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr()) as *mut ArrowSchema;
        if !ptr.is_null() {
            let schema = &mut *ptr;
            if let Some(release) = schema.release {
                release(schema);
            }
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Capsule destructor for ArrowArray.
unsafe extern "C" fn arrow_array_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        let name = c"arrow_array";
        let ptr = pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr()) as *mut ArrowArray;
        if !ptr.is_null() {
            let array = &mut *ptr;
            if let Some(release) = array.release {
                release(array);
            }
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Capsule destructor for ArrowArrayStream.
unsafe extern "C" fn arrow_stream_capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    unsafe {
        let name = c"arrow_array_stream";
        let ptr =
            pyo3::ffi::PyCapsule_GetPointer(capsule, name.as_ptr()) as *mut ArrowArrayStream;
        if !ptr.is_null() {
            let stream = &mut *ptr;
            if let Some(release) = stream.release {
                release(stream);
            }
            let _ = Box::from_raw(ptr);
        }
    }
}

/// Exports a MinArrow array as a pair of PyCapsules (schema, array).
///
/// Returns `(schema_capsule, array_capsule)` following the Arrow PyCapsule Interface.
/// The capsules have destructors that call the Arrow release callbacks if the
/// capsules are not consumed by a recipient.
pub fn array_to_capsules<'py>(
    array: Arc<Array>,
    field: &Field,
    py: Python<'py>,
) -> PyResult<(PyObject, PyObject)> {
    let schema = Schema::from(vec![field.clone()]);
    let (arr_ptr, sch_ptr) = export_to_c(array, schema);

    // Create schema capsule
    let schema_name = c"arrow_schema";
    let schema_capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            sch_ptr as *mut std::ffi::c_void,
            schema_name.as_ptr(),
            Some(arrow_schema_capsule_destructor),
        );
        if cap.is_null() {
            // Clean up on failure
            let s = &mut *sch_ptr;
            if let Some(release) = s.release {
                release(sch_ptr);
            }
            let _ = Box::from_raw(sch_ptr);
            let a = &mut *arr_ptr;
            if let Some(release) = a.release {
                release(arr_ptr);
            }
            let _ = Box::from_raw(arr_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create schema PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    // Create array capsule
    let array_name = c"arrow_array";
    let array_capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            arr_ptr as *mut std::ffi::c_void,
            array_name.as_ptr(),
            Some(arrow_array_capsule_destructor),
        );
        if cap.is_null() {
            let a = &mut *arr_ptr;
            if let Some(release) = a.release {
                release(arr_ptr);
            }
            let _ = Box::from_raw(arr_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create array PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    Ok((schema_capsule.unbind(), array_capsule.unbind()))
}

/// Exports a MinArrow Table as an ArrowArrayStream PyCapsule.
///
/// The stream yields one struct array (record batch) corresponding to the table.
pub fn table_to_stream_capsule<'py>(table: &Table, py: Python<'py>) -> PyResult<PyObject> {
    let fields: Vec<Field> = table.cols.iter().map(|fa| (*fa.field).clone()).collect();
    let columns: Vec<(Arc<Array>, Schema)> = table
        .cols
        .iter()
        .map(|fa| {
            (
                Arc::new(fa.array.clone()),
                Schema::from(vec![(*fa.field).clone()]),
            )
        })
        .collect();

    let metadata = table_name_metadata(&table.name);
    let stream = export_record_batch_stream_with_metadata(vec![columns], fields, metadata);
    let stream_ptr = Box::into_raw(stream);

    let name = c"arrow_array_stream";
    let capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            stream_ptr as *mut std::ffi::c_void,
            name.as_ptr(),
            Some(arrow_stream_capsule_destructor),
        );
        if cap.is_null() {
            // Clean up
            let s = &mut *stream_ptr;
            if let Some(release) = s.release {
                release(stream_ptr);
            }
            let _ = Box::from_raw(stream_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create stream PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    Ok(capsule.unbind())
}

/// Exports a MinArrow SuperTable as an ArrowArrayStream PyCapsule.
///
/// The stream yields one struct array per batch in the SuperTable.
pub fn super_table_to_stream_capsule<'py>(
    super_table: &SuperTable,
    py: Python<'py>,
) -> PyResult<PyObject> {
    if super_table.batches.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot export empty SuperTable as stream capsule",
        ));
    }

    // Extract fields from the first batch
    let fields: Vec<Field> = super_table.batches[0]
        .cols
        .iter()
        .map(|fa| (*fa.field).clone())
        .collect();

    // Convert each batch to column (Arc<Array>, Schema) pairs
    let batches: Vec<Vec<(Arc<Array>, Schema)>> = super_table
        .batches
        .iter()
        .map(|table| {
            table
                .cols
                .iter()
                .map(|fa| {
                    (
                        Arc::new(fa.array.clone()),
                        Schema::from(vec![(*fa.field).clone()]),
                    )
                })
                .collect()
        })
        .collect();

    let metadata = table_name_metadata(&super_table.name);
    let stream = export_record_batch_stream_with_metadata(batches, fields, metadata);
    let stream_ptr = Box::into_raw(stream);

    let name = c"arrow_array_stream";
    let capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            stream_ptr as *mut std::ffi::c_void,
            name.as_ptr(),
            Some(arrow_stream_capsule_destructor),
        );
        if cap.is_null() {
            let s = &mut *stream_ptr;
            if let Some(release) = s.release {
                release(stream_ptr);
            }
            let _ = Box::from_raw(stream_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create stream PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    Ok(capsule.unbind())
}

/// Exports a MinArrow SuperArray as an ArrowArrayStream PyCapsule.
///
/// The stream yields one plain array per chunk.
pub fn super_array_to_stream_capsule<'py>(
    super_array: &SuperArray,
    py: Python<'py>,
) -> PyResult<PyObject> {
    let chunks = super_array.chunks();
    if chunks.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot export empty SuperArray as stream capsule",
        ));
    }

    let field = super_array.field_ref().clone();
    let array_chunks: Vec<Arc<Array>> = chunks.iter().map(|c| Arc::new(c.clone())).collect();

    let stream = export_array_stream(array_chunks, field);
    let stream_ptr = Box::into_raw(stream);

    let name = c"arrow_array_stream";
    let capsule = unsafe {
        let cap = pyo3::ffi::PyCapsule_New(
            stream_ptr as *mut std::ffi::c_void,
            name.as_ptr(),
            Some(arrow_stream_capsule_destructor),
        );
        if cap.is_null() {
            let s = &mut *stream_ptr;
            if let Some(release) = s.release {
                release(stream_ptr);
            }
            let _ = Box::from_raw(stream_ptr);
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Failed to create stream PyCapsule",
            ));
        }
        Bound::from_owned_ptr(py, cap)
    };

    Ok(capsule.unbind())
}
