//! # Comprehensive Roundtrip Test Runner
//!
//! This example runs all the roundtrip tests for MinArrow ↔ PyArrow conversions.
//! It tests arrays, tables, nullability, and edge cases like empty arrays.
//!
//! ## Prerequisites
//!
//! 1. Python with PyArrow installed:
//!    ```bash
//!    pip install pyarrow
//!    # or use the project's venv:
//!    cd pyo3 && source .venv/bin/activate
//!    ```
//!
//! ## Running the Tests
//!
//! From the `pyo3/` directory:
//!
//! ```bash
//! # Set PYTHONHOME to where libpython is installed
//! # Find it via: ldd target/debug/examples/run_tests | grep python
//! # Then set prefix accordingly (e.g., /usr/local or /usr)
//!
//! PYTHONHOME=/usr/local cargo run --example run_tests \
//!     --no-default-features \
//!     --features "datetime,extended_numeric_types,extended_categorical"
//! ```
//!
//! ### Why --no-default-features?
//!
//! The default features include `extension-module` which tells PyO3 not to link
//! against libpython (since Python loads the extension at runtime). For standalone
//! binaries like this test runner, we need to link against libpython, so we disable
//! that feature.
//!
//! ### Why PYTHONHOME?
//!
//! When embedding Python in a Rust binary, Python needs to know where its standard
//! library is located. Set PYTHONHOME to the Python prefix (e.g., `/usr/local` or
//! `/usr`). You can find this via: `python3 -c "import sys; print(sys.prefix)"`

use minarrow::ffi::arrow_dtype::ArrowType;
use minarrow::{
    Array, BooleanArray, Field, FieldArray, FloatArray, IntegerArray, MaskedArray, NumericArray,
    StringArray, Table, TextArray,
};
use minarrow_pyo3::{PyArray, PyRecordBatch};
use pyo3::prelude::*;

fn main() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        println!("=== MinArrow ↔ PyArrow Comprehensive Tests ===\n");

        let mut passed = 0;
        let mut failed = 0;

        // Integer tests
        run_test("i32 roundtrip", || test_int32_roundtrip(py), &mut passed, &mut failed);
        run_test("i64 roundtrip", || test_int64_roundtrip(py), &mut passed, &mut failed);
        run_test("u32 roundtrip", || test_uint32_roundtrip(py), &mut passed, &mut failed);
        run_test("u64 roundtrip", || test_uint64_roundtrip(py), &mut passed, &mut failed);

        // Float tests
        run_test("f32 roundtrip", || test_float32_roundtrip(py), &mut passed, &mut failed);
        run_test("f64 roundtrip", || test_float64_roundtrip(py), &mut passed, &mut failed);

        // Boolean test
        run_test("boolean roundtrip", || test_boolean_roundtrip(py), &mut passed, &mut failed);

        // String tests
        run_test("string32 roundtrip", || test_string32_roundtrip(py), &mut passed, &mut failed);

        // Nullable tests
        run_test("nullable i32 roundtrip", || test_nullable_int32_roundtrip(py), &mut passed, &mut failed);
        run_test("nullable f64 roundtrip", || test_nullable_float64_roundtrip(py), &mut passed, &mut failed);
        run_test("nullable string roundtrip", || test_nullable_string_roundtrip(py), &mut passed, &mut failed);
        run_test("nullable boolean roundtrip", || test_nullable_boolean_roundtrip(py), &mut passed, &mut failed);

        // Table tests
        run_test("table roundtrip", || test_table_roundtrip(py), &mut passed, &mut failed);
        run_test("table with nulls roundtrip", || test_table_with_nulls_roundtrip(py), &mut passed, &mut failed);

        // Edge case tests
        run_test("empty array roundtrip", || test_empty_array_roundtrip(py), &mut passed, &mut failed);
        run_test("single element roundtrip", || test_single_element_array_roundtrip(py), &mut passed, &mut failed);
        run_test("all nulls roundtrip", || test_all_nulls_array_roundtrip(py), &mut passed, &mut failed);
        run_test("empty strings roundtrip", || test_empty_string_array_roundtrip(py), &mut passed, &mut failed);
        run_test("large array roundtrip", || test_large_array_roundtrip(py), &mut passed, &mut failed);

        println!("\n=== Test Results ===");
        println!("Passed: {}", passed);
        println!("Failed: {}", failed);

        if failed > 0 {
            std::process::exit(1);
        }

        Ok(())
    })
}

fn run_test<F>(name: &str, test_fn: F, passed: &mut usize, failed: &mut usize)
where
    F: FnOnce() -> Result<(), String>,
{
    match test_fn() {
        Ok(()) => {
            println!("✓ {}", name);
            *passed += 1;
        }
        Err(e) => {
            println!("✗ {} - {}", name, e);
            *failed += 1;
        }
    }
}

/// Helper to convert MinArrow Array to PyArrow and back
fn roundtrip_array(py: Python<'_>, array: Array) -> Result<Array, String> {
    let py_array = PyArray::from(array);
    let py_obj = py_array
        .into_pyobject(py)
        .map_err(|e| format!("Failed to convert to PyArrow: {}", e))?;
    let back: PyArray = py_obj
        .extract()
        .map_err(|e| format!("Failed to convert from PyArrow: {}", e))?;
    Ok(back.into_inner().array)
}

// 
// Integer Array Tests
// 

fn test_int32_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<i32>::default();
    arr.push(1);
    arr.push(2);
    arr.push(3);
    arr.push(i32::MAX);
    arr.push(i32::MIN);

    let original = Array::from_int32(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::Int32(a)),
            Array::NumericArray(NumericArray::Int32(b)),
        ) => {
            for i in 0..a.len() {
                if a.get(i) != b.get(i) {
                    return Err(format!("Mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

fn test_int64_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<i64>::default();
    arr.push(100);
    arr.push(200);
    arr.push(300);
    arr.push(i64::MAX);
    arr.push(i64::MIN);

    let original = Array::from_int64(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::Int64(a)),
            Array::NumericArray(NumericArray::Int64(b)),
        ) => {
            for i in 0..a.len() {
                if a.get(i) != b.get(i) {
                    return Err(format!("Mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

fn test_uint32_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<u32>::default();
    arr.push(0);
    arr.push(1);
    arr.push(u32::MAX);

    let original = Array::from_uint32(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::UInt32(a)),
            Array::NumericArray(NumericArray::UInt32(b)),
        ) => {
            for i in 0..a.len() {
                if a.get(i) != b.get(i) {
                    return Err(format!("Mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

fn test_uint64_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<u64>::default();
    arr.push(0);
    arr.push(1);
    arr.push(u64::MAX);

    let original = Array::from_uint64(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::UInt64(a)),
            Array::NumericArray(NumericArray::UInt64(b)),
        ) => {
            for i in 0..a.len() {
                if a.get(i) != b.get(i) {
                    return Err(format!("Mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

// 
// Float Array Tests
// 

fn test_float32_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = FloatArray::<f32>::default();
    arr.push(1.5);
    arr.push(2.25);
    arr.push(f32::MAX);
    arr.push(f32::MIN);
    arr.push(0.0);
    arr.push(-0.0);

    let original = Array::from_float32(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::Float32(a)),
            Array::NumericArray(NumericArray::Float32(b)),
        ) => {
            for i in 0..a.len() {
                let av = a.get(i).unwrap();
                let bv = b.get(i).unwrap();
                if !((av - bv).abs() < f32::EPSILON || (av.is_nan() && bv.is_nan())) {
                    return Err(format!("Mismatch at index {}: {} vs {}", i, av, bv));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

fn test_float64_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = FloatArray::<f64>::default();
    arr.push(1.5);
    arr.push(2.25);
    arr.push(std::f64::consts::PI);
    arr.push(std::f64::consts::E);
    arr.push(f64::MAX);
    arr.push(f64::MIN);

    let original = Array::from_float64(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::Float64(a)),
            Array::NumericArray(NumericArray::Float64(b)),
        ) => {
            for i in 0..a.len() {
                let av = a.get(i).unwrap();
                let bv = b.get(i).unwrap();
                if !((av - bv).abs() < f64::EPSILON || (av.is_nan() && bv.is_nan())) {
                    return Err(format!("Mismatch at index {}: {} vs {}", i, av, bv));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

// 
// Boolean Array Tests
// 

fn test_boolean_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = BooleanArray::default();
    arr.push(true);
    arr.push(false);
    arr.push(true);
    arr.push(false);
    arr.push(true);

    let original = Array::from_bool(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (Array::BooleanArray(a), Array::BooleanArray(b)) => {
            for i in 0..a.len() {
                if a.get(i) != b.get(i) {
                    return Err(format!("Mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

// 
// String Array Tests
// 

fn test_string32_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = StringArray::<u32>::default();
    arr.push_str("hello");
    arr.push_str("world");
    arr.push_str("");
    arr.push_str("with spaces");
    arr.push_str("special: αβγ 日本語 🎉");

    let original = Array::from_string32(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::TextArray(TextArray::String32(a)),
            Array::TextArray(TextArray::String32(b)),
        ) => {
            for i in 0..a.len() {
                if a.get(i) != b.get(i) {
                    return Err(format!("Mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

// 
// Nullable Array Tests
// 

fn test_nullable_int32_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<i32>::default();
    arr.push(1);
    arr.push_null();
    arr.push(3);
    arr.push_null();
    arr.push(5);

    let original = Array::from_int32(arr);
    if original.null_count() != 2 {
        return Err(format!("Expected 2 nulls, got {}", original.null_count()));
    }

    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }
    if original.null_count() != result.null_count() {
        return Err(format!(
            "Null count mismatch: {} vs {}",
            original.null_count(),
            result.null_count()
        ));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::Int32(a)),
            Array::NumericArray(NumericArray::Int32(b)),
        ) => {
            for i in 0..a.len() {
                if a.is_null(i) != b.is_null(i) {
                    return Err(format!("Null mismatch at index {}", i));
                }
                if !a.is_null(i) && a.get(i) != b.get(i) {
                    return Err(format!("Value mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

fn test_nullable_float64_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = FloatArray::<f64>::default();
    arr.push(1.1);
    arr.push_null();
    arr.push(3.3);
    arr.push_null();

    let original = Array::from_float64(arr);
    if original.null_count() != 2 {
        return Err(format!("Expected 2 nulls, got {}", original.null_count()));
    }

    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }
    if original.null_count() != result.null_count() {
        return Err(format!(
            "Null count mismatch: {} vs {}",
            original.null_count(),
            result.null_count()
        ));
    }
    Ok(())
}

fn test_nullable_string_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = StringArray::<u32>::default();
    arr.push_str("hello");
    arr.push_null();
    arr.push_str("world");
    arr.push_null();

    let original = Array::from_string32(arr);
    if original.null_count() != 2 {
        return Err(format!("Expected 2 nulls, got {}", original.null_count()));
    }

    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }
    if original.null_count() != result.null_count() {
        return Err(format!(
            "Null count mismatch: {} vs {}",
            original.null_count(),
            result.null_count()
        ));
    }

    match (&original, &result) {
        (
            Array::TextArray(TextArray::String32(a)),
            Array::TextArray(TextArray::String32(b)),
        ) => {
            for i in 0..a.len() {
                if a.is_null(i) != b.is_null(i) {
                    return Err(format!("Null mismatch at index {}", i));
                }
                if !a.is_null(i) && a.get(i) != b.get(i) {
                    return Err(format!("Value mismatch at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch after roundtrip".to_string()),
    }
    Ok(())
}

fn test_nullable_boolean_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = BooleanArray::default();
    arr.push(true);
    arr.push_null();
    arr.push(false);
    arr.push_null();

    let original = Array::from_bool(arr);
    if original.null_count() != 2 {
        return Err(format!("Expected 2 nulls, got {}", original.null_count()));
    }

    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }
    if original.null_count() != result.null_count() {
        return Err(format!(
            "Null count mismatch: {} vs {}",
            original.null_count(),
            result.null_count()
        ));
    }
    Ok(())
}

// 
// Table / RecordBatch Tests
// 

fn test_table_roundtrip(py: Python<'_>) -> Result<(), String> {
    // Create arrays
    let mut int_arr = IntegerArray::<i64>::default();
    int_arr.push(100);
    int_arr.push(200);
    int_arr.push(300);

    let mut float_arr = FloatArray::<f64>::default();
    float_arr.push(1.1);
    float_arr.push(2.2);
    float_arr.push(3.3);

    let mut str_arr = StringArray::<u32>::default();
    str_arr.push_str("alpha");
    str_arr.push_str("beta");
    str_arr.push_str("gamma");

    let mut bool_arr = BooleanArray::default();
    bool_arr.push(true);
    bool_arr.push(false);
    bool_arr.push(true);

    // Create FieldArrays
    let fa1 = FieldArray::new(
        Field::new("id", ArrowType::Int64, false, None),
        Array::from_int64(int_arr),
    );
    let fa2 = FieldArray::new(
        Field::new("value", ArrowType::Float64, false, None),
        Array::from_float64(float_arr),
    );
    let fa3 = FieldArray::new(
        Field::new("name", ArrowType::String, false, None),
        Array::from_string32(str_arr),
    );
    let fa4 = FieldArray::new(
        Field::new("active", ArrowType::Boolean, false, None),
        Array::from_bool(bool_arr),
    );

    // Create Table
    let original = Table::new("test".to_string(), Some(vec![fa1, fa2, fa3, fa4]));

    // Roundtrip
    let py_batch = PyRecordBatch::from(original.clone());
    let py_obj = py_batch
        .into_pyobject(py)
        .map_err(|e| format!("Failed to convert to PyArrow: {}", e))?;
    let back: PyRecordBatch = py_obj
        .extract()
        .map_err(|e| format!("Failed to convert from PyArrow: {}", e))?;
    let result = back.into_inner();

    // Verify
    if original.n_rows() != result.n_rows() {
        return Err(format!(
            "Row count mismatch: {} vs {}",
            original.n_rows(),
            result.n_rows()
        ));
    }
    if original.n_cols() != result.n_cols() {
        return Err(format!(
            "Column count mismatch: {} vs {}",
            original.n_cols(),
            result.n_cols()
        ));
    }

    // Verify each column
    for (i, (orig_col, result_col)) in original.cols.iter().zip(result.cols.iter()).enumerate() {
        if orig_col.array.len() != result_col.array.len() {
            return Err(format!("Column {} length mismatch", i));
        }
    }

    Ok(())
}

fn test_table_with_nulls_roundtrip(py: Python<'_>) -> Result<(), String> {
    // Create arrays with nulls
    let mut int_arr = IntegerArray::<i32>::default();
    int_arr.push(1);
    int_arr.push_null();
    int_arr.push(3);

    let mut str_arr = StringArray::<u32>::default();
    str_arr.push_str("a");
    str_arr.push_null();
    str_arr.push_str("c");

    // Create FieldArrays
    let fa1 = FieldArray::new(
        Field::new("id", ArrowType::Int32, true, None),
        Array::from_int32(int_arr),
    );
    let fa2 = FieldArray::new(
        Field::new("name", ArrowType::String, true, None),
        Array::from_string32(str_arr),
    );

    // Create Table
    let original = Table::new("nullable_test".to_string(), Some(vec![fa1, fa2]));

    // Roundtrip
    let py_batch = PyRecordBatch::from(original.clone());
    let py_obj = py_batch
        .into_pyobject(py)
        .map_err(|e| format!("Failed to convert to PyArrow: {}", e))?;
    let back: PyRecordBatch = py_obj
        .extract()
        .map_err(|e| format!("Failed to convert from PyArrow: {}", e))?;
    let result = back.into_inner();

    // Verify
    if original.n_rows() != result.n_rows() {
        return Err(format!(
            "Row count mismatch: {} vs {}",
            original.n_rows(),
            result.n_rows()
        ));
    }
    if original.n_cols() != result.n_cols() {
        return Err(format!(
            "Column count mismatch: {} vs {}",
            original.n_cols(),
            result.n_cols()
        ));
    }

    // Verify null counts preserved
    for (i, (orig_col, result_col)) in original.cols.iter().zip(result.cols.iter()).enumerate() {
        if orig_col.null_count != result_col.null_count {
            return Err(format!("Column {} null count mismatch", i));
        }
    }

    Ok(())
}

// 
// Edge Case Tests
// 

fn test_empty_array_roundtrip(py: Python<'_>) -> Result<(), String> {
    let arr = IntegerArray::<i32>::default();
    let original = Array::from_int32(arr);
    if original.len() != 0 {
        return Err(format!("Expected empty array, got length {}", original.len()));
    }

    let result = roundtrip_array(py, original)?;
    if result.len() != 0 {
        return Err(format!("Expected empty result, got length {}", result.len()));
    }
    Ok(())
}

fn test_single_element_array_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<i32>::default();
    arr.push(42);
    let original = Array::from_int32(arr);

    let result = roundtrip_array(py, original.clone())?;
    if result.len() != 1 {
        return Err(format!("Expected 1 element, got {}", result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::Int32(a)),
            Array::NumericArray(NumericArray::Int32(b)),
        ) => {
            if a.get(0) != b.get(0) {
                return Err("Value mismatch".to_string());
            }
        }
        _ => return Err("Type mismatch".to_string()),
    }
    Ok(())
}

fn test_all_nulls_array_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<i32>::default();
    arr.push_null();
    arr.push_null();
    arr.push_null();

    let original = Array::from_int32(arr);
    if original.null_count() != 3 {
        return Err(format!("Expected 3 nulls, got {}", original.null_count()));
    }

    let result = roundtrip_array(py, original)?;
    if result.len() != 3 {
        return Err(format!("Expected 3 elements, got {}", result.len()));
    }
    if result.null_count() != 3 {
        return Err(format!("Expected 3 nulls in result, got {}", result.null_count()));
    }
    Ok(())
}

fn test_empty_string_array_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = StringArray::<u32>::default();
    arr.push_str("");
    arr.push_str("");
    arr.push_str("");

    let original = Array::from_string32(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }

    match (&original, &result) {
        (
            Array::TextArray(TextArray::String32(a)),
            Array::TextArray(TextArray::String32(b)),
        ) => {
            for i in 0..a.len() {
                if a.get(i) != b.get(i) {
                    return Err(format!("Mismatch at index {}", i));
                }
                if a.get(i) != Some("") {
                    return Err(format!("Expected empty string at index {}", i));
                }
            }
        }
        _ => return Err("Type mismatch".to_string()),
    }
    Ok(())
}

fn test_large_array_roundtrip(py: Python<'_>) -> Result<(), String> {
    let mut arr = IntegerArray::<i64>::default();
    for i in 0..10_000 {
        arr.push(i);
    }

    let original = Array::from_int64(arr);
    let result = roundtrip_array(py, original.clone())?;

    if original.len() != result.len() {
        return Err(format!("Length mismatch: {} vs {}", original.len(), result.len()));
    }
    if result.len() != 10_000 {
        return Err(format!("Expected 10000 elements, got {}", result.len()));
    }

    match (&original, &result) {
        (
            Array::NumericArray(NumericArray::Int64(a)),
            Array::NumericArray(NumericArray::Int64(b)),
        ) => {
            // Spot check
            if a.get(0) != b.get(0) {
                return Err("Mismatch at index 0".to_string());
            }
            if a.get(5000) != b.get(5000) {
                return Err("Mismatch at index 5000".to_string());
            }
            if a.get(9999) != b.get(9999) {
                return Err("Mismatch at index 9999".to_string());
            }
        }
        _ => return Err("Type mismatch".to_string()),
    }
    Ok(())
}
