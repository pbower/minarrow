//! # Roundtrip Tests for Rust -> Python -> Rust
//!
//! Tests MinArrow -> PyArrow -> MinArrow conversion for all supported types.
//!
//! ## Important: These tests cannot run via `cargo test`
//!
//! PyO3's `extension-module` feature (enabled by default) tells the linker not to link
//! against libpython, since Python loads the extension at runtime. This means standalone
//! Rust binaries that call `Python::with_gil()` will fail with undefined symbol errors.
//!
//! ## How to run these tests
//!
//! Use the `run_tests` example instead, which disables `extension-module`:
//!
//! ```bash
//! cd pyo3
//!
//! # Build without extension-module to enable libpython linking
//! cargo build --example run_tests \
//!     --no-default-features \
//!     --features "datetime,extended_numeric_types,extended_categorical"
//!
//! # Find which Python the binary links against
//! ldd target/debug/examples/run_tests | grep python
//! # e.g., libpython3.12.so.1.0 => /usr/local/lib/libpython3.12.so.1.0
//!
//! # Set PYTHONHOME to that Python's prefix and run
//! PYTHONHOME=/usr/local cargo run --example run_tests \
//!     --no-default-features \
//!     --features "datetime,extended_numeric_types,extended_categorical"
//! ```
//!
//! The example at `examples/run_tests.rs` contains equivalent tests to those below.

#[cfg(test)]
mod tests {
    use crate::types::{PyArray, PyRecordBatch};
    use minarrow::ffi::arrow_dtype::ArrowType;
    use minarrow::{
        Array, BooleanArray, Field, FieldArray, FloatArray, IntegerArray, MaskedArray,
        NumericArray, StringArray, Table, TextArray,
    };
    use pyo3::prelude::*;
    use pyo3::types::IntoPyDict;
    use std::sync::Arc;

    /// Helper to run a test within Python GIL
    fn with_py<F, R>(f: F) -> R
    where
        F: FnOnce(Python<'_>) -> R,
    {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(f)
    }

    /// Helper to convert MinArrow Array to PyArrow and back
    fn roundtrip_array(py: Python<'_>, array: Array) -> PyResult<Array> {
        let py_array = PyArray::from(array);
        let py_obj = py_array.into_pyobject(py)?;
        let back: PyArray = py_obj.extract()?;
        Ok(back.into_inner().array)
    }

    // Integer Array Tests

    #[test]
    fn test_int32_roundtrip() {
        with_py(|py| {
            let mut arr = IntegerArray::<i32>::default();
            arr.push(1);
            arr.push(2);
            arr.push(3);
            arr.push(i32::MAX);
            arr.push(i32::MIN);

            let original = Array::from_int32(arr);
            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            match (&original, &result) {
                (
                    Array::NumericArray(NumericArray::Int32(a)),
                    Array::NumericArray(NumericArray::Int32(b)),
                ) => {
                    for i in 0..a.len() {
                        assert_eq!(a.get(i), b.get(i), "Mismatch at index {}", i);
                    }
                }
                _ => panic!("Type mismatch after roundtrip"),
            }
            println!("✓ i32 roundtrip passed");
        });
    }

    #[test]
    fn test_int64_roundtrip() {
        with_py(|py| {
            let mut arr = IntegerArray::<i64>::default();
            arr.push(100);
            arr.push(200);
            arr.push(300);
            arr.push(i64::MAX);
            arr.push(i64::MIN);

            let original = Array::from_int64(arr);
            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            match (&original, &result) {
                (
                    Array::NumericArray(NumericArray::Int64(a)),
                    Array::NumericArray(NumericArray::Int64(b)),
                ) => {
                    for i in 0..a.len() {
                        assert_eq!(a.get(i), b.get(i), "Mismatch at index {}", i);
                    }
                }
                _ => panic!("Type mismatch after roundtrip"),
            }
            println!("✓ i64 roundtrip passed");
        });
    }

    #[test]
    fn test_uint32_roundtrip() {
        with_py(|py| {
            let mut arr = IntegerArray::<u32>::default();
            arr.push(0);
            arr.push(1);
            arr.push(u32::MAX);

            let original = Array::from_uint32(arr);
            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            match (&original, &result) {
                (
                    Array::NumericArray(NumericArray::UInt32(a)),
                    Array::NumericArray(NumericArray::UInt32(b)),
                ) => {
                    for i in 0..a.len() {
                        assert_eq!(a.get(i), b.get(i), "Mismatch at index {}", i);
                    }
                }
                _ => panic!("Type mismatch after roundtrip"),
            }
            println!("✓ u32 roundtrip passed");
        });
    }

    #[test]
    fn test_uint64_roundtrip() {
        with_py(|py| {
            let mut arr = IntegerArray::<u64>::default();
            arr.push(0);
            arr.push(1);
            arr.push(u64::MAX);

            let original = Array::from_uint64(arr);
            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            match (&original, &result) {
                (
                    Array::NumericArray(NumericArray::UInt64(a)),
                    Array::NumericArray(NumericArray::UInt64(b)),
                ) => {
                    for i in 0..a.len() {
                        assert_eq!(a.get(i), b.get(i), "Mismatch at index {}", i);
                    }
                }
                _ => panic!("Type mismatch after roundtrip"),
            }
            println!("✓ u64 roundtrip passed");
        });
    }

    // Float Array Tests

    #[test]
    fn test_float32_roundtrip() {
        with_py(|py| {
            let mut arr = FloatArray::<f32>::default();
            arr.push(1.5);
            arr.push(2.25);
            arr.push(f32::MAX);
            arr.push(f32::MIN);
            arr.push(0.0);
            arr.push(-0.0);

            let original = Array::from_float32(arr);
            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            match (&original, &result) {
                (
                    Array::NumericArray(NumericArray::Float32(a)),
                    Array::NumericArray(NumericArray::Float32(b)),
                ) => {
                    for i in 0..a.len() {
                        let av = a.get(i).unwrap();
                        let bv = b.get(i).unwrap();
                        assert!(
                            (av - bv).abs() < f32::EPSILON || (av.is_nan() && bv.is_nan()),
                            "Mismatch at index {}: {} vs {}",
                            i,
                            av,
                            bv
                        );
                    }
                }
                _ => panic!("Type mismatch after roundtrip"),
            }
            println!("✓ f32 roundtrip passed");
        });
    }

    #[test]
    fn test_float64_roundtrip() {
        with_py(|py| {
            let mut arr = FloatArray::<f64>::default();
            arr.push(1.5);
            arr.push(2.25);
            arr.push(std::f64::consts::PI);
            arr.push(std::f64::consts::E);
            arr.push(f64::MAX);
            arr.push(f64::MIN);

            let original = Array::from_float64(arr);
            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            match (&original, &result) {
                (
                    Array::NumericArray(NumericArray::Float64(a)),
                    Array::NumericArray(NumericArray::Float64(b)),
                ) => {
                    for i in 0..a.len() {
                        let av = a.get(i).unwrap();
                        let bv = b.get(i).unwrap();
                        assert!(
                            (av - bv).abs() < f64::EPSILON || (av.is_nan() && bv.is_nan()),
                            "Mismatch at index {}: {} vs {}",
                            i,
                            av,
                            bv
                        );
                    }
                }
                _ => panic!("Type mismatch after roundtrip"),
            }
            println!("✓ f64 roundtrip passed");
        });
    }

    // Boolean Array Tests

    #[test]
    fn test_boolean_roundtrip() {
        with_py(|py| {
            let mut arr = BooleanArray::default();
            arr.push(true);
            arr.push(false);
            arr.push(true);
            arr.push(false);
            arr.push(true);

            let original = Array::from_bool(arr);
            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            match (&original, &result) {
                (Array::BooleanArray(a), Array::BooleanArray(b)) => {
                    for i in 0..a.len() {
                        assert_eq!(a.get(i), b.get(i), "Mismatch at index {}", i);
                    }
                }
                _ => panic!("Type mismatch after roundtrip"),
            }
            println!("✓ boolean roundtrip passed");
        });
    }

    // String Array Tests

    #[test]
    fn test_string32_roundtrip() {
        with_py(|py| {
            let mut arr = StringArray::<u32>::default();
            arr.push_str("hello");
            arr.push_str("world");
            arr.push_str("");
            arr.push_str("with spaces");
            arr.push_str("special: αβγ 日本語 🎉");

            let original = Array::from_string32(arr);
            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            match (&original, &result) {
                (
                    Array::TextArray(TextArray::String32(a)),
                    Array::TextArray(TextArray::String32(b)),
                ) => {
                    for i in 0..a.len() {
                        assert_eq!(a.get(i), b.get(i), "Mismatch at index {}", i);
                    }
                }
                _ => panic!("Type mismatch after roundtrip"),
            }
            println!("✓ string (u32 offsets) roundtrip passed");
        });
    }

    #[test]
    #[cfg(feature = "large_string")]
    fn test_string64_roundtrip() {
        with_py(|py| {
            let mut arr = StringArray::<u64>::default();
            arr.push_str("large");
            arr.push_str("string");
            arr.push_str("array");

            let original = Array::from_string64(arr);
            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            match (&original, &result) {
                (
                    Array::TextArray(TextArray::String64(a)),
                    Array::TextArray(TextArray::String64(b)),
                ) => {
                    for i in 0..a.len() {
                        assert_eq!(a.get(i), b.get(i), "Mismatch at index {}", i);
                    }
                }
                _ => panic!("Type mismatch after roundtrip"),
            }
            println!("✓ large string (u64 offsets) roundtrip passed");
        });
    }

    // Nullable Array Tests

    #[test]
    fn test_nullable_int32_roundtrip() {
        with_py(|py| {
            let mut arr = IntegerArray::<i32>::default();
            arr.push(1);
            arr.push_null();
            arr.push(3);
            arr.push_null();
            arr.push(5);

            let original = Array::from_int32(arr);
            assert_eq!(original.null_count(), 2);

            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            assert_eq!(original.null_count(), result.null_count());

            match (&original, &result) {
                (
                    Array::NumericArray(NumericArray::Int32(a)),
                    Array::NumericArray(NumericArray::Int32(b)),
                ) => {
                    for i in 0..a.len() {
                        assert_eq!(a.is_null(i), b.is_null(i), "Null mismatch at index {}", i);
                        if !a.is_null(i) {
                            assert_eq!(a.get(i), b.get(i), "Value mismatch at index {}", i);
                        }
                    }
                }
                _ => panic!("Type mismatch after roundtrip"),
            }
            println!("✓ nullable i32 roundtrip passed");
        });
    }

    #[test]
    fn test_nullable_float64_roundtrip() {
        with_py(|py| {
            let mut arr = FloatArray::<f64>::default();
            arr.push(1.1);
            arr.push_null();
            arr.push(3.3);
            arr.push_null();

            let original = Array::from_float64(arr);
            assert_eq!(original.null_count(), 2);

            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            assert_eq!(original.null_count(), result.null_count());
            println!("✓ nullable f64 roundtrip passed");
        });
    }

    #[test]
    fn test_nullable_string_roundtrip() {
        with_py(|py| {
            let mut arr = StringArray::<u32>::default();
            arr.push_str("hello");
            arr.push_null();
            arr.push_str("world");
            arr.push_null();

            let original = Array::from_string32(arr);
            assert_eq!(original.null_count(), 2);

            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            assert_eq!(original.null_count(), result.null_count());

            match (&original, &result) {
                (
                    Array::TextArray(TextArray::String32(a)),
                    Array::TextArray(TextArray::String32(b)),
                ) => {
                    for i in 0..a.len() {
                        assert_eq!(a.is_null(i), b.is_null(i), "Null mismatch at index {}", i);
                        if !a.is_null(i) {
                            assert_eq!(a.get(i), b.get(i), "Value mismatch at index {}", i);
                        }
                    }
                }
                _ => panic!("Type mismatch after roundtrip"),
            }
            println!("✓ nullable string roundtrip passed");
        });
    }

    #[test]
    fn test_nullable_boolean_roundtrip() {
        with_py(|py| {
            let mut arr = BooleanArray::default();
            arr.push(true);
            arr.push_null();
            arr.push(false);
            arr.push_null();

            let original = Array::from_bool(arr);
            assert_eq!(original.null_count(), 2);

            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            assert_eq!(original.null_count(), result.null_count());
            println!("✓ nullable boolean roundtrip passed");
        });
    }

    // Table / RecordBatch Tests

    #[test]
    fn test_table_roundtrip() {
        with_py(|py| {
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
            let py_obj = py_batch.into_pyobject(py).unwrap();
            let back: PyRecordBatch = py_obj.extract().unwrap();
            let result = back.into_inner();

            // Verify
            assert_eq!(original.n_rows(), result.n_rows());
            assert_eq!(original.n_cols(), result.n_cols());

            // Verify each column
            for (i, (orig_col, result_col)) in
                original.cols.iter().zip(result.cols.iter()).enumerate()
            {
                assert_eq!(
                    orig_col.array.len(),
                    result_col.array.len(),
                    "Column {} length mismatch",
                    i
                );
            }

            println!("✓ table roundtrip passed (4 columns, 3 rows)");
        });
    }

    #[test]
    fn test_table_with_nulls_roundtrip() {
        with_py(|py| {
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
            let py_obj = py_batch.into_pyobject(py).unwrap();
            let back: PyRecordBatch = py_obj.extract().unwrap();
            let result = back.into_inner();

            // Verify
            assert_eq!(original.n_rows(), result.n_rows());
            assert_eq!(original.n_cols(), result.n_cols());

            // Verify null counts preserved
            for (i, (orig_col, result_col)) in
                original.cols.iter().zip(result.cols.iter()).enumerate()
            {
                assert_eq!(
                    orig_col.null_count, result_col.null_count,
                    "Column {} null count mismatch",
                    i
                );
            }

            println!("✓ table with nulls roundtrip passed");
        });
    }

    // Edge Case Tests

    #[test]
    fn test_empty_array_roundtrip() {
        with_py(|py| {
            let arr = IntegerArray::<i32>::default();
            let original = Array::from_int32(arr);
            assert_eq!(original.len(), 0);

            let result = roundtrip_array(py, original.clone()).unwrap();
            assert_eq!(result.len(), 0);
            println!("✓ empty array roundtrip passed");
        });
    }

    #[test]
    fn test_single_element_array_roundtrip() {
        with_py(|py| {
            let mut arr = IntegerArray::<i32>::default();
            arr.push(42);
            let original = Array::from_int32(arr);

            let result = roundtrip_array(py, original.clone()).unwrap();
            assert_eq!(result.len(), 1);

            match (&original, &result) {
                (
                    Array::NumericArray(NumericArray::Int32(a)),
                    Array::NumericArray(NumericArray::Int32(b)),
                ) => {
                    assert_eq!(a.get(0), b.get(0));
                }
                _ => panic!("Type mismatch"),
            }
            println!("✓ single element array roundtrip passed");
        });
    }

    #[test]
    fn test_all_nulls_array_roundtrip() {
        with_py(|py| {
            let mut arr = IntegerArray::<i32>::default();
            arr.push_null();
            arr.push_null();
            arr.push_null();

            let original = Array::from_int32(arr);
            assert_eq!(original.null_count(), 3);

            let result = roundtrip_array(py, original.clone()).unwrap();
            assert_eq!(result.len(), 3);
            assert_eq!(result.null_count(), 3);
            println!("✓ all-nulls array roundtrip passed");
        });
    }

    #[test]
    fn test_empty_string_array_roundtrip() {
        with_py(|py| {
            let mut arr = StringArray::<u32>::default();
            arr.push_str("");
            arr.push_str("");
            arr.push_str("");

            let original = Array::from_string32(arr);
            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            match (&original, &result) {
                (
                    Array::TextArray(TextArray::String32(a)),
                    Array::TextArray(TextArray::String32(b)),
                ) => {
                    for i in 0..a.len() {
                        assert_eq!(a.get(i), b.get(i));
                        assert_eq!(a.get(i), Some(""));
                    }
                }
                _ => panic!("Type mismatch"),
            }
            println!("✓ empty strings array roundtrip passed");
        });
    }

    #[test]
    fn test_large_array_roundtrip() {
        with_py(|py| {
            let mut arr = IntegerArray::<i64>::default();
            for i in 0..10_000 {
                arr.push(i);
            }

            let original = Array::from_int64(arr);
            let result = roundtrip_array(py, original.clone()).unwrap();

            assert_eq!(original.len(), result.len());
            assert_eq!(result.len(), 10_000);

            match (&original, &result) {
                (
                    Array::NumericArray(NumericArray::Int64(a)),
                    Array::NumericArray(NumericArray::Int64(b)),
                ) => {
                    // Spot check
                    assert_eq!(a.get(0), b.get(0));
                    assert_eq!(a.get(5000), b.get(5000));
                    assert_eq!(a.get(9999), b.get(9999));
                }
                _ => panic!("Type mismatch"),
            }
            println!("✓ large array (10k elements) roundtrip passed");
        });
    }
}
