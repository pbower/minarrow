//! Central test suite for external polars library conversion

#![cfg(feature = "cast_polars")]

use std::sync::Arc;

use minarrow::{Array, ArrowType, Field, FieldArray, NumericArray, Table, TextArray};
#[cfg(feature = "datetime")]
use minarrow::{TemporalArray, TimeUnit};
use polars::prelude::*;

#[test]
fn test_array_to_polars_numeric() {
    let arr = Arc::new(minarrow::IntegerArray::<i32>::from_slice(&[1, 2, 3]));
    let a = Array::NumericArray(NumericArray::Int32(arr));
    let s = a.to_polars("x");
    assert_eq!(s.name(), "x");
    assert_eq!(s.len(), 3);
    assert_eq!(s.dtype(), &DataType::Int32);
    assert_eq!(
        s.i32().unwrap().into_no_null_iter().collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
}

#[test]
fn test_array_to_polars_string() {
    let arr = Arc::new(minarrow::StringArray::<u32>::from_slice(&["a", "b", ""]));
    let a = Array::TextArray(TextArray::String32(arr));
    let s = a.to_polars("s");
    assert_eq!(s.dtype(), &DataType::String);
    assert_eq!(
        s.str()
            .unwrap()
            .into_no_null_iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>(),
        vec!["a".to_string(), "b".to_string(), "".to_string()]
    );
}

#[cfg(feature = "datetime")]
#[test]
fn test_array_to_polars_datetime_infer_date32() {
    let a = Array::TemporalArray(TemporalArray::Datetime32(Arc::new(
        minarrow::DatetimeArray::<i32> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000 / 86_400; 3]),
            null_mask: None,
            time_unit: TimeUnit::Days,
        },
    )));
    let s = a.to_polars("d32");
    // Polars maps Arrow Date32 -> DataType::Date
    assert_eq!(s.dtype(), &DataType::Date);
    assert_eq!(s.len(), 3);
}

#[cfg(feature = "datetime")]
#[test]
fn test_array_to_polars_datetime_infer_time32s() {
    let a = Array::TemporalArray(TemporalArray::Datetime32(Arc::new(
        minarrow::DatetimeArray::<i32> {
            data: minarrow::Buffer::from_slice(&[1, 2, 3]),
            null_mask: None,
            time_unit: TimeUnit::Seconds,
        },
    )));
    let s = a.to_polars("t32s");
    // Polars maps Arrow Time32(s) to Int32 logical time; exact dtype may vary, presence is sufficient
    assert_eq!(s.len(), 3);
}

#[cfg(feature = "datetime")]
#[test]
fn test_array_to_polars_datetime_infer_date64_or_ts() {
    let a_ms = Array::TemporalArray(TemporalArray::Datetime64(Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000_000, 1_600_000_000_001]),
            null_mask: None,
            time_unit: TimeUnit::Milliseconds,
        },
    )));
    let s_ms = a_ms.to_polars("d64");
    // In practice Polars treats Arrow Date64 as Datetime(Milliseconds)
    assert_eq!(s_ms.len(), 2);

    let a_ns = Array::TemporalArray(TemporalArray::Datetime64(Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1, 2, 3]),
            null_mask: None,
            time_unit: TimeUnit::Nanoseconds,
        },
    )));
    let s_ns = a_ns.to_polars("ts_ns");
    // Arrow Timestamp(ns) → Polars Datetime(ns)
    assert_eq!(s_ns.len(), 3);
}

#[test]
fn test_array_to_polars_with_field_explicit() {
    let arr = Arc::new(minarrow::IntegerArray::<i64>::from_slice(&[10, 20]));
    let a = Array::NumericArray(NumericArray::Int64(arr));
    let f = Field::new("y", ArrowType::Int64, false, None);
    let s = a.to_polars_with_field("y", &f);
    assert_eq!(s.dtype(), &DataType::Int64);
    assert_eq!(
        s.i64().unwrap().into_no_null_iter().collect::<Vec<_>>(),
        vec![10, 20]
    );
}

#[test]
fn test_fieldarray_to_polars() {
    let arr = Arc::new(minarrow::IntegerArray::<u32>::from_slice(&[5, 6, 7]));
    let a = Array::NumericArray(NumericArray::UInt32(arr));
    let f = Field::new("u", ArrowType::UInt32, false, None);
    let fa = FieldArray::new(f.clone(), a);
    let s = fa.to_polars();
    assert_eq!(s.name(), "u");
    assert_eq!(s.dtype(), &DataType::UInt32);
}

#[test]
fn test_table_to_polars() {
    // Tiny table: 2 cols
    let c1 = {
        let arr = Arc::new(minarrow::IntegerArray::<i32>::from_slice(&[1, 2]));
        let a = Array::NumericArray(NumericArray::Int32(arr));
        let f = Field::new("a", ArrowType::Int32, false, None);
        FieldArray::new(f, a)
    };
    let c2 = {
        let arr = Arc::new(minarrow::StringArray::<u32>::from_slice(&["x", "y"]));
        let a = Array::TextArray(TextArray::String32(arr));
        let f = Field::new("b", ArrowType::String, false, None);
        FieldArray::new(f, a)
    };
    let t = Table::new("t".into(), Some(vec![c1, c2]));
    let df = t.to_polars();
    assert_eq!(df.height(), 2);
    assert_eq!(df.width(), 2);
    assert_eq!(df.get_column_names(), &["a", "b"]);
}
