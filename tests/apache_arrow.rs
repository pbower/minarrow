//! Central test suite for Apache Arrow conversion

#![cfg(feature = "cast_arrow")]

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, Date32Array, Date64Array, Int32Array, Int64Array, StringArray,
    Time32SecondArray, TimestampNanosecondArray, UInt32Array,
};
use arrow::datatypes::{DataType as ADataType, TimeUnit as ATimeUnit};
use arrow::record_batch::RecordBatch;

use minarrow::{Array as MArray, ArrowType, Field, FieldArray, NumericArray, Table, TextArray};

#[cfg(feature = "datetime")]
use minarrow::{TemporalArray, TimeUnit};

// -------------------------------
// Array -> Arrow (numeric)
// -------------------------------
#[test]
fn test_array_to_arrow_numeric() {
    let arr = Arc::new(minarrow::IntegerArray::<i32>::from_slice(&[1, 2, 3]));
    let a = MArray::NumericArray(NumericArray::Int32(arr));
    let ar: ArrayRef = a.to_apache_arrow("x");

    assert_eq!(ar.data_type(), &ADataType::Int32);

    let col = ar.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(col.len(), 3);
    assert_eq!(col.value(0), 1);
    assert_eq!(col.value(1), 2);
    assert_eq!(col.value(2), 3);
}

// -------------------------------
// Array -> Arrow (utf8)
// -------------------------------
#[test]
fn test_array_to_arrow_string() {
    let arr = Arc::new(minarrow::StringArray::<u32>::from_slice(&["a", "b", ""]));
    let a = MArray::TextArray(TextArray::String32(arr));
    let ar = a.to_apache_arrow("s");

    assert_eq!(ar.data_type(), &ADataType::Utf8);

    let col = ar.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(col.len(), 3);
    assert_eq!(col.value(0), "a");
    assert_eq!(col.value(1), "b");
    assert_eq!(col.value(2), "");
}

#[cfg(feature = "datetime")]
#[test]
fn test_array_to_arrow_datetime_infer_date32() {
    // Date32 = days since epoch
    let a = MArray::TemporalArray(TemporalArray::Datetime32(Arc::new(
        minarrow::DatetimeArray::<i32> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000 / 86_400; 3]),
            null_mask: None,
            time_unit: TimeUnit::Days,
        },
    )));
    let ar = a.to_apache_arrow("d32");

    assert_eq!(ar.data_type(), &ADataType::Date32);
    let col = ar.as_any().downcast_ref::<Date32Array>().unwrap();
    assert_eq!(col.len(), 3);
}

#[cfg(feature = "datetime")]
#[test]
fn test_array_to_arrow_datetime_infer_time32s() {
    // Time32(Second) — use explicit Field so logical type matches Seconds
    let a = MArray::TemporalArray(TemporalArray::Datetime32(Arc::new(
        minarrow::DatetimeArray::<i32> {
            data: minarrow::Buffer::from_slice(&[1, 2, 3]),
            null_mask: None,
            time_unit: TimeUnit::Seconds,
        },
    )));
    let f = Field::new("t32s", ArrowType::Time32(TimeUnit::Seconds), false, None);
    let ar = a.to_apache_arrow_with_field(&f);

    assert_eq!(ar.data_type(), &ADataType::Time32(ATimeUnit::Second));
    let col = ar.as_any().downcast_ref::<Time32SecondArray>().unwrap();
    assert_eq!(col.len(), 3);
    assert_eq!(col.value(0), 1);
    assert_eq!(col.value(1), 2);
    assert_eq!(col.value(2), 3);
}

#[cfg(feature = "datetime")]
#[test]
fn test_array_to_arrow_datetime_infer_date64_and_ts_ns() {
    // Date64 = ms since epoch
    let a_ms = MArray::TemporalArray(TemporalArray::Datetime64(Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1_600_000_000_000, 1_600_000_000_001]),
            null_mask: None,
            time_unit: TimeUnit::Milliseconds,
        },
    )));
    let ar_ms = a_ms.to_apache_arrow("d64");
    assert_eq!(ar_ms.data_type(), &ADataType::Date64);
    let c_ms = ar_ms.as_any().downcast_ref::<Date64Array>().unwrap();
    assert_eq!(c_ms.len(), 2);
    assert_eq!(c_ms.value(0), 1_600_000_000_000);
    assert_eq!(c_ms.value(1), 1_600_000_000_001);

    // Timestamp(ns) — explicit Field
    let a_ns = MArray::TemporalArray(TemporalArray::Datetime64(Arc::new(
        minarrow::DatetimeArray::<i64> {
            data: minarrow::Buffer::from_slice(&[1, 2, 3]),
            null_mask: None,
            time_unit: TimeUnit::Nanoseconds,
        },
    )));
    let f_tsns = Field::new(
        "ts_ns",
        ArrowType::Timestamp(TimeUnit::Nanoseconds, None),
        false,
        None,
    );
    let ar_ns = a_ns.to_apache_arrow_with_field(&f_tsns);
    assert_eq!(
        ar_ns.data_type(),
        &ADataType::Timestamp(ATimeUnit::Nanosecond, None)
    );
    let c_ns = ar_ns
        .as_any()
        .downcast_ref::<TimestampNanosecondArray>()
        .unwrap();
    assert_eq!(c_ns.len(), 3);
    assert_eq!(c_ns.value(0), 1);
    assert_eq!(c_ns.value(1), 2);
    assert_eq!(c_ns.value(2), 3);
}

#[test]
fn test_array_to_arrow_with_field_explicit() {
    let arr = Arc::new(minarrow::IntegerArray::<i64>::from_slice(&[10, 20]));
    let a = MArray::NumericArray(NumericArray::Int64(arr));
    let f = Field::new("y", ArrowType::Int64, false, None);
    let ar = a.to_apache_arrow_with_field(&f);

    assert_eq!(ar.data_type(), &ADataType::Int64);
    let col = ar.as_any().downcast_ref::<Int64Array>().unwrap();
    assert_eq!(col.len(), 2);
    assert_eq!(col.value(0), 10);
    assert_eq!(col.value(1), 20);
}

#[test]
fn test_fieldarray_to_arrow() {
    let arr = Arc::new(minarrow::IntegerArray::<u32>::from_slice(&[5, 6, 7]));
    let a = MArray::NumericArray(NumericArray::UInt32(arr));
    let f = Field::new("u", ArrowType::UInt32, false, None);
    let fa = FieldArray::new(f.clone(), a);

    let ar = fa.to_apache_arrow();
    assert_eq!(ar.data_type(), &ADataType::UInt32);

    let col = ar.as_any().downcast_ref::<UInt32Array>().unwrap();
    assert_eq!(col.len(), 3);
    assert_eq!(col.value(0), 5);
    assert_eq!(col.value(1), 6);
    assert_eq!(col.value(2), 7);
}

#[test]
fn test_table_to_arrow_record_batch() {
    // 2 columns
    let c1 = {
        let arr = Arc::new(minarrow::IntegerArray::<i32>::from_slice(&[1, 2]));
        let a = MArray::NumericArray(NumericArray::Int32(arr));
        let f = Field::new("a", ArrowType::Int32, false, None);
        FieldArray::new(f, a)
    };
    let c2 = {
        let arr = Arc::new(minarrow::StringArray::<u32>::from_slice(&["x", "y"]));
        let a = MArray::TextArray(TextArray::String32(arr));
        let f = Field::new("b", ArrowType::String, false, None);
        FieldArray::new(f, a)
    };
    let t = Table::new("t".into(), Some(vec![c1, c2]));

    let rb: RecordBatch = t.to_apache_arrow();
    assert_eq!(rb.num_rows(), 2);
    assert_eq!(rb.num_columns(), 2);

    let a = rb.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    let b = rb.column(1).as_any().downcast_ref::<StringArray>().unwrap();

    assert_eq!(a.value(0), 1);
    assert_eq!(a.value(1), 2);
    assert_eq!(b.value(0), "x");
    assert_eq!(b.value(1), "y");
}
