//! ---------------------------------------------------------
//! Builds a demo table and prints it.
//!
//! Run with:
//!     cargo run --example print_arrays
//! ---------------------------------------------------------

#[cfg(feature = "large_string")]
use std::sync::Arc;

use minarrow::aliases::{BoolArr, CatArr, FltArr, IntArr, StrArr};
use minarrow::enums::array::Array;
use minarrow::{
    ArrayV, Bitmask, BitmaskV, MaskedArray, NumericArray, NumericArrayV, Print, TextArray,
    TextArrayV
};
#[cfg(feature = "datetime")]
use minarrow::{DatetimeArray, TemporalArray};

fn main() {
    // Numeric (Integer, Float, all sizes)
    let col_i32 = IntArr::<i32>::from_slice(&[1, 2, 3, 4, 5]);
    let col_u32 = IntArr::<u32>::from_slice(&[100, 200, 300, 400, 500]);
    let col_f32 = FltArr::<f32>::from_slice(&[1.1, 2.2, 3.3, 4.4, 5.5]);

    // Boolean with nulls
    let mut col_bool = BoolArr::from_slice(&[true, false, true, false, true]);
    col_bool.set_null_mask(Some(Bitmask::from_bools(&[true, true, true, false, true])));

    // String and Dictionary/Categorical
    let col_str32 = StrArr::from_slice(&["red", "blue", "green", "yellow", "purple"]);

    let col_cat32 = CatArr::<u32>::from_values(
        ["apple", "banana", "cherry", "banana", "apple"].iter().copied()
    );

    // Datetime
    #[cfg(feature = "datetime")]
    let col_dt32 = DatetimeArray::<i32>::from_slice(
        &[1000, 2000, 3000, 4000, 5000],
        Some(minarrow::enums::time_units::TimeUnit::Milliseconds)
    );
    #[cfg(feature = "datetime")]
    let col_dt64 = DatetimeArray::<i64>::from_slice(
        &[1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000, 5_000_000_000],
        Some(minarrow::enums::time_units::TimeUnit::Nanoseconds)
    );

    col_dt32.print();
    println!("\n");
    col_dt64.print();
    println!("\n");

    // --- Print NumericArray, TextArray, TemporalArray enums
    println!("\n--- Enums: NumericArray, TextArray, TemporalArray ---");
    NumericArray::Int32(Arc::new(col_i32.clone())).print();
    println!("\n");
    NumericArray::UInt32(Arc::new(col_u32.clone())).print();
    println!("\n");
    NumericArray::Float32(Arc::new(col_f32.clone())).print();
    println!("\n");
    TextArray::String32(Arc::new(col_str32.clone())).print();
    println!("\n");
    let _ = &TextArray::Categorical32(Arc::new(col_cat32.clone())).print();

    println!("\n/ *** To display as dates, enable the optional 'chrono' feature *** /\n");

    #[cfg(feature = "datetime")]
    let _ = &TemporalArray::Datetime32(Arc::new(col_dt32.clone())).print();
    println!("\n");
    #[cfg(feature = "datetime")]
    let _ = &TemporalArray::Datetime64(Arc::new(col_dt64.clone())).print();
    println!("\n");

    println!("\n--- Array (top-level) ---");
    Array::from_int32(col_i32.clone()).print();
    println!("\n");
    Array::from_uint32(col_u32.clone()).print();
    println!("\n");
    Array::from_float32(col_f32.clone()).print();
    println!("\n");
    Array::from_string32(col_str32.clone()).print();
    println!("\n");
    Array::from_categorical32(col_cat32.clone()).print();
    println!("\n");
    #[cfg(feature = "datetime")]
    Array::from_datetime_i32(col_dt32.clone()).print();
    println!("\n");
    // --- Print Array Views (ArrayV, NumericArrayV, TextArrayV, TemporalArrayV)
    println!("\n--- Array Views ---");
    ArrayV::new(Array::from_int32(col_i32.clone()), 1, 3).print();

    let num_arr = NumericArray::Int32(Arc::new(col_i32.clone()));
    num_arr.print();
    let num_view = NumericArrayV::new(num_arr, 1, 3);
    num_view.print();

    let txt_arr = TextArray::String32(Arc::new(col_str32.clone()));
    txt_arr.print();
    let txt_view = TextArrayV::new(txt_arr, 1, 3);
    txt_view.print();

    #[cfg(feature = "datetime")]
    {
        use minarrow::TemporalArrayV;

        let tmp_arr = TemporalArray::Datetime32(Arc::new(col_dt32.clone()));
        tmp_arr.print();
        TemporalArrayV::new(tmp_arr, 1, 3).print();
    }

    // --- Print Bitmask and BitmaskV
    println!("\n--- Bitmask & BitmaskV ---");
    let bm = Bitmask::from_bools(&[true, false, true, true, false]);
    bm.print();
    BitmaskV::new(bm.clone(), 1, 3).print();
}
