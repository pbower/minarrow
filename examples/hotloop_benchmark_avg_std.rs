//! ---------------------------------------------------------
//! Runs averaged sum benchmark comparisons on `Minarrow` and `Arrow-Rs`,
//! at various layers of library abstraction, using SIMD:
//!
//!     1. Raw Vec / Vec64
//!     2. Typed "inner" arrays
//!     3. Top-level unified `Array` type
//!
//! Run with:
//!     RUSTFLAGS="-C target-cpu=native" cargo run --release --example hotloop_benchmark_simd
//!
//! The *RUSTFLAGS* argument ensures it compiles to your host instruction-set.
//!
//! Use 2, 4, 8, or 16 LANES as per your processor's SIMD support.
//! ---------------------------------------------------------

#[cfg(feature = "cast_arrow")]
use crate::benchmarks_avg::run_benchmark;
#[cfg(feature = "cast_arrow")]
mod benchmarks_avg {
    use std::hint::black_box;
    use std::sync::Arc;
    use std::time::Instant;

    use arrow::array::{
        Array as ArrowArrayTrait, ArrayRef, Float64Array as ArrowF64Array,
        Int64Array as ArrowI64Array
    };
    use minarrow::{Array, Buffer, FloatArray, IntegerArray, NumericArray, Vec64};

    use crate::fmt_duration_ns;

    const N: usize = 1000;
    const ITERATIONS: usize = 1000;

    pub(crate) fn run_benchmark() {
        let mut total_arrow_dyn_i64 = std::time::Duration::ZERO;
        let mut total_arrow_struct_i64 = std::time::Duration::ZERO;
        let mut total_minarrow_enum_i64 = std::time::Duration::ZERO;
        let mut total_minarrow_direct_i64 = std::time::Duration::ZERO;
        let mut total_vec_i64 = std::time::Duration::ZERO;
        let mut total_arrow_dyn_f64 = std::time::Duration::ZERO;
        let mut total_arrow_struct_f64 = std::time::Duration::ZERO;
        let mut total_minarrow_enum_f64 = std::time::Duration::ZERO;
        let mut total_minarrow_direct_f64 = std::time::Duration::ZERO;
        let mut total_vec_f64 = std::time::Duration::ZERO;

        for _ in 0..ITERATIONS {
            // ----------- Arrow i64 (dynamic) -----------
            let data: Vec<i64> = (0..N as i64).collect();
            let start = Instant::now();
            let arr: ArrayRef = Arc::new(ArrowI64Array::from(data));
            let mut acc = 0i64;
            if let Some(int) = arr.as_any().downcast_ref::<ArrowI64Array>() {
                for i in 0..int.len() {
                    acc += int.value(i);
                }
            }
            let dur_arrow_dyn_i64 = start.elapsed();
            total_arrow_dyn_i64 += dur_arrow_dyn_i64;
            black_box(acc);

            // ----------- Arrow i64 (struct direct) -----------
            let data: Vec<i64> = (0..N as i64).collect();
            let start = Instant::now();
            let arr = ArrowI64Array::from(data);
            let mut acc = 0i64;
            for i in 0..arr.len() {
                acc += arr.value(i);
            }
            let dur_arrow_struct_i64 = start.elapsed();
            total_arrow_struct_i64 += dur_arrow_struct_i64;
            black_box(acc);

            // ----------- Minarrow i64 (enum) -----------
            let min_data: Vec64<i64> = (0..N as i64).collect();
            let start = Instant::now();
            let array = Array::NumericArray(NumericArray::Int64(Arc::new(IntegerArray {
                data: Buffer::from(min_data),
                null_mask: None
            })));
            let mut acc = 0i64;
            let int_arr = array.num().i64().unwrap();
            let slice = int_arr.data.as_slice();
            for &v in slice {
                acc += v;
            }
            let dur_minarrow_enum_i64 = start.elapsed();
            total_minarrow_enum_i64 += dur_minarrow_enum_i64;
            black_box(acc);

            // ----------- Minarrow i64 (direct struct, no enum) -----------
            let min_data: Vec64<i64> = (0..N as i64).collect();
            let start = Instant::now();
            let int_arr = IntegerArray {
                data: Buffer::from(min_data),
                null_mask: None
            };
            let mut acc = 0i64;
            let slice = int_arr.data.as_slice();
            for &v in slice {
                acc += v;
            }
            let dur_minarrow_direct_i64 = start.elapsed();
            total_minarrow_direct_i64 += dur_minarrow_direct_i64;
            black_box(acc);

            // ----------- Raw Vec<i64> -----------
            let raw_vec: Vec<i64> = (0..N as i64).collect();
            let start = Instant::now();
            let mut acc = 0i64;
            for &v in &raw_vec {
                acc += v;
            }
            let dur_vec_i64 = start.elapsed();
            total_vec_i64 += dur_vec_i64;
            black_box(acc);

            // ----------- Arrow f64 (dynamic) -----------
            let data_f64: Vec<f64> = (0..N as i64).map(|x| x as f64).collect();
            let start = Instant::now();
            let arr: ArrayRef = Arc::new(ArrowF64Array::from(data_f64));
            let mut acc = 0.0f64;
            if let Some(f) = arr.as_any().downcast_ref::<ArrowF64Array>() {
                for i in 0..f.len() {
                    acc += f.value(i);
                }
            }
            let dur_arrow_dyn_f64 = start.elapsed();
            total_arrow_dyn_f64 += dur_arrow_dyn_f64;
            black_box(acc);

            // ----------- Arrow f64 (struct direct) -----------
            let data_f64: Vec<f64> = (0..N as i64).map(|x| x as f64).collect();
            let start = Instant::now();
            let arr = ArrowF64Array::from(data_f64);
            let mut acc = 0.0f64;
            for i in 0..arr.len() {
                acc += arr.value(i);
            }
            let dur_arrow_struct_f64 = start.elapsed();
            total_arrow_struct_f64 += dur_arrow_struct_f64;
            black_box(acc);

            // ----------- Minarrow f64 (enum) -----------
            let min_data_f64: Vec64<f64> = (0..N as i64).map(|x| x as f64).collect();
            let start = Instant::now();
            let array = Array::NumericArray(NumericArray::Float64(Arc::new(FloatArray {
                data: Buffer::from(min_data_f64),
                null_mask: None
            })));
            let mut acc = 0.0f64;
            let float_arr = array.num().f64().unwrap();
            let slice = float_arr.data.as_slice();
            for &v in slice {
                acc += v;
            }
            let dur_minarrow_enum_f64 = start.elapsed();
            total_minarrow_enum_f64 += dur_minarrow_enum_f64;
            black_box(acc);

            // ----------- Minarrow f64 (direct struct, no enum) -----------
            let min_data_f64: Vec64<f64> = (0..N as i64).map(|x| x as f64).collect();
            let start = Instant::now();
            let float_arr = FloatArray {
                data: Buffer::from(min_data_f64),
                null_mask: None
            };
            let mut acc = 0.0f64;
            let slice = float_arr.data.as_slice();
            for &v in slice {
                acc += v;
            }
            let dur_minarrow_direct_f64 = start.elapsed();
            total_minarrow_direct_f64 += dur_minarrow_direct_f64;
            black_box(acc);

            // ----------- Raw Vec<f64> -----------
            let raw_vec: Vec<f64> = (0..N as i64).map(|x| x as f64).collect();
            let start = Instant::now();
            let mut acc = 0.0f64;
            for &v in &raw_vec {
                acc += v;
            }
            let dur_vec_f64 = start.elapsed();
            total_vec_f64 += dur_vec_f64;
            black_box(acc);
        }

        println!("Averaged Results from {} runs:", ITERATIONS);
        println!("---------------------------------");

        let avg_vec_i64 = total_vec_i64.as_nanos() as f64 / ITERATIONS as f64;
        let avg_minarrow_direct_i64 =
            total_minarrow_direct_i64.as_nanos() as f64 / ITERATIONS as f64;
        let avg_arrow_struct_i64 = total_arrow_struct_i64.as_nanos() as f64 / ITERATIONS as f64;
        let avg_minarrow_enum_i64 = total_minarrow_enum_i64.as_nanos() as f64 / ITERATIONS as f64;
        let avg_arrow_dyn_i64 = total_arrow_dyn_i64.as_nanos() as f64 / ITERATIONS as f64;

        let avg_vec_f64 = total_vec_f64.as_nanos() as f64 / ITERATIONS as f64;
        let avg_minarrow_direct_f64 =
            total_minarrow_direct_f64.as_nanos() as f64 / ITERATIONS as f64;
        let avg_arrow_struct_f64 = total_arrow_struct_f64.as_nanos() as f64 / ITERATIONS as f64;
        let avg_minarrow_enum_f64 = total_minarrow_enum_f64.as_nanos() as f64 / ITERATIONS as f64;
        let avg_arrow_dyn_f64 = total_arrow_dyn_f64.as_nanos() as f64 / ITERATIONS as f64;

        println!(
            "raw vec: Vec<i64>                             avg = {} (n={})",
            fmt_duration_ns(avg_vec_i64),
            ITERATIONS
        );
        println!(
            "minarrow direct: IntegerArray                 avg = {} (n={})",
            fmt_duration_ns(avg_minarrow_direct_i64),
            ITERATIONS
        );
        println!(
            "arrow-rs struct: Int64Array                   avg = {} (n={})",
            fmt_duration_ns(avg_arrow_struct_i64),
            ITERATIONS
        );
        println!();
        println!(
            "minarrow enum: IntegerArray                   avg = {} (n={})",
            fmt_duration_ns(avg_minarrow_enum_i64),
            ITERATIONS
        );
        println!(
            "arrow-rs dyn: Int64Array                      avg = {} (n={})",
            fmt_duration_ns(avg_arrow_dyn_i64),
            ITERATIONS
        );
        println!();
        println!(
            "raw vec: Vec<f64>                             avg = {} (n={})",
            fmt_duration_ns(avg_vec_f64),
            ITERATIONS
        );
        println!(
            "minarrow direct: FloatArray                   avg = {} (n={})",
            fmt_duration_ns(avg_minarrow_direct_f64),
            ITERATIONS
        );
        println!(
            "arrow-rs struct: Float64Array                 avg = {} (n={})",
            fmt_duration_ns(avg_arrow_struct_f64),
            ITERATIONS
        );
        println!();
        println!(
            "minarrow enum: FloatArray                     avg = {} (n={})",
            fmt_duration_ns(avg_minarrow_enum_f64),
            ITERATIONS
        );
        println!(
            "arrow-rs dyn: Float64Array                    avg = {} (n={})",
            fmt_duration_ns(avg_arrow_dyn_f64),
            ITERATIONS
        );
    }
}

#[cfg(feature = "cast_arrow")]
fn fmt_duration_ns(avg_ns: f64) -> String {
    if avg_ns < 1000.0 {
        format!("{:.0} ns", avg_ns)
    } else if avg_ns < 1_000_000.0 {
        format!("{:.3} µs", avg_ns / 1000.0)
    } else {
        format!("{:.3} ms", avg_ns / 1_000_000.0)
    }
}

fn main() {
    if cfg!(feature = "cast_arrow") {
        #[cfg(feature = "cast_arrow")]
        run_benchmark()
    } else {
        println!(
            "The hotloop_benchmark_avg_std example requires enabling the `cast_arrow` feature."
        )
    }
}
