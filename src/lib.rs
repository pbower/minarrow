//! # Minarrow - The Latest and Greatest Rust Innovation Since One-Nanosecond Ago
//!
//! Next-generation Rust library implementing the Apache Arrow zero-copy memory layout —
//! built for high-performance computing, streaming, embedded
//! systems, and for Rust developers who demand simple, powerful, and efficient data.
//!
//! ## Standout features:
//! - Very fast compile times (< 1.5 s standard features, < 0.15 s rebuilds)
//! - Automatic 64-byte SIMD alignment
//! - Exceptional runtime performance - see benchmarks below
//! - Cohesive, well-documented API with extensive coverage
//! - FFI compatibility, including easy *to_apache_arrow()* and *to_polars()*
//!
//! ## Coming soon
//! 1. **Lightstream-IO** — IPC streaming and Tokio async integration  
//! 2. **Kernels** — Large library of SIMD-ready computation kernels  
//!
//! Both integrate seamlessly with `Minarrow`.  
//! Interested in contributing? Reach out.
//! 
//! ## Thank you
//! Thank you for using `Minarrow`. If you find it useful, share it with a friend or colleague.
//! 
//! ## Copyright
//! 
//! Copyright © 2025 Peter Garfield Bower. All rights reserved.
//! Licensed under the MIT license.
//! 
//! ## Arrow compatibility
//! This library closely follows the *Apache Arrow* format,
//! but simplifies certain APIs while maintaining strong compatibility.
//! `Minarrow` uses the documented memory layouts for all array types, but
//! consolidates logical types in some cases when they share identical physical storage
//! (e.g., `DateTimeArray`). Additional types are provided where they add value.
//! 
//! Learn more about *Apache Arrow* at: <https://arrow.apache.org/overview/>  
//! 
//! `Minarrow` is not affiliated with *Apache Arrow* or the *Apache Software Foundation*.
//! *Apache Arrow* is a registered trademark of the ASF, referenced here under
//! [fair use](https://www.apache.org/foundation/marks/).
//! 
//! ## Editions
//! Requires Rust nightly for modern, yet unstable, features such as `allocator_api`.
//!
//! ## Benchmarks
//!
//! ***Sum of 1 billion sequential integers starting at 0.***
//!
//! Intel(R) Core(TM) Ultra 7 155H | x86_64 | 22 CPUs  
//! Averaged over 1,000 runs (release).  
//!
//! ### No SIMD 
//!
//! ***(n=1000, lanes=4, iters=1000)***
//!
//! | Case                               | Avg time  |
//! |------------------------------------|-----------|
//! | **Integer (i64)**                  |           |
//! | raw vec: `Vec<i64>`                | 85 ns     |
//! | minarrow direct: `IntegerArray`    | 88 ns     |
//! | arrow-rs struct: `Int64Array`      | 147 ns    |
//! | minarrow enum: `IntegerArray`      | 124 ns    |
//! | arrow-rs dyn: `Int64Array`         | 181 ns    |
//! | **Float (f64)**                    |           |
//! | raw vec: `Vec<f64>`                | 475 ns    |
//! | minarrow direct: `FloatArray`      | 476 ns    |
//! | arrow-rs struct: `Float64Array`    | 527 ns    |
//! | minarrow enum: `FloatArray`        | 507 ns    |
//! | arrow-rs dyn: `Float64Array`       | 1.952 µs  |
//!
//! ### SIMD 
//!
//! ***(n=1000, lanes=4, iters=1000)***
//!
//! | Case                                  | Avg (ns) |
//! |---------------------------------------|---------:|
//! | raw vec: `Vec<i64>`                   | 64       |
//! | raw vec64: `Vec64<i64>`               | 55       |
//! | minarrow direct: `IntegerArray`       | 88       |
//! | arrow-rs struct: `Int64Array`         | 162      |
//! | minarrow enum: `IntegerArray`         | 170      |
//! | arrow-rs dyn: `Int64Array`            | 173      |
//! | raw vec: `Vec<f64>`                   | 57       |
//! | raw vec64: `Vec64<f64>`               | 58       |
//! | minarrow direct: `FloatArray`         | 91       |
//! | arrow-rs struct: `Float64Array`       | 181      |
//! | minarrow enum: `FloatArray`           | 180      |
//! | arrow-rs dyn: `Float64Array`          | 196      |
//!
//! ### SIMD + Rayon
//!
//! ***(n=1000, lanes=4, iters=1000)***
//!
//! | Case                                    | Time (ms)   |
//! |-----------------------------------------|-------------|
//! | SIMD + Rayon `IntegerArray<i64>`        | 113.874     |
//! | SIMD + Rayon `FloatArray<f64>`          | 114.095     |
//!
//! ### Other benchmark factors
//! Vec<i64> construction (generating + allocating 1000 elements - avg): 87 ns  
//! Vec64<i64> construction (avg): 84 ns  
//!
//! _The construction delta is not included in the benchmark timings above._

#![feature(allocator_api)]
#![feature(slice_ptr_get)]

pub mod enums {
    pub mod time_units;
    #[cfg(feature = "scalar_type")]
    pub mod scalar;
    #[cfg(feature = "value_type")]
    pub mod value;
    pub mod array;
    pub mod error;
    pub mod collections {
        pub mod numeric_array;
        pub mod text_array;
        #[cfg(feature = "datetime")]
        pub mod temporal_array;
    }
}

pub mod structs {

    #[cfg(feature = "chunked")]
    pub mod chunked {
        pub mod super_array;
        pub mod super_table;
    }

    pub mod variants {
        pub mod boolean;
        pub mod categorical;
        #[cfg(feature = "datetime")]
        pub mod datetime;
        pub mod float;
        pub mod integer;
        pub mod string;
    }
    pub mod views {
        #[cfg(feature = "views")]
        #[cfg(feature = "chunked")]
        pub mod chunked {
            pub mod super_array_view;
            pub mod super_table_view;
        }
        #[cfg(feature = "views")]
        pub mod collections {
            pub mod numeric_array_view;
            #[cfg(feature = "datetime")]
            pub mod temporal_array_view;
            pub mod text_array_view;
        }
        #[cfg(feature = "views")]
        pub mod array_view;
        pub mod bitmask_view;

        #[cfg(feature = "views")]
        pub mod table_view;
    }
    pub mod buffer;
    pub mod shared_buffer;
    pub mod allocator;
    pub mod bitmask;
    pub mod field;
    pub mod field_array;
    pub mod table;
    #[cfg(feature = "cube")]
    pub mod cube;
    #[cfg(feature = "matrix")]
    pub mod matrix;
    pub mod vec64;
}

pub mod ffi {
    pub mod arrow_c_ffi;
    pub mod arrow_dtype;
    pub mod schema;
}
    
pub mod traits {
    pub mod masked_array;
    #[cfg(feature = "views")]
    pub mod view;
    pub mod print;
    pub mod type_unions;
    pub mod custom_value;
}

pub mod aliases;
pub mod macros;
pub mod utils;
pub mod conversions;

pub use aliases::{
    BytesLength, 
    DictLength, Length,
    Offset, ArrayVT, BitmaskVT, StringAVT, StringAVTExt,
    CategoricalAVT, CategoricalAVTExt, IntegerAVT, FloatAVT,
    BooleanAVT
};

#[cfg(feature = "datetime")]
pub use aliases::DatetimeAVT;
#[cfg(feature = "datetime")]
pub use enums::time_units::{IntervalUnit, TimeUnit};
#[cfg(feature = "value_type")]
pub use enums::value::Value;
#[cfg(feature = "scalar_type")]
pub use enums::scalar::Scalar;
pub use enums::array::Array;
pub use enums::collections::numeric_array::NumericArray;
#[cfg(feature = "datetime")]
pub use enums::collections::temporal_array::TemporalArray;
pub use enums::collections::text_array::TextArray;

pub use structs::buffer::Buffer;
pub use structs::bitmask::Bitmask;
pub use structs::views::bitmask_view::BitmaskV;
#[cfg(feature = "chunked")]
pub use structs::chunked::{super_array::SuperArray, super_table::SuperTable};
#[cfg(feature = "views")]
#[cfg(feature = "chunked")]
pub use structs::views::chunked::{
    super_array_view::SuperArrayV, super_table_view::SuperTableV
};
#[cfg(feature = "views")]
pub use structs::views::array_view::ArrayV;
#[cfg(feature = "views")]
pub use structs::views::collections::numeric_array_view::NumericArrayV;
#[cfg(feature = "views")]
#[cfg(feature = "datetime")]
pub use structs::views::collections::temporal_array_view::TemporalArrayV;
#[cfg(feature = "views")]
pub use structs::views::collections::text_array_view::TextArrayV;

pub use structs::field::Field;
pub use structs::field_array::FieldArray;
pub use structs::table::Table;
#[cfg(feature = "cube")]
pub use structs::cube::Cube;
#[cfg(feature = "matrix")]
pub use structs::matrix::Matrix;
pub use structs::variants::boolean::BooleanArray;
pub use structs::variants::categorical::CategoricalArray;
#[cfg(feature = "datetime")]
pub use structs::variants::datetime::DatetimeArray;
pub use structs::variants::float::FloatArray;
pub use structs::variants::integer::IntegerArray;
pub use structs::variants::string::StringArray;
pub use structs::vec64::Vec64;
#[cfg(feature = "views")]
pub use structs::views::table_view::TableV;
pub use traits::masked_array::MaskedArray;
pub use traits::print::Print;
pub use traits::type_unions::{Float, Integer, Numeric, Primitive};
pub use ffi::arrow_dtype::ArrowType;
pub use structs::shared_buffer::SharedBuffer;
