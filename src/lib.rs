//! # **Minarrow** – High-Performance Rust with Apache Arrow Compatibility
//!
//! Modern Rust implementation of the Apache Arrow zero-copy memory layout,
//! for high-performance computing, streaming, and embedded systems.
//! Built for those who like it fast and simple.
//!
//! ## Key Features
//! - **Fast compile times** – typically <1.5s for standard builds, <0.15s for rebuilds.
//! - **64-byte SIMD alignment** for optimal CPU utilisation.
//! - **High runtime performance** – see benchmarks below.
//! - Cohesive, well-documented API with extensive coverage.
//! - Built-in FFI with simple `to_apache_arrow()` and `to_polars()` conversions.
//! - MIT Licensed.
//!
//! ## Upcoming Additions
//! 1. **Lightstream-IO** – IPC streaming and Tokio async integration.  
//! 2. **SIMD Kernels** – Large library of pre-optimised computation kernels.  
//!
//! ## Compatibility
//! Implements Apache Arrow’s documented memory layouts while simplifying some APIs.
//! Additional logical types are provided where they add practical value.
//! Learn more about Apache Arrow at: <https://arrow.apache.org/overview/>.  
//!
//! Minarrow is not affiliated with Apache Arrow or the Apache Software Foundation.
//! *Apache Arrow* is a registered trademark of the ASF, referenced under fair use.
//!
//! ## Acknowledgements
//! Thanks to the Apache Arrow community and contributors, with inspiration
//! from `Arrow2` and `Polars`.
//!
//! ## Requirements
//! Requires Rust nightly for features such as `allocator_api`.
//!
//! ## Benchmarks
//!
//! **Intel(R) Core(TM) Ultra 7 155H | x86_64 | 22 CPUs**  
//!
//! ### No SIMD
//! ***(n=1000, lanes=4, iters=1000)***
//!
//! | Case                            | Avg time |
//! |---------------------------------|----------|
//! | Vec<i64>                        | 85 ns    |
//! | Minarrow direct IntegerArray    | 88 ns    |
//! | arrow-rs struct Int64Array      | 147 ns   |
//! | Minarrow enum IntegerArray      | 124 ns   |
//! | arrow-rs dyn Int64Array         | 181 ns   |
//! | Vec<f64>                        | 475 ns   |
//! | Minarrow direct FloatArray      | 476 ns   |
//! | arrow-rs struct Float64Array    | 527 ns   |
//! | Minarrow enum FloatArray        | 507 ns   |
//! | arrow-rs dyn Float64Array       | 1.952 µs |
//!
//! ### SIMD
//! ***(n=1000, lanes=4, iters=1000)***
//!
//! | Case                            | Avg time |
//! |---------------------------------|----------|
//! | Vec<i64>                        | 64 ns    |
//! | Vec64<i64>                      | 55 ns    |
//! | Minarrow direct IntegerArray    | 88 ns    |
//! | arrow-rs struct Int64Array      | 162 ns   |
//! | Minarrow enum IntegerArray      | 170 ns   |
//! | arrow-rs dyn Int64Array         | 173 ns   |
//! | Vec<f64>                        | 57 ns    |
//! | Vec64<f64>                      | 58 ns    |
//! | Minarrow direct FloatArray      | 91 ns    |
//! | arrow-rs struct Float64Array    | 181 ns   |
//! | Minarrow enum FloatArray        | 180 ns   |
//! | arrow-rs dyn Float64Array       | 196 ns   |
//!
//! ### SIMD + Rayon
//! ***(n=1,000,000,000, lanes=4)***
//!
//! | Case                              | Time (ms) |
//! |-----------------------------------|-----------|
//! | SIMD + Rayon IntegerArray<i64>    | 113.874   |
//! | SIMD + Rayon FloatArray<f64>      | 114.095   |
//!
//! _Construction time for Vec<i64> (87 ns) and Vec64<i64> (84 ns) excluded from benchmarks._

#![feature(allocator_api)]
#![feature(slice_ptr_get)]
#![feature(portable_simd)]

pub use vec64::{Vec64, vec64};

/// **Array**, **TextArray**, **NumericArray**...- *All the *High-Level Array containers* are here.*
pub mod enums {
    pub mod array;
    pub mod error;
    #[cfg(feature = "scalar_type")]
    pub mod scalar;
    pub mod time_units;
    #[cfg(feature = "value_type")]
    pub mod value;
    pub mod collections {
        pub mod numeric_array;
        #[cfg(feature = "datetime")]
        pub mod temporal_array;
        pub mod text_array;
    }
    pub mod operators;
    pub mod shape_dim;
}

/// Contains SIMD-accelerated kernels for the 'essentials' that are highly coupled to this crate
/// The extensive set is available downstream in the simd-kernels crate, including a full
/// set of univariate distributions.
pub mod kernels {
    pub mod arithmetic;
    pub mod bitmask;
    #[cfg(feature = "broadcast")]
    pub mod broadcast;
    #[cfg(feature = "views")]
    pub mod routing;
    pub mod string;
}

/// **Table**, **IntegerArray**, **FloatArray**, **Vec64** - *All the **Low-Level Control**, **Tables** and **Views***.
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
    pub mod bitmask;
    pub mod buffer;
    #[cfg(feature = "cube")]
    pub mod cube;
    pub mod field;
    pub mod field_array;
    #[cfg(feature = "matrix")]
    pub mod matrix;
    pub mod shared_buffer;
    pub mod table;
}

/// **Shared Memory** - *Sending data over FFI like a Pro? Look here.*
pub mod ffi {
    pub mod arrow_c_ffi;
    pub mod arrow_dtype;
    pub mod schema;
}

/// **Type Standardisation** - `MaskedArray`, `View`, `Print` traits + more,
pub mod traits {
    #[cfg(feature = "size")]
    pub mod byte_size;
    pub mod concatenate;
    pub mod custom_value;
    pub mod masked_array;
    pub mod print;
    pub mod shape;
    pub mod type_unions;
    #[cfg(feature = "views")]
    pub mod view;
}

pub mod aliases;
pub mod conversions;
pub mod macros;
pub mod utils;

pub use aliases::{
    ArrayVT, BitmaskVT, BooleanAVT, BytesLength, CategoricalAVT, CategoricalAVTExt, DictLength,
    FloatAVT, IntegerAVT, Length, Offset, StringAVT, StringAVTExt,
};

#[cfg(feature = "datetime")]
pub use aliases::DatetimeAVT;
pub use enums::array::Array;
pub use enums::collections::numeric_array::NumericArray;
#[cfg(feature = "datetime")]
pub use enums::collections::temporal_array::TemporalArray;
pub use enums::collections::text_array::TextArray;
#[cfg(feature = "scalar_type")]
pub use enums::scalar::Scalar;
#[cfg(feature = "datetime")]
pub use enums::time_units::{IntervalUnit, TimeUnit};
#[cfg(feature = "value_type")]
pub use enums::value::Value;

pub use structs::bitmask::Bitmask;
pub use structs::buffer::Buffer;
#[cfg(feature = "chunked")]
pub use structs::chunked::{super_array::SuperArray, super_table::SuperTable};
#[cfg(feature = "views")]
pub use structs::views::array_view::ArrayV;
pub use structs::views::bitmask_view::BitmaskV;
#[cfg(feature = "views")]
#[cfg(feature = "chunked")]
pub use structs::views::chunked::{super_array_view::SuperArrayV, super_table_view::SuperTableV};
#[cfg(feature = "views")]
pub use structs::views::collections::numeric_array_view::NumericArrayV;
#[cfg(feature = "views")]
#[cfg(feature = "datetime")]
pub use structs::views::collections::temporal_array_view::TemporalArrayV;
#[cfg(feature = "views")]
pub use structs::views::collections::text_array_view::TextArrayV;

pub use ffi::arrow_dtype::ArrowType;
#[cfg(feature = "cube")]
pub use structs::cube::Cube;
pub use structs::field::Field;
pub use structs::field_array::FieldArray;
#[cfg(feature = "matrix")]
pub use structs::matrix::Matrix;
pub use structs::shared_buffer::SharedBuffer;
pub use structs::table::Table;
pub use structs::variants::boolean::BooleanArray;
pub use structs::variants::categorical::CategoricalArray;
#[cfg(feature = "datetime")]
pub use structs::variants::datetime::DatetimeArray;
pub use structs::variants::float::FloatArray;
pub use structs::variants::integer::IntegerArray;
pub use structs::variants::string::StringArray;
#[cfg(feature = "views")]
pub use structs::views::table_view::TableV;
#[cfg(feature = "size")]
pub use traits::byte_size::ByteSize;
pub use traits::concatenate::Concatenate;
pub use traits::masked_array::MaskedArray;
pub use traits::print::Print;
pub use traits::type_unions::{Float, Integer, Numeric, Primitive};
