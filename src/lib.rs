//! Copyright © 2025 Peter Garfield Bower. All rights reserved.

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
        #[cfg(feature = "slicing_extras")]
        #[cfg(feature = "chunked")]
        pub mod chunked {
            pub mod super_array_view;
            pub mod super_table_view;
        }
        #[cfg(feature = "collection_views")]
        pub mod collections {
            pub mod numeric_array_view;
            #[cfg(feature = "datetime")]
            pub mod temporal_array_view;
            pub mod text_array_view;
        }
        #[cfg(feature = "slicing_extras")]
        pub mod array_view;
        pub mod bitmask_view;

        #[cfg(feature = "slicing_extras")]
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
    #[cfg(feature = "collection_views")]
    pub mod view;
    pub mod print;
    pub mod type_unions;
    pub mod custom_value;
}

pub mod aliases;
pub mod macros;
pub mod utils;
#[cfg(feature = "typecasting")]
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
#[cfg(feature = "slicing_extras")]
#[cfg(feature = "chunked")]
pub use structs::views::chunked::{
    super_array_view::SuperArrayV, super_table_view::SuperTableV
};
#[cfg(feature = "slicing_extras")]
pub use structs::views::array_view::ArrayV;
#[cfg(feature = "collection_views")]
pub use structs::views::collections::numeric_array_view::NumericArrayV;
#[cfg(feature = "collection_views")]
#[cfg(feature = "datetime")]
pub use structs::views::collections::temporal_array_view::TemporalArrayV;
#[cfg(feature = "collection_views")]
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
#[cfg(feature = "slicing_extras")]
pub use structs::views::table_view::TableV;
pub use traits::masked_array::MaskedArray;
pub use traits::print::Print;
pub use traits::type_unions::{Float, Integer, Numeric, Primitive};
pub use ffi::arrow_dtype::ArrowType;
pub use structs::shared_buffer::SharedBuffer;
