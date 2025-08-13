//! # ArrowDType Module - *Arrow type tagging for self-documenting data*
//!
//! Unified Minarrow representations of supported *Apache Arrow* data types.
//!
//! ## Overview
//! - Covers integer, floating-point, boolean, string, dictionary-encoded, and optional temporal types  
//!   (date, time, duration, timestamp, interval).
//! - Each Minarrow array type implements `arrow_type()` to return its matching `ArrowType`.
//! - Enables consistent Arrow FFI compatibility without requiring the full Arrow type system.
//!
//! ## CategoricalIndexType
//! - Specifies the integer size of dictionary keys for categorical arrays.
//! - Supports multiple unsigned integer widths depending on feature flags.
//!
//! ## Display
//! - Human-readable type names are produced for all variants.
//! - Temporal types include their units in the rendered output.
//!
//! ## Interoperability
//! - Implements a focused subset of the public Arrow specification.
//! - Maintains compatibility while keeping Minarrow minimal.
//!
//! ## Copyright Notice
//! - The `Minarrow` crate is not affiliated with the `Apache Arrow` project.
//! - The term `Apache Arrow` is a trademark of the *Apache Software Foundation*.
//! - The term `Arrow` is used here under fair use to implement the public FFI compatibility standard,  
//!   in accordance with the official guidance: <https://www.apache.org/foundation/marks/>.
//!
//! See `./LICENSE` for more information.

use std::any::TypeId;
use std::fmt::{Display, Formatter, Result as FmtResult};

#[cfg(feature = "datetime")]
use crate::enums::time_units::{IntervalUnit, TimeUnit};
#[cfg(feature = "datetime")]
use crate::DatetimeArray;
use crate::{
    BooleanArray, CategoricalArray, Float, FloatArray, Integer,
    StringArray
};

/// # ArrowType
///
/// Unified representation of supported *Apache Arrow* data types in Minarrow.
///
/// ## Purpose
/// - Encodes the physical type and, for temporal variants, associated unit information for all supported Minarrow arrays.
/// - Provides a single discriminant used across the crate for schema definitions, type matching, and Arrow FFI export.
/// - Implements a focused subset of the official Arrow type specification:  
///   <https://arrow.apache.org/docs/python/api/datatypes.html>.
///
/// ## Coverage
/// - **Core primitives**: integer, floating-point, boolean.
/// - **Strings**: UTF-8 (`String`) and optionally large UTF-8 (`LargeString`).
/// - **Dictionary-encoded strings**: via `Dictionary(CategoricalIndexType)`.
/// - **Optional temporal types**: `date`, `time`, `duration`, `timestamp`, and `interval` with explicit units.
/// - **`Null`**: placeholder or metadata-only fields.
///
/// ## Interoperability
/// - Directly compatible with the Apache Arrow C Data Interface type descriptors.
/// - Preserves type and temporal unit information when arrays are transmitted over FFI.
/// - Simplifies Minarrow’s type system *(e.g., one `DatetimeArray` type)* while tagging `ArrowType` on `Field` for ecosystem compatibility. 
///
/// ## Notes
/// - For `DatetimeArray` types, `ArrowType` reflects only the physical encoding.  
///   Logical distinctions (e.g., interpreting a `Date64` as a timestamp vs. a duration) are stored in `Field` metadata.
/// - Dictionary key widths are defined by the associated `CategoricalIndexType`.
#[derive(PartialEq, Clone, Debug)]
pub enum ArrowType {
    Null,
    Boolean,
    #[cfg(feature = "extended_numeric_types")]
    Int8,
    #[cfg(feature = "extended_numeric_types")]
    Int16,
    Int32,
    Int64,
    #[cfg(feature = "extended_numeric_types")]
    UInt8,
    #[cfg(feature = "extended_numeric_types")]
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    #[cfg(feature = "datetime")]
    Date32,
    #[cfg(feature = "datetime")]
    Date64,
    #[cfg(feature = "datetime")]
    Time32(TimeUnit),
    #[cfg(feature = "datetime")]
    Time64(TimeUnit),
    #[cfg(feature = "datetime")]
    Duration32(TimeUnit),
    #[cfg(feature = "datetime")]
    Duration64(TimeUnit),
    #[cfg(feature = "datetime")]
    Timestamp(TimeUnit),
    #[cfg(feature = "datetime")]
    Interval(IntervalUnit),
    String,
    #[cfg(feature = "large_string")]
    LargeString,

    // Integer size for the categorical dictionary key,
    // and therefore how much storage space for each entry there is,
    // on top of the base string collection.
    Dictionary(CategoricalIndexType)
}

/// # CategoricalIndexType
///
/// Specifies the unsigned integer width used for dictionary keys in categorical arrays.
///
/// ## Overview
/// - Determines the storage size of the key column that indexes into the categorical dictionary.
/// - Smaller widths reduce memory footprint for low-cardinality data.
/// - Larger widths enable more distinct categories without overflow.
/// - Variant availability depends on feature flags:
///   - `UInt8` and `UInt16` require both `extended_categorical` and `extended_numeric_types`.
///   - `UInt64` requires `extended_categorical`.
///   - `UInt32` is always available.
///
/// ## Interoperability
/// - Maps directly to the integer index type in Apache Arrow's `DictionaryType`.
/// - Preserved when sending categorical arrays over the Arrow C Data Interface.

#[derive(PartialEq, Clone, Debug)]
pub enum CategoricalIndexType {
    #[cfg(all(feature = "extended_categorical", feature = "extended_numeric_types"))]
    UInt8,
    #[cfg(all(feature = "extended_categorical", feature = "extended_numeric_types"))]
    UInt16,
    UInt32,
    #[cfg(all(feature = "extended_categorical"))]
    UInt64
}


// Design documentation: arrow_type()
//
// Whilst `arrow_type()` could be on a trait, the ergonomics of using one aren't great
// due to then needing to import the trait at every usage point, for one method.
// Additionally, for cases like `DateTime`, the user is required to select a type when
// preparing `Field` metadata, and thus it is misleading. For this reason, they are
// here on the main objects as regular methods, so that they are available for most
// individual cases, but uniform dispatch methods that then don't work for datetime
// exceptions are implicitly discouraged. Adding it to MaskedArray as a method is also not a great
// option as the above still applies but then customising it per type and variant would
// require extra type storage on the MaskedArray trait which is too much for this.
// The other option is to use our types, rather than Arrow's for `Field`s, but that complicates
// FFI, as it's much better that once that's written it's compatible, so we settle on the below,
// which means the experience is:
// - "Field::new("myfield", existing_arr.arrow_type(), false, None)",
// - "Field::new("key", ArrowType::Date32, false, None)" when working with dates.

impl BooleanArray<()> {
    /// The arrow type that backs this array
    pub fn arrow_type() -> ArrowType {
        ArrowType::Boolean
    }
}

impl<T: Integer> CategoricalArray<T> {
    /// The arrow type that backs this array
    pub fn arrow_type() -> ArrowType {
        let t = TypeId::of::<T>();
        #[cfg(feature = "extended_categorical")]
        if t == TypeId::of::<u8>() {
            return ArrowType::Dictionary(CategoricalIndexType::UInt8)
        }
        #[cfg(feature = "extended_categorical")]
        if t == TypeId::of::<u16>() {
            return ArrowType::Dictionary(CategoricalIndexType::UInt16)
        }
        if t == TypeId::of::<u32>() {
            return ArrowType::Dictionary(CategoricalIndexType::UInt32)
        }
        #[cfg(feature = "extended_categorical")]
        if t == TypeId::of::<u64>() {
            return ArrowType::Dictionary(CategoricalIndexType::UInt64)
        }
        unsafe { std::hint::unreachable_unchecked() }
    }
}

impl<T: Float> FloatArray<T> {
    /// The arrow type that backs this array
    pub fn arrow_type() -> ArrowType {
        let t = TypeId::of::<T>();
        if t == TypeId::of::<f32>() {
            ArrowType::Float32
        } else if t == TypeId::of::<f64>() {
            ArrowType::Float64
        } else {
            unsafe { std::hint::unreachable_unchecked() }
        }
    }
}

impl<T: Integer> StringArray<T> {
    /// The arrow type that backs this array
    pub fn arrow_type() -> ArrowType {
        let t = TypeId::of::<T>();
        if t == TypeId::of::<u32>() {
            return ArrowType::String
        }
        #[cfg(feature = "large_string")]
        if t == TypeId::of::<u64>() {
            return ArrowType::LargeString
        } 
        unsafe { std::hint::unreachable_unchecked() }
    }
}

#[cfg(feature = "datetime")]
impl<T: Integer> DatetimeArray<T> {
    /// For DateTime, the logical type is undocumented until attached to the type with a `Field` via `Field::new`.
    /// At this stage, one can convert the array into a `FieldArray` which makes it immutable and hooks it into Arrow FFI-ready
    /// format. This helps enable reducing 8 separate logical *Arrow* types down to 1 `DateTimeArray` data structure, 
    /// keeping *MinArrow* minimal whilst retaining a compatibility path.
    pub fn arrow_type() -> ArrowType {
        ArrowType::Null
    }
}


impl Display for ArrowType {
    /// Render the ArrowType as its variant name, including associated units where applicable.
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            ArrowType::Null => f.write_str("Null"),
            ArrowType::Boolean => f.write_str("Boolean"),

            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int8 => f.write_str("Int8"),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int16 => f.write_str("Int16"),
            ArrowType::Int32 => f.write_str("Int32"),
            ArrowType::Int64 => f.write_str("Int64"),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt8 => f.write_str("UInt8"),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt16 => f.write_str("UInt16"),
            ArrowType::UInt32 => f.write_str("UInt32"),
            ArrowType::UInt64 => f.write_str("UInt64"),

            ArrowType::Float32 => f.write_str("Float32"),
            ArrowType::Float64 => f.write_str("Float64"),

            #[cfg(feature = "datetime")]
            ArrowType::Date32 => f.write_str("Date32"),
            #[cfg(feature = "datetime")]
            ArrowType::Date64 => f.write_str("Date64"),

            #[cfg(feature = "datetime")]
            ArrowType::Time32(unit) => write!(f, "Time32({unit})"),
            #[cfg(feature = "datetime")]
            ArrowType::Time64(unit) => write!(f, "Time64({unit})"),
            #[cfg(feature = "datetime")]
            ArrowType::Duration32(unit) => write!(f, "Duration32({unit})"),
            #[cfg(feature = "datetime")]
            ArrowType::Duration64(unit) => write!(f, "Duration64({unit})"),
            #[cfg(feature = "datetime")]
            ArrowType::Timestamp(unit) => write!(f, "Timestamp({unit})"),
            #[cfg(feature = "datetime")]
            ArrowType::Interval(interval) => write!(f, "Interval({interval})"),

            ArrowType::String => f.write_str("String"),
            #[cfg(feature = "large_string")]
            ArrowType::LargeString => f.write_str("LargeString"),

            ArrowType::Dictionary(key_type) => write!(f, "Dictionary({key_type})"),
        }
    }
}


impl Display for CategoricalIndexType {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            #[cfg(feature = "extended_categorical")]
            CategoricalIndexType::UInt8  => f.write_str("UInt8"),
            #[cfg(feature = "extended_categorical")]
            CategoricalIndexType::UInt16 => f.write_str("UInt16"),
            CategoricalIndexType::UInt32 => f.write_str("UInt32"),
            #[cfg(feature = "extended_categorical")]
            CategoricalIndexType::UInt64 => f.write_str("UInt64"),
        }
    }
}
