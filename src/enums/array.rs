//! # **Array Module** - *Main High-Level Array Type*
//! 
//! `Array` is the primary unified container for all array types in Minarrow.
//!   
//! ## Features:
//! - direct variant access to numeric, temporal, text, and other array categories
//! - zero-cost casts when the contained type is known
//! - lossless conversions between compatible array types
//! - simplifies function signatures by allowing `impl Into<Array>`
//! - centralises dispatch for all array operations
//! - preserves SIMD-aligned buffers and metadata across variants.

use std::any::TypeId;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

#[cfg(feature = "cast_arrow")]
use crate::ffi::arrow_c_ffi::export_to_c;
#[cfg(feature = "cast_arrow")]
use crate::ffi::schema::Schema;
#[cfg(feature = "cast_arrow")]
use arrow::array::{ArrayRef, make_array};
#[cfg(feature = "cast_arrow")]
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
#[cfg(any(feature = "cast_arrow", feature = "cast_polars"))]
use crate::Field;

#[cfg(feature = "views")]
use crate::ArrayV;
#[cfg(feature = "views")]
use crate::ArrayVT;
#[cfg(feature = "datetime")]
use crate::DatetimeArray;
#[cfg(feature = "datetime")]
use crate::TemporalArray;
use crate::ffi::arrow_dtype::{ArrowType, CategoricalIndexType};
use crate::utils::{float_to_text_array, int_to_text_array};
use crate::{
    Bitmask, BooleanArray, CategoricalArray, FloatArray, IntegerArray, MaskedArray,
    NumericArray, StringArray, TextArray, Vec64, match_array,
};

/// # Array
/// 
/// Standard `Array` type. Wrap in a `FieldArray` when using inside a `Table`
/// or as a standalone value requiring tagged metadata.
///
/// ## Overview
/// The dual-enum approach may look verbose but works well in practice:
///
/// - Enables clean function signatures with direct access to concrete types
///   (e.g. `&NumericArray`), supporting trait-aligned dispatch without
///   exhaustive matches at every call site.
/// - Supports ergonomic categorisation: functions typically match on the
///   outer enum for broad category handling *(numeric, text, temporal, boolean)*,
///   while allowing inner variant matching for precise type handling.
/// - The focused typeset (no nested types) helps keeps enum size efficient 
///   as memory is allocated for the largest variant.
/// 
/// ## Usage
/// Functions can accept references tailored to the intended match granularity:
///
/// - `&IntegerArray`: direct reference to the inner type e.g., `arr.num().i64()`.
/// - `&NumericArray`: any numeric type via `arr.num()`.
/// - `&Array`: match on categories or individual types.
/// 
/// ## Benefits
/// - No heap allocation or runtime indirection — all enum variants are inline
///   with minimal discriminant cost.
/// - Unified call sites with compiler-enforced type safety.
/// - Easy casting to inner types (e.g., `.str()` for strings).
/// - Supports aggressive compiler inlining, unlike approaches relying on
///   dynamic dispatch and downcasting.
/// 
/// ## Trade-offs
/// - Adds ~30–100 ns latency compared to direct inner type calls — only
///   noticeable in extreme low-latency contexts such as HFT.
/// - Requires enum matching at dispatch sites compared to direct inner type usage.

/// ## Examples
/// ```rust
/// use minarrow::{
///     Array, IntegerArray, NumericArray, arr_bool, arr_cat32, arr_f64, arr_i32, arr_i64,
///     arr_str32, vec64
/// };
///
/// // Fast macro construction
/// let int_arr = arr_i32![1, 2, 3, 4];
/// let float_arr = arr_f64![0.5, 1.5, 2.5];
/// let bool_arr = arr_bool![true, false, true];
/// let str_arr = arr_str32!["a", "b", "c"];
/// let cat_arr = arr_cat32!["x", "y", "x", "z"];
///
/// assert_eq!(int_arr.len(), 4);
/// assert_eq!(str_arr.len(), 3);
///
/// // Manual construction
/// let int = IntegerArray::<i64>::from_slice(&[100, 200]);
/// let wrapped: NumericArray = NumericArray::Int64(std::sync::Arc::new(int));
/// let array = Array::NumericArray(wrapped);
/// ```
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub enum Array {
    NumericArray(NumericArray),
    TextArray(TextArray),
    #[cfg(feature = "datetime")]
    TemporalArray(TemporalArray),
    BooleanArray(Arc<BooleanArray<()>>),
    #[default]
    Null, // Default Marker for mem::take
}

impl Array {
    /// Creates an Array enum with an Int8 array.
    #[cfg(feature = "extended_numeric_types")]
    pub fn from_int8(arr: IntegerArray<i8>) -> Self {
        Array::NumericArray(NumericArray::Int8(Arc::new(arr)))
    }

    /// Creates an Array enum with an UInt8 array.
    #[cfg(feature = "extended_numeric_types")]
    pub fn from_uint8(arr: IntegerArray<u8>) -> Self {
        Array::NumericArray(NumericArray::UInt8(Arc::new(arr)))
    }

    /// Creates an Array enum with an Int16 array.
    #[cfg(feature = "extended_numeric_types")]
    pub fn from_int16(arr: IntegerArray<i16>) -> Self {
        Array::NumericArray(NumericArray::Int16(Arc::new(arr)))
    }

    /// Creates an Array enum with an UInt16 array.
    #[cfg(feature = "extended_numeric_types")]
    pub fn from_uint16(arr: IntegerArray<u16>) -> Self {
        Array::NumericArray(NumericArray::UInt16(Arc::new(arr)))
    }

    /// Creates an Array enum with an Int32 array.
    pub fn from_int32(arr: IntegerArray<i32>) -> Self {
        Array::NumericArray(NumericArray::Int32(Arc::new(arr)))
    }

    /// Creates an Array enum with an Int64 array.
    pub fn from_int64(arr: IntegerArray<i64>) -> Self {
        Array::NumericArray(NumericArray::Int64(Arc::new(arr)))
    }

    /// Creates an Array enum with a UInt32 array.
    pub fn from_uint32(arr: IntegerArray<u32>) -> Self {
        Array::NumericArray(NumericArray::UInt32(Arc::new(arr)))
    }

    /// Creates an Array enum with an UInt64 array.
    pub fn from_uint64(arr: IntegerArray<u64>) -> Self {
        Array::NumericArray(NumericArray::UInt64(Arc::new(arr)))
    }

    /// Creates an Array enum with a Float32 array.
    pub fn from_float32(arr: FloatArray<f32>) -> Self {
        Array::NumericArray(NumericArray::Float32(Arc::new(arr)))
    }

    /// Creates an Array enum with a Float64 array.
    pub fn from_float64(arr: FloatArray<f64>) -> Self {
        Array::NumericArray(NumericArray::Float64(Arc::new(arr)))
    }

    /// Creates an Array enum with a String32 array.
    pub fn from_string32(arr: StringArray<u32>) -> Self {
        Array::TextArray(TextArray::String32(Arc::new(arr)))
    }

    /// Creates an Array enum with a String64 array.
    #[cfg(feature = "large_string")]
    pub fn from_string64(arr: StringArray<u64>) -> Self {
        Array::TextArray(TextArray::String64(Arc::new(arr)))
    }

    /// Creates an Array enum with a Categorical32 array.
    pub fn from_categorical32(arr: CategoricalArray<u32>) -> Self {
        Array::TextArray(TextArray::Categorical32(Arc::new(arr)))
    }

    /// Creates an Array enum with a Categorical8 array.
    #[cfg(feature = "extended_categorical")]
    pub fn from_categorical8(arr: CategoricalArray<u8>) -> Self {
        Array::TextArray(TextArray::Categorical8(Arc::new(arr)))
    }

    /// Creates an Array enum with a Categorical16 array.
    #[cfg(feature = "extended_categorical")]
    pub fn from_categorical16(arr: CategoricalArray<u16>) -> Self {
        Array::TextArray(TextArray::Categorical16(Arc::new(arr)))
    }

    /// Creates an Array enum with a Categorical64 array.
    #[cfg(feature = "extended_categorical")]
    pub fn from_categorical64(arr: CategoricalArray<u64>) -> Self {
        Array::TextArray(TextArray::Categorical64(Arc::new(arr)))
    }

    /// Creates an Array enum with a DatetimeI32 array.
    #[cfg(feature = "datetime")]
    pub fn from_datetime_i32(arr: DatetimeArray<i32>) -> Self {
        Array::TemporalArray(TemporalArray::Datetime32(Arc::new(arr)))
    }

    /// Creates an Array enum with a DatetimeI64 array.
    #[cfg(feature = "datetime")]
    pub fn from_datetime_i64(arr: DatetimeArray<i64>) -> Self {
        Array::TemporalArray(TemporalArray::Datetime64(Arc::new(arr)))
    }

    /// Creates an Array enum with a Boolean array.
    pub fn from_bool(arr: BooleanArray<()>) -> Self {
        Array::BooleanArray(Arc::new(arr))
    }

    // The below provides common accessors that reformat the data into the given type.
    // Because this library leans on enums, it makes for essential ergonomics once operating
    // in the top layer and one needs to match for e.g., to `T: Numeric` etc., as one can
    // can then go `.num()` to get access to all the numerical methods. This avoids polluting the top-level
    // `Array` API with method signatures that would otherwise panic for unsupported variants and flood IDE intellisense.
    // Additionally, when binding to Python, it follows common semantics.
    // I.e., '.dt` for datetime methods to appear, `.str` for strings, etc.
    //
    // Each accessor provides zero-copy for the already native type(s), conversion paths
    // for non-native (e.g., *bool -> integer* ), whilst propagating nulls for rarer nonsensical casts.

    /// Returns an inner `NumericArray`, consuming self.
    /// - If already a `NumericArray`, consumes and returns the inner value with no clone.
    /// - Other types: casts and copies.
    pub fn num(self) -> NumericArray {
        match self {
            Array::NumericArray(arr) => arr,

            Array::BooleanArray(mut arr) => {
                // If the Arc<BooleanArray> is not unique, clone here
                let arr_mut = Arc::make_mut(&mut arr);
                let mut out = Vec64::with_capacity(arr_mut.len);
                for i in 0..arr_mut.len {
                    let v = match arr_mut.get(i) {
                        Some(true) => 1,
                        Some(false) => 0,
                        None => 0,
                    };
                    out.push(v);
                }
                let null_mask = arr_mut.null_mask.take();
                NumericArray::Int32(Arc::new(IntegerArray::<i32>::from_vec64(out, null_mask)))
            }

            #[cfg(feature = "datetime")]
            Array::TemporalArray(arr) => match arr {
                TemporalArray::Datetime32(mut arc_dt) => {
                    let dt = Arc::make_mut(&mut arc_dt);
                    let data = Vec64::from_slice(&dt.data);
                    let null_mask = dt.null_mask.take();
                    NumericArray::Int32(Arc::new(IntegerArray::<i32>::from_vec64(data, null_mask)))
                }
                TemporalArray::Datetime64(mut arc_dt) => {
                    let dt = Arc::make_mut(&mut arc_dt);
                    let data = Vec64::from_slice(&dt.data);
                    let null_mask = dt.null_mask.take();
                    NumericArray::Int64(Arc::new(IntegerArray::<i64>::from_vec64(data, null_mask)))
                }
                TemporalArray::Null => NumericArray::Null,
            },

            Array::TextArray(arr) => match arr {
                TextArray::String32(s) => {
                    let len = s.len();
                    let mut out = Vec64::with_capacity(len);
                    let mut null_mask = Bitmask::with_capacity(len);
                    for i in 0..len {
                        if s.is_null(i) {
                            out.push(0);
                            null_mask.set(i, false);
                            continue;
                        }
                        let raw = match s.get_str(i) {
                            Some(val) => val,
                            None => {
                                out.push(0);
                                null_mask.set(i, false);
                                continue;
                            }
                        };
                        match raw.parse::<i32>() {
                            Ok(val) => {
                                out.push(val);
                                null_mask.set(i, true);
                            }
                            Err(_) => {
                                out.push(0);
                                null_mask.set(i, false);
                            }
                        }
                    }
                    NumericArray::Int32(Arc::new(IntegerArray::<i32>::from_vec64(
                        out,
                        Some(null_mask),
                    )))
                }

                #[cfg(feature = "large_string")]
                TextArray::String64(s) => {
                    let len = s.len();
                    let mut out = Vec64::with_capacity(len);
                    let mut null_mask = Bitmask::with_capacity(len);
                    for i in 0..len {
                        if s.is_null(i) {
                            out.push(0);
                            null_mask.set(i, false);
                            continue;
                        }
                        let raw = match s.get_str(i) {
                            Some(val) => val,
                            None => {
                                out.push(0);
                                null_mask.set(i, false);
                                continue;
                            }
                        };
                        match raw.parse::<i64>() {
                            Ok(val) => {
                                out.push(val);
                                null_mask.set(i, true);
                            }
                            Err(_) => {
                                out.push(0);
                                null_mask.set(i, false);
                            }
                        }
                    }
                    NumericArray::Int64(Arc::new(IntegerArray::<i64>::from_vec64(
                        out,
                        Some(null_mask),
                    )))
                }

                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical8(cat) => {
                    let mut out = Vec64::with_capacity(cat.len());
                    let mut mask = Bitmask::with_capacity(cat.len());
                    for i in 0..cat.len() {
                        if cat.is_null(i) {
                            out.push(0);
                            mask.set(i, false);
                        } else {
                            let idx = cat.data[i] as usize;
                            let raw = &cat.unique_values[idx];
                            match raw.parse::<i32>() {
                                Ok(val) => {
                                    out.push(val);
                                    mask.set(i, true);
                                }
                                Err(_) => {
                                    out.push(0);
                                    mask.set(i, false);
                                }
                            }
                        }
                    }
                    NumericArray::Int32(Arc::new(IntegerArray::<i32>::from_vec64(out, Some(mask))))
                }

                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(cat) => {
                    let mut out = Vec64::with_capacity(cat.len());
                    let mut mask = Bitmask::with_capacity(cat.len());
                    for i in 0..cat.len() {
                        if cat.is_null(i) {
                            out.push(0);
                            mask.set(i, false);
                        } else {
                            let idx = cat.data[i] as usize;
                            let raw = &cat.unique_values[idx];
                            match raw.parse::<i32>() {
                                Ok(val) => {
                                    out.push(val);
                                    mask.set(i, true);
                                }
                                Err(_) => {
                                    out.push(0);
                                    mask.set(i, false);
                                }
                            }
                        }
                    }
                    NumericArray::Int32(Arc::new(IntegerArray::<i32>::from_vec64(out, Some(mask))))
                }

                TextArray::Categorical32(cat) => {
                    let mut out = Vec64::with_capacity(cat.len());
                    let mut mask = Bitmask::with_capacity(cat.len());
                    for i in 0..cat.len() {
                        if cat.is_null(i) {
                            out.push(0);
                            mask.set(i, false);
                        } else {
                            let idx = cat.data[i] as usize;
                            let raw = &cat.unique_values[idx];
                            match raw.parse::<i32>() {
                                Ok(val) => {
                                    out.push(val);
                                    mask.set(i, true);
                                }
                                Err(_) => {
                                    out.push(0);
                                    mask.set(i, false);
                                }
                            }
                        }
                    }
                    NumericArray::Int32(Arc::new(IntegerArray::<i32>::from_vec64(out, Some(mask))))
                }

                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(cat) => {
                    let mut out = Vec64::with_capacity(cat.len());
                    let mut mask = Bitmask::with_capacity(cat.len());
                    for i in 0..cat.len() {
                        if cat.is_null(i) {
                            out.push(0);
                            mask.set(i, false);
                        } else {
                            let idx = cat.data[i] as usize;
                            let raw = &cat.unique_values[idx];
                            match raw.parse::<i64>() {
                                Ok(val) => {
                                    out.push(val);
                                    mask.set(i, true);
                                }
                                Err(_) => {
                                    out.push(0);
                                    mask.set(i, false);
                                }
                            }
                        }
                    }
                    NumericArray::Int64(Arc::new(IntegerArray::<i64>::from_vec64(out, Some(mask))))
                }

                TextArray::Null => NumericArray::Null,
            },

            Array::Null => NumericArray::Null,
        }
    }

    /// Returns an inner `TextArray`, consuming self.
    /// - If already a `TextArray`, consumes and returns the inner value with no clone.
    /// - Other types: casts *(to string)* and copies.
    pub fn str(self) -> TextArray {
        match self {
            Array::TextArray(arr) => arr,

            Array::BooleanArray(arr) => {
                let mut strings: Vec<String> = Vec::with_capacity(arr.len);
                for i in 0..arr.len {
                    match arr.get(i) {
                        Some(true) => strings.push("true".to_string()),
                        Some(false) => strings.push("false".to_string()),
                        None => strings.push(String::new()),
                    }
                }
                TextArray::String32(Arc::new(StringArray::<u32>::from_slice(
                    &strings.iter().map(String::as_str).collect::<Vec<_>>(),
                )))
            }

            Array::NumericArray(arr) => match arr {
                NumericArray::Int32(a) => int_to_text_array::<i32>(&a),
                NumericArray::Int64(a) => int_to_text_array::<i64>(&a),
                NumericArray::UInt32(a) => int_to_text_array::<u32>(&a),
                NumericArray::UInt64(a) => int_to_text_array::<u64>(&a),
                NumericArray::Float32(a) => float_to_text_array::<f32>(&a),
                NumericArray::Float64(a) => float_to_text_array::<f64>(&a),
                _ => TextArray::Null,
            },

            #[cfg(feature = "datetime")]
            Array::TemporalArray(arr) => match arr {
                TemporalArray::Datetime32(mut arc_dt) => {
                    let dt = Arc::make_mut(&mut arc_dt);
                    let mut strings = Vec::with_capacity(dt.len());
                    for i in 0..dt.len() {
                        if dt.is_null(i) {
                            strings.push(String::new());
                        } else {
                            strings.push(format!("{}", dt.data[i]));
                        }
                    }
                    TextArray::String32(Arc::new(StringArray::<u32>::from_slice(
                        &strings.iter().map(String::as_str).collect::<Vec<_>>(),
                    )))
                }
                TemporalArray::Datetime64(mut arc_dt) => {
                    let dt = Arc::make_mut(&mut arc_dt);
                    let mut strings = Vec::with_capacity(dt.len());
                    for i in 0..dt.len() {
                        if dt.is_null(i) {
                            strings.push(String::new());
                        } else {
                            strings.push(format!("{}", dt.data[i]));
                        }
                    }
                    TextArray::String32(Arc::new(StringArray::<u32>::from_slice(
                        &strings.iter().map(String::as_str).collect::<Vec<_>>(),
                    )))
                }
                _ => TextArray::Null,
            },

            Array::Null => TextArray::Null,
        }
    }

    /// Returns the inner `BooleanArray`, consuming self.
    /// - If already a `BooleanArray`, consumes and returns the inner value with no clone.
    /// - Other types: calculates the boolean mask based on whether the value is present, and non-zero,
    ///  and copies. In these cases, any null mask is preserved, rather than becoming `false`.
    pub fn bool(self) -> Arc<BooleanArray<()>> {
        match self {
            Array::BooleanArray(arr) => arr,
            Array::NumericArray(arr) => {
                macro_rules! to_bool {
                    ($a:expr, $t:ty) => {{
                        let mut bm = Bitmask::with_capacity($a.len());
                        let mut out = Bitmask::with_capacity($a.len());
                        for i in 0..$a.len() {
                            let valid = !$a.is_null(i);
                            bm.set(i, valid);
                            let v = if valid && $a.data[i] != <$t>::default() {
                                true
                            } else {
                                false
                            };
                            out.set(i, v);
                        }
                        BooleanArray {
                            data: out,
                            null_mask: Some(bm),
                            len: $a.len(),
                            _phantom: std::marker::PhantomData,
                        }
                        .into()
                    }};
                }
                match arr {
                    NumericArray::Int32(a) => to_bool!(a, i32),
                    NumericArray::Int64(a) => to_bool!(a, i64),
                    NumericArray::UInt32(a) => to_bool!(a, u32),
                    NumericArray::UInt64(a) => to_bool!(a, u64),
                    NumericArray::Float32(a) => to_bool!(a, f32),
                    NumericArray::Float64(a) => to_bool!(a, f64),
                    _ => BooleanArray::default().into(),
                }
            }
            #[cfg(feature = "datetime")]
            Array::TemporalArray(arr) => match arr {
                TemporalArray::Datetime32(a) => {
                    let mut bm = Bitmask::with_capacity(a.len());
                    let mut out = Bitmask::with_capacity(a.len());
                    for i in 0..a.len() {
                        let valid = !a.is_null(i);
                        bm.set(i, valid);
                        out.set(i, valid);
                    }
                    BooleanArray {
                        data: out,
                        null_mask: Some(bm),
                        len: a.len(),
                        _phantom: std::marker::PhantomData,
                    }
                    .into()
                }
                TemporalArray::Datetime64(a) => {
                    let mut bm = Bitmask::with_capacity(a.len());
                    let mut out = Bitmask::with_capacity(a.len());
                    for i in 0..a.len() {
                        let valid = !a.is_null(i);
                        bm.set(i, valid);
                        out.set(i, valid);
                    }
                    BooleanArray {
                        data: out,
                        null_mask: Some(bm),
                        len: a.len(),
                        _phantom: std::marker::PhantomData,
                    }
                    .into()
                }
                _ => BooleanArray::default().into(),
            },
            Array::TextArray(arr) => match arr {
                TextArray::String32(s) => {
                    let mut bm = Bitmask::with_capacity(s.len());
                    let mut out = Bitmask::with_capacity(s.len());
                    for i in 0..s.len() {
                        let valid = !s.is_null(i);
                        bm.set(i, valid);

                        let str_val = if valid { s.get_str(i).unwrap() } else { "" };
                        let true_val = !str_val.eq_ignore_ascii_case("0")
                            && !str_val.eq_ignore_ascii_case("false")
                            && !str_val.eq_ignore_ascii_case("f")
                            && !str_val.is_empty();
                        out.set(i, if str_val.is_empty() { false } else { true_val });
                    }
                    BooleanArray {
                        data: out,
                        null_mask: Some(bm),
                        len: s.len(),
                        _phantom: std::marker::PhantomData,
                    }
                    .into()
                }
                _ => BooleanArray::default().into(),
            },
            Array::Null => BooleanArray::default().into(),
        }
    }

    /// Returns the inner `TemporalArray`, consuming self.
    /// - If already a `TemporalArray`, consumes and returns the inner value with no clone.
    /// - Other types: casts and (often) copies using clone on write.
    ///
    /// ### Datetime conversions
    /// - **String** parses a timestamp in milliseconds since the Unix epoch.
    /// If the `chrono` feature is on, it also attempts common ISO8601/RFC3339 and `%Y-%m-%d` formats.
    /// Keep this in mind, because your API will break if you toggle the `chrono` feature on/off but
    /// keep the previous code.
    /// - **Integer** becomes *milliseconds since epoch*.
    /// - **Floats** round as integers to *milliseconds since epoch*.
    /// - **Boolean** returns `TemporalArray::Null`.
    #[cfg(feature = "datetime")]
    pub fn dt(self) -> TemporalArray {
        use crate::enums::time_units::TimeUnit;
        match self {
            Array::TemporalArray(arr) => arr, // move, not clone
            Array::NumericArray(arr) => match arr {
                NumericArray::Int32(mut a) => {
                    let a_mut = Arc::make_mut(&mut a);
                    TemporalArray::Datetime64(Arc::new(DatetimeArray::<i64>::from_vec64(
                        a_mut.data.iter().map(|v| *v as i64).collect(),
                        a_mut.null_mask.take(),
                        Some(TimeUnit::Milliseconds),
                    )))
                }
                NumericArray::Int64(mut a) => {
                    let a_mut = Arc::make_mut(&mut a);
                    TemporalArray::Datetime64(Arc::new(DatetimeArray::<i64>::from_vec64(
                        a_mut.data.iter().copied().collect(),
                        a_mut.null_mask.take(),
                        Some(TimeUnit::Milliseconds),
                    )))
                }
                NumericArray::UInt32(mut a) => {
                    let a_mut = Arc::make_mut(&mut a);
                    TemporalArray::Datetime64(Arc::new(DatetimeArray::<i64>::from_vec64(
                        a_mut.data.iter().map(|v| *v as i64).collect(),
                        a_mut.null_mask.take(),
                        Some(TimeUnit::Milliseconds),
                    )))
                }
                NumericArray::UInt64(mut a) => {
                    let a_mut = Arc::make_mut(&mut a);
                    TemporalArray::Datetime64(Arc::new(DatetimeArray::<i64>::from_vec64(
                        a_mut.data.iter().map(|v| *v as i64).collect(),
                        a_mut.null_mask.take(),
                        Some(TimeUnit::Milliseconds),
                    )))
                }
                NumericArray::Float32(mut a) => {
                    let a_mut = Arc::make_mut(&mut a);
                    TemporalArray::Datetime64(Arc::new(DatetimeArray::<i64>::from_vec64(
                        a_mut.data.iter().map(|v| *v as i64).collect(),
                        a_mut.null_mask.take(),
                        Some(TimeUnit::Milliseconds),
                    )))
                }
                NumericArray::Float64(mut a) => {
                    let a_mut = Arc::make_mut(&mut a);
                    TemporalArray::Datetime64(Arc::new(DatetimeArray::<i64>::from_vec64(
                        a_mut.data.iter().map(|v| *v as i64).collect(),
                        a_mut.null_mask.take(),
                        Some(TimeUnit::Milliseconds),
                    )))
                }
                _ => TemporalArray::Null,
            },
            Array::BooleanArray(_) => TemporalArray::Null,
            Array::TextArray(arr) => match arr {
                TextArray::String32(s) => {
                    let mut out = Vec64::with_capacity(s.len());
                    let mut null_mask = Bitmask::with_capacity(s.len());
                    for i in 0..s.len() {
                        let valid = !s.is_null(i);
                        let val = if valid {
                            use crate::utils::parse_datetime_str;
                            let str_val = unsafe { s.get_str_unchecked(i) };
                            parse_datetime_str(str_val)
                        } else {
                            None
                        };
                        match val {
                            Some(dt) => {
                                out.push(dt);
                                null_mask.set(i, true);
                            }
                            None => {
                                out.push(0);
                                null_mask.set(i, false);
                            }
                        }
                    }
                    TemporalArray::Datetime64(Arc::new(DatetimeArray::<i64>::from_vec64(
                        out,
                        Some(null_mask),
                        Some(TimeUnit::Milliseconds),
                    )))
                }
                _ => TemporalArray::Null,
            },
            Array::Null => TemporalArray::Null,
        }
    }

    /// Returns the length of the array.
    pub fn len(&self) -> usize {
        match self {
            Self::Null => 0,
            _ => match_array!(self, len),
        }
    }

    /// Returns a metadata view and reference over the specified window of this array.
    ///
    /// Does not slice the object (yet).
    ///
    /// Panics if out of bounds.
    #[cfg(feature = "views")]
    pub fn to_window(&self, offset: usize, len: usize) -> ArrayV {
        assert!(offset <= self.len(), "offset out of bounds");
        assert!(offset + len <= self.len(), "slice window out of bounds");
        ArrayV::new(self.clone(), offset, len)
    }

    /// Returns a metadata view and reference over the specified window of this array.
    ///
    /// Does not slice the object (yet).
    ///
    /// Panics if out of bounds.
    #[cfg(feature = "views")]
    pub fn to_window_tuple(&self, offset: usize, len: usize) -> ArrayVT {
        assert!(offset <= self.len(), "offset out of bounds");
        assert!(offset + len <= self.len(), "slice window out of bounds");
        (self, offset, len)
    }

    /// Returns a reference to the inner array as type `Arc<T>`.
    ///
    /// This is compile-time safe if `T` matches the actual payload, but will panic otherwise.
    /// Prefer `.inner_check()` for Option-based pattern.
    #[inline]
    pub fn inner<T: 'static>(&self) -> &Arc<T> {
        macro_rules! match_arm {
            ($inner_enum:ident, $variant:ident, $ty:ty) => {
                if let Array::$inner_enum(inner) = self {
                    if let $inner_enum::$variant(inner2) = inner {
                        // Arc<T> always lives here, so we compare against Arc<T>
                        if TypeId::of::<T>() == TypeId::of::<$ty>() {
                            // inner2: Arc<Ty>, T == Ty
                            // safe to cast: Arc<Ty> -> Arc<T>
                            return unsafe { &*(inner2 as *const Arc<$ty> as *const Arc<T>) };
                        }
                    }
                }
            };
        }

        // NumericArray
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, Int8, IntegerArray<i8>);
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, Int16, IntegerArray<i16>);
        match_arm!(NumericArray, Int32, IntegerArray<i32>);
        match_arm!(NumericArray, Int64, IntegerArray<i64>);
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, UInt8, IntegerArray<u8>);
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, UInt16, IntegerArray<u16>);
        match_arm!(NumericArray, UInt32, IntegerArray<u32>);
        match_arm!(NumericArray, UInt64, IntegerArray<u64>);
        match_arm!(NumericArray, Float32, FloatArray<f32>);
        match_arm!(NumericArray, Float64, FloatArray<f64>);

        // TextArray
        match_arm!(TextArray, String32, StringArray<u32>);
        #[cfg(feature = "large_string")]
        match_arm!(TextArray, String64, StringArray<u64>);
        #[cfg(feature = "extended_categorical")]
        match_arm!(TextArray, Categorical8, CategoricalArray<u8>);
        #[cfg(feature = "extended_categorical")]
        match_arm!(TextArray, Categorical16, CategoricalArray<u16>);
        match_arm!(TextArray, Categorical32, CategoricalArray<u32>);
        #[cfg(feature = "extended_categorical")]
        match_arm!(TextArray, Categorical64, CategoricalArray<u64>);

        // TemporalArray
        #[cfg(feature = "datetime")]
        match_arm!(TemporalArray, Datetime32, DatetimeArray<i32>);
        #[cfg(feature = "datetime")]
        match_arm!(TemporalArray, Datetime64, DatetimeArray<i64>);

        // Boolean
        if let Array::BooleanArray(inner) = self {
            if TypeId::of::<T>() == TypeId::of::<BooleanArray<()>>() {
                return unsafe { &*(inner as *const Arc<BooleanArray<()>> as *const Arc<T>) };
            }
        }

        panic!(
            "Type mismatch: attempted to access Array::{:?} as incompatible type",
            self.arrow_type()
        );
    }

    /// Returns a mutable reference to the inner array as type `T`.
    ///
    /// This method is compile-time safe when the type `T` matches the actual inner type,
    /// but relies on `TypeId` checks and unsafe casting. If an incorrect type is specified,
    /// this will panic at runtime.
    ///
    /// Prefer `inner_check_mut` if you want an `Option`-based version that avoids panics.
    #[inline]
    pub fn inner_mut<T: 'static>(&mut self) -> &mut Arc<T> {
        use std::any::TypeId;

        macro_rules! match_arm {
            ($inner_enum:ident, $variant:ident, $ty:ty) => {
                if let Array::$inner_enum(inner) = self {
                    if let $inner_enum::$variant(inner2) = inner {
                        if TypeId::of::<T>() == TypeId::of::<$ty>() {
                            return unsafe { &mut *(inner2 as *mut Arc<$ty> as *mut Arc<T>) };
                        }
                    }
                }
            };
        }

        // NumericArray
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, Int8, IntegerArray<i8>);
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, Int16, IntegerArray<i16>);
        match_arm!(NumericArray, Int32, IntegerArray<i32>);
        match_arm!(NumericArray, Int64, IntegerArray<i64>);
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, UInt8, IntegerArray<u8>);
        #[cfg(feature = "extended_numeric_types")]
        match_arm!(NumericArray, UInt16, IntegerArray<u16>);
        match_arm!(NumericArray, UInt32, IntegerArray<u32>);
        match_arm!(NumericArray, UInt64, IntegerArray<u64>);
        match_arm!(NumericArray, Float32, FloatArray<f32>);
        match_arm!(NumericArray, Float64, FloatArray<f64>);

        // TextArray
        match_arm!(TextArray, String32, StringArray<u32>);
        #[cfg(feature = "large_string")]
        match_arm!(TextArray, String64, StringArray<u64>);
        #[cfg(feature = "extended_categorical")]
        match_arm!(TextArray, Categorical8, CategoricalArray<u8>);
        #[cfg(feature = "extended_categorical")]
        match_arm!(TextArray, Categorical16, CategoricalArray<u16>);
        match_arm!(TextArray, Categorical32, CategoricalArray<u32>);
        #[cfg(feature = "extended_categorical")]
        match_arm!(TextArray, Categorical64, CategoricalArray<u64>);

        // TemporalArray
        #[cfg(feature = "datetime")]
        match_arm!(TemporalArray, Datetime32, DatetimeArray<i32>);
        #[cfg(feature = "datetime")]
        match_arm!(TemporalArray, Datetime64, DatetimeArray<i64>);

        // Boolean
        if let Array::BooleanArray(inner) = self {
            if TypeId::of::<T>() == TypeId::of::<BooleanArray<()>>() {
                return unsafe { &mut *(inner as *mut Arc<BooleanArray<()>> as *mut Arc<T>) };
            }
        }

        panic!(
            "Type mismatch: attempted to mutably access Array::{:?} as incompatible type",
            self.arrow_type()
        )
    }

    /// Returns a reference to the inner array as type `T`, if the type matches.
    ///
    /// This method performs a runtime `TypeId` check to verify that the provided type `T`
    /// corresponds to the actual inner variant. If the types match, returns `Some(&T)`;
    /// otherwise, returns `None` without panicking.
    ///
    /// Use when the type of the variant is uncertain at compile time.
    #[inline]
    pub fn inner_check<T: 'static>(&self) -> Option<&Arc<T>> {
        use std::any::TypeId;

        macro_rules! match_inner_type {
            ($outer:ident, $variant:ident, $ty:ty) => {
                if TypeId::of::<T>() == TypeId::of::<$ty>() {
                    if let Array::$outer(inner) = self {
                        if let $outer::$variant(inner2) = inner {
                            return Some(unsafe { &*(inner2 as *const Arc<$ty> as *const Arc<T>) });
                        }
                    }
                }
            };
        }

        #[cfg(feature = "extended_numeric_types")]
        match_inner_type!(NumericArray, Int8, IntegerArray<i8>);
        #[cfg(feature = "extended_numeric_types")]
        match_inner_type!(NumericArray, Int16, IntegerArray<i16>);
        match_inner_type!(NumericArray, Int32, IntegerArray<i32>);
        match_inner_type!(NumericArray, Int64, IntegerArray<i64>);
        #[cfg(feature = "extended_numeric_types")]
        match_inner_type!(NumericArray, UInt8, IntegerArray<u8>);
        #[cfg(feature = "extended_numeric_types")]
        match_inner_type!(NumericArray, UInt16, IntegerArray<u16>);
        match_inner_type!(NumericArray, UInt32, IntegerArray<u32>);
        match_inner_type!(NumericArray, UInt64, IntegerArray<u64>);
        match_inner_type!(NumericArray, Float32, FloatArray<f32>);
        match_inner_type!(NumericArray, Float64, FloatArray<f64>);

        match_inner_type!(TextArray, String32, StringArray<u32>);
        #[cfg(feature = "large_string")]
        match_inner_type!(TextArray, String64, StringArray<u64>);
        #[cfg(feature = "extended_categorical")]
        match_inner_type!(TextArray, Categorical8, CategoricalArray<u8>);
        #[cfg(feature = "extended_categorical")]
        match_inner_type!(TextArray, Categorical16, CategoricalArray<u16>);
        match_inner_type!(TextArray, Categorical32, CategoricalArray<u32>);
        #[cfg(feature = "extended_categorical")]
        match_inner_type!(TextArray, Categorical64, CategoricalArray<u64>);

        #[cfg(feature = "datetime")]
        match_inner_type!(TemporalArray, Datetime32, DatetimeArray<i32>);
        #[cfg(feature = "datetime")]
        match_inner_type!(TemporalArray, Datetime64, DatetimeArray<i64>);

        if TypeId::of::<T>() == TypeId::of::<BooleanArray<()>>() {
            if let Array::BooleanArray(inner) = self {
                return Some(unsafe { &*(inner as *const Arc<BooleanArray<()>> as *const Arc<T>) });
            }
        }

        None
    }

    /// Returns a mutable reference to the inner array as type `T`, if the type matches.
    ///
    /// This method performs a runtime `TypeId` check to verify that the provided type `T`
    /// corresponds to the actual inner variant. If the types match, returns `Some(&mut T)`;
    /// otherwise, returns `None` without panicking.
    ///
    /// Use when the type of the variant is uncertain at compile time.
    #[inline]
    pub fn inner_check_mut<T: 'static>(&mut self) -> Option<&mut Arc<T>> {
        use std::any::TypeId;

        macro_rules! match_inner_type_mut {
            ($outer:ident, $variant:ident, $ty:ty) => {
                if TypeId::of::<T>() == TypeId::of::<$ty>() {
                    if let Array::$outer(inner) = self {
                        if let $outer::$variant(inner2) = inner {
                            return Some(unsafe { &mut *(inner2 as *mut Arc<$ty> as *mut Arc<T>) });
                        }
                    }
                }
            };
        }

        #[cfg(feature = "extended_numeric_types")]
        match_inner_type_mut!(NumericArray, Int8, IntegerArray<i8>);
        #[cfg(feature = "extended_numeric_types")]
        match_inner_type_mut!(NumericArray, Int16, IntegerArray<i16>);
        match_inner_type_mut!(NumericArray, Int32, IntegerArray<i32>);
        match_inner_type_mut!(NumericArray, Int64, IntegerArray<i64>);
        #[cfg(feature = "extended_numeric_types")]
        match_inner_type_mut!(NumericArray, UInt8, IntegerArray<u8>);
        #[cfg(feature = "extended_numeric_types")]
        match_inner_type_mut!(NumericArray, UInt16, IntegerArray<u16>);
        match_inner_type_mut!(NumericArray, UInt32, IntegerArray<u32>);
        match_inner_type_mut!(NumericArray, UInt64, IntegerArray<u64>);
        match_inner_type_mut!(NumericArray, Float32, FloatArray<f32>);
        match_inner_type_mut!(NumericArray, Float64, FloatArray<f64>);

        match_inner_type_mut!(TextArray, String32, StringArray<u32>);
        #[cfg(feature = "large_string")]
        match_inner_type_mut!(TextArray, String64, StringArray<u64>);
        #[cfg(feature = "extended_categorical")]
        match_inner_type_mut!(TextArray, Categorical8, CategoricalArray<u8>);
        #[cfg(feature = "extended_categorical")]
        match_inner_type_mut!(TextArray, Categorical16, CategoricalArray<u16>);
        match_inner_type_mut!(TextArray, Categorical32, CategoricalArray<u32>);
        #[cfg(feature = "extended_categorical")]
        match_inner_type_mut!(TextArray, Categorical64, CategoricalArray<u64>);

        #[cfg(feature = "datetime")]
        match_inner_type_mut!(TemporalArray, Datetime32, DatetimeArray<i32>);
        #[cfg(feature = "datetime")]
        match_inner_type_mut!(TemporalArray, Datetime64, DatetimeArray<i64>);

        if TypeId::of::<T>() == TypeId::of::<BooleanArray<()>>() {
            if let Array::BooleanArray(inner) = self {
                return Some(unsafe { &mut *(inner as *mut Arc<BooleanArray<()>> as *mut Arc<T>) });
            }
        }

        None
    }

    #[inline]
    pub fn as_slice<T>(&self, offset: usize, len: usize) -> &[T] {
        match self {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(arr) => {
                    cast_slice::<i8, T>(arr.data(), offset, len).expect("cast failed")
                }

                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(arr) => {
                    cast_slice::<i16, T>(arr.data(), offset, len).expect("cast failed")
                }

                NumericArray::Int32(arr) => {
                    cast_slice::<i32, T>(arr.data(), offset, len).expect("cast failed")
                }

                NumericArray::Int64(arr) => {
                    cast_slice::<i64, T>(arr.data(), offset, len).expect("cast failed")
                }

                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(arr) => {
                    cast_slice::<u8, T>(arr.data(), offset, len).expect("cast failed")
                }

                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(arr) => {
                    cast_slice::<u16, T>(arr.data(), offset, len).expect("cast failed")
                }

                NumericArray::UInt32(arr) => {
                    cast_slice::<u32, T>(arr.data(), offset, len).expect("cast failed")
                }

                NumericArray::UInt64(arr) => {
                    cast_slice::<u64, T>(arr.data(), offset, len).expect("cast failed")
                }

                NumericArray::Float32(arr) => {
                    cast_slice::<f32, T>(arr.data(), offset, len).expect("cast failed")
                }

                NumericArray::Float64(arr) => {
                    cast_slice::<f64, T>(arr.data(), offset, len).expect("cast failed")
                }

                NumericArray::Null => panic!("Null array has no data payload"),
            },

            Array::TextArray(inner) => match inner {
                TextArray::String32(_) | TextArray::Null => {
                    panic!(
                        "Strings use UTF-8 + offsets. Use logical accessor instead, or `slice_raw` if you do want byte access."
                    )
                }
                #[cfg(feature = "large_string")]
                TextArray::String64(_) => {
                    panic!(
                        "Strings use UTF-8 + offsets. Use logical accessor instead, or `slice_raw` if you do want byte access."
                    )
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical8(arr) => {
                    cast_slice::<u8, T>(arr.data(), offset, len).expect("cast failed")
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(arr) => {
                    cast_slice::<u16, T>(arr.data(), offset, len).expect("cast failed")
                }
                TextArray::Categorical32(arr) => {
                    cast_slice::<u32, T>(arr.data(), offset, len).expect("cast failed")
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(arr) => {
                    cast_slice::<u64, T>(arr.data(), offset, len).expect("cast failed")
                }
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(arr) => {
                    cast_slice::<i32, T>(arr.data(), offset, len).expect("cast failed")
                }
                TemporalArray::Datetime64(arr) => {
                    cast_slice::<i64, T>(arr.data(), offset, len).expect("cast failed")
                }
                TemporalArray::Null => panic!("Null array has no data payload"),
            },

            Array::BooleanArray(_) => {
                panic!(
                    "Bool arrays are bit-packed; use logical accessor instead, or `slice_raw` if you do want byte access."
                )
            }

            Array::Null => panic!("Null array has no data payload"),
        }
    }

    #[inline]
    pub fn slice_raw<T: 'static>(&self, offset: usize, len: usize) -> Option<&[T]> {
        use std::any::TypeId;

        match self {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) if TypeId::of::<T>() == TypeId::of::<i8>() => {
                    cast_slice::<i8, T>(&a.data, offset, len)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) if TypeId::of::<T>() == TypeId::of::<i16>() => {
                    cast_slice::<i16, T>(&a.data, offset, len)
                }
                NumericArray::Int32(a) if TypeId::of::<T>() == TypeId::of::<i32>() => {
                    cast_slice::<i32, T>(&a.data, offset, len)
                }
                NumericArray::Int64(a) if TypeId::of::<T>() == TypeId::of::<i64>() => {
                    cast_slice::<i64, T>(&a.data, offset, len)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                    cast_slice::<u8, T>(&a.data, offset, len)
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) if TypeId::of::<T>() == TypeId::of::<u16>() => {
                    cast_slice::<u16, T>(&a.data, offset, len)
                }
                NumericArray::UInt32(a) if TypeId::of::<T>() == TypeId::of::<u32>() => {
                    cast_slice::<u32, T>(&a.data, offset, len)
                }
                NumericArray::UInt64(a) if TypeId::of::<T>() == TypeId::of::<u64>() => {
                    cast_slice::<u64, T>(&a.data, offset, len)
                }
                NumericArray::Float32(a) if TypeId::of::<T>() == TypeId::of::<f32>() => {
                    cast_slice::<f32, T>(&a.data, offset, len)
                }
                NumericArray::Float64(a) if TypeId::of::<T>() == TypeId::of::<f64>() => {
                    cast_slice::<f64, T>(&a.data, offset, len)
                }
                _ => None,
            },

            Array::BooleanArray(a) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                let start = offset / 8;
                let end = (offset + len + 7) / 8;
                let slice = &a[start..end];
                Some(unsafe { &*(slice as *const [u8] as *const [T]) })
            }

            Array::TextArray(inner) => match inner {
                TextArray::String32(a) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                    cast_slice::<u8, T>(&a.data, offset, len)
                }
                #[cfg(feature = "large_string")]
                TextArray::String64(a) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                    cast_slice::<u8, T>(&a.data, offset, len)
                }
                TextArray::Categorical32(a) if TypeId::of::<T>() == TypeId::of::<u32>() => {
                    cast_slice::<u32, T>(&a.data, offset, len)
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical8(a) if TypeId::of::<T>() == TypeId::of::<u8>() => {
                    cast_slice::<u8, T>(&a.data, offset, len)
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(a) if TypeId::of::<T>() == TypeId::of::<u16>() => {
                    cast_slice::<u16, T>(&a.data, offset, len)
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(a) if TypeId::of::<T>() == TypeId::of::<u64>() => {
                    cast_slice::<u64, T>(&a.data, offset, len)
                }
                _ => None,
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(a) if TypeId::of::<T>() == TypeId::of::<i32>() => {
                    cast_slice::<i32, T>(&a.data, offset, len)
                }
                TemporalArray::Datetime64(a) if TypeId::of::<T>() == TypeId::of::<i64>() => {
                    cast_slice::<i64, T>(&a.data, offset, len)
                }
                _ => None,
            },

            _ => None,
        }
    }

    /// Returns a new `Array` of the same variant sliced to the given offset and length
    /// .
    /// Copies the data of the scoped range that's selected.
    ///  
    /// If out-of-bounds, returns Self::Null.
    /// All null mask, offsets, etc. are trimmed.
    #[inline]
    pub fn slice_clone(&self, offset: usize, len: usize) -> Self {
        match self {
            Array::NumericArray(inner) => Self::NumericArray(match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(arr) => NumericArray::Int8(arr.slice_clone(offset, len)),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(arr) => NumericArray::Int16(arr.slice_clone(offset, len)),
                NumericArray::Int32(arr) => NumericArray::Int32(arr.slice_clone(offset, len)),
                NumericArray::Int64(arr) => NumericArray::Int64(arr.slice_clone(offset, len)),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(arr) => NumericArray::UInt8(arr.slice_clone(offset, len)),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(arr) => NumericArray::UInt16(arr.slice_clone(offset, len)),
                NumericArray::UInt32(arr) => NumericArray::UInt32(arr.slice_clone(offset, len)),
                NumericArray::UInt64(arr) => NumericArray::UInt64(arr.slice_clone(offset, len)),
                NumericArray::Float32(arr) => NumericArray::Float32(arr.slice_clone(offset, len)),
                NumericArray::Float64(arr) => NumericArray::Float64(arr.slice_clone(offset, len)),
                NumericArray::Null => NumericArray::Null,
            }),
            Array::TextArray(inner) => Self::TextArray(match inner {
                TextArray::String32(arr) => TextArray::String32(arr.slice_clone(offset, len)),
                #[cfg(feature = "large_string")]
                TextArray::String64(arr) => TextArray::String64(arr.slice_clone(offset, len)),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical8(arr) => {
                    TextArray::Categorical8(arr.slice_clone(offset, len))
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(arr) => {
                    TextArray::Categorical16(arr.slice_clone(offset, len))
                }
                TextArray::Categorical32(arr) => {
                    TextArray::Categorical32(arr.slice_clone(offset, len))
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(arr) => {
                    TextArray::Categorical64(arr.slice_clone(offset, len))
                }
                TextArray::Null => TextArray::Null,
            }),
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => Self::TemporalArray(match inner {
                TemporalArray::Datetime32(arr) => {
                    TemporalArray::Datetime32(arr.slice_clone(offset, len))
                }
                TemporalArray::Datetime64(arr) => {
                    TemporalArray::Datetime64(arr.slice_clone(offset, len))
                }
                TemporalArray::Null => TemporalArray::Null,
            }),
            Array::BooleanArray(arr) => Self::BooleanArray(arr.slice_clone(offset, len)),
            Array::Null => Self::Null,
        }
    }

    /// Arrow physical type for this array.
    pub fn arrow_type(&self) -> ArrowType {
        match self {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(_) => ArrowType::Int8,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(_) => ArrowType::Int16,
                NumericArray::Int32(_) => ArrowType::Int32,
                NumericArray::Int64(_) => ArrowType::Int64,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(_) => ArrowType::UInt8,
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(_) => ArrowType::UInt16,
                NumericArray::UInt32(_) => ArrowType::UInt32,
                NumericArray::UInt64(_) => ArrowType::UInt64,
                NumericArray::Float32(_) => ArrowType::Float32,
                NumericArray::Float64(_) => ArrowType::Float64,
                NumericArray::Null => ArrowType::Null,
            },
            Array::TextArray(inner) => match inner {
                TextArray::String32(_) => ArrowType::String,
                #[cfg(feature = "large_string")]
                TextArray::String64(_) => ArrowType::LargeString,
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical8(_) => ArrowType::Dictionary(CategoricalIndexType::UInt8),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(_) => ArrowType::Dictionary(CategoricalIndexType::UInt16),
                TextArray::Categorical32(_) => ArrowType::Dictionary(CategoricalIndexType::UInt32),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(_) => ArrowType::Dictionary(CategoricalIndexType::UInt64),
                TextArray::Null => ArrowType::Null,
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(_) => ArrowType::Date32,
                TemporalArray::Datetime64(_) => ArrowType::Date64,
                TemporalArray::Null => ArrowType::Null,
            },
            Array::BooleanArray(_) => ArrowType::Boolean,
            Array::Null => ArrowType::Null,
        }
    }

    /// Column nullability
    pub fn is_nullable(&self) -> bool {
        match self {
            Self::Null => true,
            _ => match_array!(self, is_nullable),
        }
    }

    /// Returns the underlying null mask of the array
    pub fn null_mask(&self) -> Option<&Bitmask> {
        match self {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(arr) => arr.null_mask.as_ref(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(arr) => arr.null_mask.as_ref(),
                NumericArray::Int32(arr) => arr.null_mask.as_ref(),
                NumericArray::Int64(arr) => arr.null_mask.as_ref(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(arr) => arr.null_mask.as_ref(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(arr) => arr.null_mask.as_ref(),
                NumericArray::UInt32(arr) => arr.null_mask.as_ref(),
                NumericArray::UInt64(arr) => arr.null_mask.as_ref(),
                NumericArray::Float32(arr) => arr.null_mask.as_ref(),
                NumericArray::Float64(arr) => arr.null_mask.as_ref(),
                NumericArray::Null => None,
            },
            Array::BooleanArray(arr) => arr.null_mask.as_ref(),
            Array::TextArray(inner) => match inner {
                TextArray::String32(arr) => arr.null_mask.as_ref(),
                #[cfg(feature = "large_string")]
                TextArray::String64(arr) => arr.null_mask.as_ref(),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical8(arr) => arr.null_mask.as_ref(),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(arr) => arr.null_mask.as_ref(),
                TextArray::Categorical32(arr) => arr.null_mask.as_ref(),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(arr) => arr.null_mask.as_ref(),
                TextArray::Null => None,
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(arr) => arr.null_mask.as_ref(),
                TemporalArray::Datetime64(arr) => arr.null_mask.as_ref(),
                TemporalArray::Null => None,
            },
            Array::Null => None,
        }
    }

    /// Returns a pointer to the backing data (contiguous bytes), length in elements, and element size.
    ///
    /// This is not logical length - it is total raw bytes in the buffer, so for non-fixed width
    /// types such as bit-packed booleans or strings, please factor this in accordingly.
    pub fn data_ptr_and_byte_len(&self) -> (*const u8, usize, usize) {
        match self {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<i8>(),
                ),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<i16>(),
                ),
                NumericArray::Int32(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<i32>(),
                ),
                NumericArray::Int64(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<i64>(),
                ),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u8>(),
                ),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u16>(),
                ),
                NumericArray::UInt32(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u32>(),
                ),
                NumericArray::UInt64(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u64>(),
                ),
                NumericArray::Float32(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<f32>(),
                ),
                NumericArray::Float64(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<f64>(),
                ),
                NumericArray::Null => (std::ptr::null(), 0, 0),
            },
            Array::BooleanArray(a) => (a.data.as_ptr() as *const u8, a.data.len(), 1),
            Array::TextArray(inner) => match inner {
                TextArray::String32(a) => (a.data.as_ptr(), a.data.len(), 1),
                #[cfg(feature = "large_string")]
                TextArray::String64(a) => (a.data.as_ptr(), a.data.len(), 1),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical8(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u8>(),
                ),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u16>(),
                ),
                TextArray::Categorical32(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u32>(),
                ),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<u64>(),
                ),
                TextArray::Null => (std::ptr::null(), 0, 0),
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<i32>(),
                ),
                TemporalArray::Datetime64(a) => (
                    a.data.as_ptr() as *const u8,
                    a.len(),
                    std::mem::size_of::<i64>(),
                ),
                TemporalArray::Null => (std::ptr::null(), 0, 0),
            },
            Array::Null => (std::ptr::null(), 0, 0),
        }
    }

    /// Returns a pointer to the null mask and its length in bytes, if present.
    pub fn null_mask_ptr_and_byte_len(&self) -> Option<(*const u8, usize)> {
        match self {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                NumericArray::Int32(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                NumericArray::Int64(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                NumericArray::UInt32(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                NumericArray::UInt64(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
                NumericArray::Float32(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                NumericArray::Float64(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                NumericArray::Null => None,
            },
            Array::BooleanArray(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity())),
            Array::TextArray(inner) => match inner {
                TextArray::String32(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.len())),
                #[cfg(feature = "large_string")]
                TextArray::String64(a) => a.null_mask.as_ref().map(|m| (m.as_ptr(), m.len())),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical8(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                TextArray::Categorical32(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                TextArray::Null => None,
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                TemporalArray::Datetime64(a) => {
                    a.null_mask.as_ref().map(|m| (m.as_ptr(), m.capacity()))
                }
                TemporalArray::Null => None,
            },
            Array::Null => None,
        }
    }

    /// Offsets pointer/len for variable-length types
    pub fn offsets_ptr_and_len(&self) -> Option<(*const u8, usize)> {
        match self {
            Array::TextArray(inner) => match inner {
                TextArray::String32(a) => Some((
                    a.offsets.as_ptr() as *const u8,
                    a.offsets.len() * std::mem::size_of::<u32>(),
                )),
                #[cfg(feature = "large_string")]
                TextArray::String64(a) => Some((
                    a.offsets.as_ptr() as *const u8,
                    a.offsets.len() * std::mem::size_of::<u64>(),
                )),
                _ => None,
            },
            _ => None,
        }
    }

    /// Returns the null count of the array
    pub fn null_count(&self) -> usize {
        match self {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(a) => a.null_count(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(a) => a.null_count(),
                NumericArray::Int32(a) => a.null_count(),
                NumericArray::Int64(a) => a.null_count(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(a) => a.null_count(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(a) => a.null_count(),
                NumericArray::UInt32(a) => a.null_count(),
                NumericArray::UInt64(a) => a.null_count(),
                NumericArray::Float32(a) => a.null_count(),
                NumericArray::Float64(a) => a.null_count(),
                NumericArray::Null => 0,
            },
            Array::BooleanArray(a) => a.null_count(),
            Array::TextArray(inner) => match inner {
                TextArray::String32(a) => a.null_count(),
                #[cfg(feature = "large_string")]
                TextArray::String64(a) => a.null_count(),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical8(a) => a.null_count(),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(a) => a.null_count(),
                TextArray::Categorical32(a) => a.null_count(),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(a) => a.null_count(),
                TextArray::Null => 0,
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(a) => a.null_count(),
                TemporalArray::Datetime64(a) => a.null_count(),
                TemporalArray::Null => 0,
            },
            Array::Null => 0,
        }
    }

    /// Appends all values (and null mask if present) from `other` into `self`.
    ///
    /// Panics if the two arrays are of different variants or incompatible types.
    ///
    /// This function uses copy-on-write semantics for arrays wrapped in `Arc`.
    /// If `self` is the only owner of its data, appends are performed in place without copying the first array.
    /// If the array data is shared (`Arc` reference count > 1), the data is first cloned
    /// (so the mutation does not affect other owners), and the append is then performed on the unique copy.
    /// The second array is allocated into the buffer, which is standard.
    pub fn concat_array(&mut self, other: &Self) {
        match (self, other) {
            (Array::NumericArray(lhs), Array::NumericArray(rhs)) => lhs.append_array(rhs),
            (Array::BooleanArray(a), Array::BooleanArray(b)) => Arc::make_mut(a).append_array(b),
            (Array::TextArray(lhs), Array::TextArray(rhs)) => lhs.append_array(rhs),
            #[cfg(feature = "datetime")]
            (Array::TemporalArray(lhs), Array::TemporalArray(rhs)) => lhs.append_array(rhs),
            (Array::Null, Array::Null) => (),
            (lhs, rhs) => panic!(
                "Cannot append {:?} into {:?}",
                rhs.arrow_type(),
                lhs.arrow_type()
            ),
        }
    }

    /// Derive a `Field` from the array (via `Field::from_array`)
    /// and call `to_apache_arrow_with_field`.
    ///
    /// For Timestamp/Time/Duration/Interval, prefer passing
    /// an explicit `Field` via `to_apache_arrow_with_field`.
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn to_apache_arrow(&self, name: &str) -> ArrayRef {
        let derived = Field::from_array(name, self, None);
        self.to_apache_arrow_with_field(&derived)
    }

    /// Export this Minarrow array via Arrow C FFI using `field` as the
    /// logical type; then import into arrow-rs as `ArrayRef`.
    ///
    /// Use this when you need explicit temporal/interval semantics.
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn to_apache_arrow_with_field(&self, field: &Field) -> ArrayRef {
        // 1) Export using the provided logical field (source of truth)
        let schema = Schema::from(vec![field.clone()]);
        let (c_arr, c_schema) = export_to_c(Arc::new(self.clone()), schema);

        // 2) Move FFI structs out (arrow-rs takes ownership)
        let arr_ptr = c_arr as *mut arrow::array::ffi::FFI_ArrowArray;
        let sch_ptr = c_schema as *mut arrow::array::ffi::FFI_ArrowSchema;
        let ffi_arr = unsafe { arr_ptr.read() };
        let ffi_sch = unsafe { sch_ptr.read() };

        // 3) Import into arrow-rs (fully qualified)
        let array_data = unsafe {
            arrow::array::ffi::from_ffi(ffi_arr, &ffi_sch).expect("arrow-rs FFI import failed")
        };

        arrow::array::make_array(array_data)
    }

    // ** The below 2 polars functions are tested under tests/polars.rs **

    /// Build a Polars Series using a derived Field (dtype/nullability from the array).
    ///
    /// If you need an explicit logical type, use `to_polars_with_field`.
    /// For **Timestamp/Time/Duration/Interval**:
    ///     - prefer `to_polars_with_field` with an explicit Field::new.
    ///     - then, append the`minarrow::ArrowType` to the `Field` that matches the desired datetime type.
    #[cfg(feature = "cast_polars")]
    pub fn to_polars(&self, name: &str) -> polars::prelude::Series {
        #[cfg(feature = "datetime")]
        use crate::{TemporalArray, TimeUnit, ffi::arrow_dtype::ArrowType};

        // We infer what is directly inferrable and expected
        let field = match self {
            #[cfg(feature = "datetime")]
            Array::TemporalArray(TemporalArray::Datetime32(a)) => {
                let ty = match a.time_unit {
                    TimeUnit::Days => ArrowType::Date32,
                    TimeUnit::Seconds => ArrowType::Time32(TimeUnit::Seconds),
                    TimeUnit::Milliseconds => ArrowType::Time32(TimeUnit::Milliseconds),
                    _ => ArrowType::Date32,
                };
                Field::new(name.to_string(), ty, a.is_nullable(), None)
            }
            #[cfg(feature = "datetime")]
            Array::TemporalArray(TemporalArray::Datetime64(a)) => {
                let ty = match a.time_unit {
                    TimeUnit::Milliseconds => ArrowType::Date64,
                    TimeUnit::Seconds => ArrowType::Timestamp(TimeUnit::Seconds),
                    TimeUnit::Microseconds => ArrowType::Timestamp(TimeUnit::Microseconds),
                    TimeUnit::Nanoseconds => ArrowType::Timestamp(TimeUnit::Nanoseconds),
                    // We default to Date64 here for compatibility rather than Date32
                    TimeUnit::Days => ArrowType::Date64,
                };
                Field::new(name.to_string(), ty, a.is_nullable(), None)
            }
            _ => Field::from_array(name.to_string(), self, None),
        };

        self.to_polars_with_field(name, &field)
    }

    /// Export via Arrow C -> import into polars_arrow -> build Series.
    /// Field supplies logical type; `name` is used for the Series.
    ///
    /// For **Timestamp/Time/Duration/Interval**: append `minarrow::ArrowType` to the `Field`
    /// that matches the desired datetime type.
    #[cfg(feature = "cast_polars")]
    pub fn to_polars_with_field(&self, name: &str, field: &Field) -> polars::prelude::Series {
        use std::sync::Arc;

        // 1) Export  Minarrow array with the provided logical Field
        let schema = crate::ffi::schema::Schema::from(vec![field.clone()]);
        let (c_arr, c_schema) =
            crate::ffi::arrow_c_ffi::export_to_c(Arc::new(self.clone()), schema);

        // 2) Move ArrowArray (ownership transfer to arrow2)
        let arr_ptr = c_arr as *mut polars_arrow::ffi::ArrowArray;
        let _sch_ptr = c_schema as *mut polars_arrow::ffi::ArrowSchema;
        let arr_val = unsafe { std::ptr::read(arr_ptr) };

        // 3) Build arrow2 dtype directly from Field
        // We do it this way to avoid (at the time of writing) unsupported FFI types like Nanoseconds.
        let a2_dtype: polars_arrow::datatypes::ArrowDataType = match &field.dtype {
            crate::ffi::arrow_dtype::ArrowType::Null => {
                polars_arrow::datatypes::ArrowDataType::Null
            }
            crate::ffi::arrow_dtype::ArrowType::Boolean => {
                polars_arrow::datatypes::ArrowDataType::Boolean
            }

            #[cfg(feature = "extended_numeric_types")]
            crate::ffi::arrow_dtype::ArrowType::Int8 => {
                polars_arrow::datatypes::ArrowDataType::Int8
            }
            #[cfg(feature = "extended_numeric_types")]
            crate::ffi::arrow_dtype::ArrowType::Int16 => {
                polars_arrow::datatypes::ArrowDataType::Int16
            }
            crate::ffi::arrow_dtype::ArrowType::Int32 => {
                polars_arrow::datatypes::ArrowDataType::Int32
            }
            crate::ffi::arrow_dtype::ArrowType::Int64 => {
                polars_arrow::datatypes::ArrowDataType::Int64
            }

            #[cfg(feature = "extended_numeric_types")]
            crate::ffi::arrow_dtype::ArrowType::UInt8 => {
                polars_arrow::datatypes::ArrowDataType::UInt8
            }
            #[cfg(feature = "extended_numeric_types")]
            crate::ffi::arrow_dtype::ArrowType::UInt16 => {
                polars_arrow::datatypes::ArrowDataType::UInt16
            }
            crate::ffi::arrow_dtype::ArrowType::UInt32 => {
                polars_arrow::datatypes::ArrowDataType::UInt32
            }
            crate::ffi::arrow_dtype::ArrowType::UInt64 => {
                polars_arrow::datatypes::ArrowDataType::UInt64
            }

            crate::ffi::arrow_dtype::ArrowType::Float32 => {
                polars_arrow::datatypes::ArrowDataType::Float32
            }
            crate::ffi::arrow_dtype::ArrowType::Float64 => {
                polars_arrow::datatypes::ArrowDataType::Float64
            }

            crate::ffi::arrow_dtype::ArrowType::String => {
                polars_arrow::datatypes::ArrowDataType::Utf8
            }
            #[cfg(feature = "large_string")]
            crate::ffi::arrow_dtype::ArrowType::LargeString => {
                polars_arrow::datatypes::ArrowDataType::LargeUtf8
            }

            #[cfg(feature = "datetime")]
            crate::ffi::arrow_dtype::ArrowType::Date32 => {
                polars_arrow::datatypes::ArrowDataType::Date32
            }
            #[cfg(feature = "datetime")]
            crate::ffi::arrow_dtype::ArrowType::Date64 => {
                polars_arrow::datatypes::ArrowDataType::Date64
            }

            #[cfg(feature = "datetime")]
            crate::ffi::arrow_dtype::ArrowType::Time32(u) => {
                polars_arrow::datatypes::ArrowDataType::Time32(match u {
                    crate::TimeUnit::Seconds => polars_arrow::datatypes::TimeUnit::Second,
                    crate::TimeUnit::Milliseconds => polars_arrow::datatypes::TimeUnit::Millisecond,
                    _ => panic!("Time32 supports Seconds or Milliseconds only"),
                })
            }
            #[cfg(feature = "datetime")]
            crate::ffi::arrow_dtype::ArrowType::Time64(u) => {
                polars_arrow::datatypes::ArrowDataType::Time64(match u {
                    crate::TimeUnit::Microseconds => polars_arrow::datatypes::TimeUnit::Microsecond,
                    crate::TimeUnit::Nanoseconds => polars_arrow::datatypes::TimeUnit::Nanosecond,
                    _ => panic!("Time64 supports Microseconds or Nanoseconds only"),
                })
            }

            #[cfg(feature = "datetime")]
            crate::ffi::arrow_dtype::ArrowType::Duration32(u) => {
                polars_arrow::datatypes::ArrowDataType::Duration(match u {
                    crate::TimeUnit::Seconds => polars_arrow::datatypes::TimeUnit::Second,
                    crate::TimeUnit::Milliseconds => polars_arrow::datatypes::TimeUnit::Millisecond,
                    _ => panic!("Duration32 supports Seconds or Milliseconds only"),
                })
            }
            #[cfg(feature = "datetime")]
            crate::ffi::arrow_dtype::ArrowType::Duration64(u) => {
                polars_arrow::datatypes::ArrowDataType::Duration(match u {
                    crate::TimeUnit::Microseconds => polars_arrow::datatypes::TimeUnit::Microsecond,
                    crate::TimeUnit::Nanoseconds => polars_arrow::datatypes::TimeUnit::Nanosecond,
                    _ => panic!("Duration64 supports Microseconds or Nanoseconds only"),
                })
            }

            #[cfg(feature = "datetime")]
            crate::ffi::arrow_dtype::ArrowType::Timestamp(u) => {
                polars_arrow::datatypes::ArrowDataType::Timestamp(
                    match u {
                        crate::TimeUnit::Seconds => polars_arrow::datatypes::TimeUnit::Second,
                        crate::TimeUnit::Milliseconds => {
                            polars_arrow::datatypes::TimeUnit::Millisecond
                        }
                        crate::TimeUnit::Microseconds => {
                            polars_arrow::datatypes::TimeUnit::Microsecond
                        }
                        crate::TimeUnit::Nanoseconds => {
                            polars_arrow::datatypes::TimeUnit::Nanosecond
                        }
                        crate::TimeUnit::Days => panic!("Timestamp(Days) is invalid"),
                    },
                    None,
                )
            }

            #[cfg(feature = "datetime")]
            crate::ffi::arrow_dtype::ArrowType::Interval(iu) => {
                polars_arrow::datatypes::ArrowDataType::Interval(match iu {
                    crate::IntervalUnit::YearMonth => {
                        polars_arrow::datatypes::IntervalUnit::YearMonth
                    }
                    crate::IntervalUnit::DaysTime => polars_arrow::datatypes::IntervalUnit::DayTime,
                    crate::IntervalUnit::MonthDaysNs => {
                        polars_arrow::datatypes::IntervalUnit::MonthDayNano
                    }
                })
            }

            crate::ffi::arrow_dtype::ArrowType::Dictionary(idx) => {
                let key: polars_arrow::datatypes::IntegerType = match idx {
                    #[cfg(feature = "extended_categorical")]
                    crate::ffi::arrow_dtype::CategoricalIndexType::UInt8 => {
                        polars_arrow::datatypes::IntegerType::UInt8
                    }
                    #[cfg(feature = "extended_categorical")]
                    crate::ffi::arrow_dtype::CategoricalIndexType::UInt16 => {
                        polars_arrow::datatypes::IntegerType::UInt16
                    }
                    crate::ffi::arrow_dtype::CategoricalIndexType::UInt32 => {
                        polars_arrow::datatypes::IntegerType::UInt32
                    }
                    #[cfg(feature = "extended_categorical")]
                    crate::ffi::arrow_dtype::CategoricalIndexType::UInt64 => {
                        polars_arrow::datatypes::IntegerType::UInt64
                    }
                };
                polars_arrow::datatypes::ArrowDataType::Dictionary(
                    key,
                    Box::new(polars_arrow::datatypes::ArrowDataType::Utf8),
                    false,
                )
            }
        };

        // 4) Import into arrow2 and build a Polars Series
        let a2_array: Box<dyn polars_arrow::array::Array> =
            unsafe { polars_arrow::ffi::import_array_from_c(arr_val, a2_dtype.clone()) }
                .expect("polars_arrow import_array_from_c failed");

        polars::prelude::Series::from_arrow(name.into(), a2_array)
            .expect("Polars Series::from_arrow failed")
    }
}

#[inline(always)]
pub fn cast_slice<'a, U, T>(data: &'a [U], offset: usize, len: usize) -> Option<&'a [T]> {
    // Safety: The caller is matching on a specific variant where U == T.
    // Only returns Some if bounds are valid.
    debug_assert_eq!(std::mem::size_of::<U>(), std::mem::size_of::<T>());
    if offset.checked_add(len)? > data.len() {
        return None;
    }
    Some(unsafe { &*(&data[offset..offset + len] as *const [U] as *const [T]) })
}

impl Display for Array {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Array::Null => writeln!(f, "Array<Null>\n[null]"),
            Array::BooleanArray(arr) => {
                writeln!(f, "Array<Boolean>")?;
                Display::fmt(arr, f)
            }
            Array::NumericArray(arr) => {
                writeln!(f, "Array<Numeric>")?;
                Display::fmt(arr, f)
            }
            #[cfg(feature = "datetime")]
            Array::TemporalArray(arr) => {
                writeln!(f, "Array<Temporal>")?;
                Display::fmt(arr, f)
            }
            Array::TextArray(arr) => {
                writeln!(f, "Array<Text>")?;
                Display::fmt(arr, f)
            }
        }
    }
}

#[inline(always)]
fn clear_bit_in_place(bits: &mut Vec64<u8>, i: usize) {
    let byte = i >> 3;
    bits[byte] &= !(1u8 << (i & 7));
}

// String option extraction for Vec64 - produces owned strings
pub fn extract_string_option_values64_owned<T: AsRef<str>>(
    options: Vec64<Option<T>>,
) -> (Vec64<String>, Option<Bitmask>) {
    let len = options.len();
    let mut values = Vec64::with_capacity(len);

    // Start with all bits valid; clear for nulls
    let mut null_bytes = Vec64::with_capacity((len + 7) / 8);
    null_bytes.resize((len + 7) / 8, 0xFFu8);

    let mut has_nulls = false;

    for (i, opt) in options.into_iter().enumerate() {
        match opt {
            Some(s) => values.push(s.as_ref().to_string()),
            None => {
                values.push(String::new());
                clear_bit_in_place(&mut null_bytes, i);
                has_nulls = true;
            }
        }
    }

    let mask = if has_nulls {
        Some(Bitmask::from_bytes(null_bytes, len))
    } else {
        None
    };
    (values, mask)
}

// ===== categoricals (borrowed &str) =====
pub fn extract_categorical_option_values64(
    options: Vec64<Option<&str>>,
) -> (Vec64<&str>, Option<Bitmask>) {
    let len = options.len();
    let mut values: Vec64<&str> = Vec64::with_capacity(len);

    // Start with all bits valid; clear for nulls
    let mut null_bytes = Vec64::with_capacity((len + 7) / 8);
    null_bytes.resize((len + 7) / 8, 0xFFu8);

    let mut has_nulls = false;

    for (i, opt) in options.into_iter().enumerate() {
        match opt {
            Some(s) => values.push(s),
            None => {
                values.push(""); // sentinel; masked out
                clear_bit_in_place(&mut null_bytes, i);
                has_nulls = true;
            }
        }
    }

    let mask = if has_nulls {
        Some(Bitmask::from_bytes(null_bytes, len))
    } else {
        None
    };
    (values, mask)
}
// Generic numeric option extraction for Vec64
pub fn extract_option_values64<T: Default + Copy>(
    options: Vec64<Option<T>>,
) -> (Vec64<T>, Option<Bitmask>) {
    let len = options.len();
    let mut values = Vec64::with_capacity(len);

    // Start with all bits valid; clear for nulls
    let mut null_bytes = Vec64::with_capacity((len + 7) / 8);
    null_bytes.resize((len + 7) / 8, 0xFFu8);

    let mut has_nulls = false;

    for (i, opt) in options.into_iter().enumerate() {
        match opt {
            Some(v) => values.push(v),
            None => {
                values.push(T::default());
                clear_bit_in_place(&mut null_bytes, i);
                has_nulls = true;
            }
        }
    }

    let mask = if has_nulls {
        Some(Bitmask::from_bytes(null_bytes, len))
    } else {
        None
    };
    (values, mask)
}

#[allow(dead_code)]
pub fn extract_categorical_option_values64_owned(
    options: Vec64<Option<&str>>,
) -> (Vec64<&str>, Option<Bitmask>) {
    extract_categorical_option_values64(options)
}

// Helper macro to detect if any element is None
#[macro_export]
macro_rules! has_nulls {
    () => { false };
    (None $(, $rest:expr)*) => { true };
    (None) => { true };
    ($first:expr $(, $rest:expr)*) => { has_nulls!($($rest),*) };
    ($first:expr) => { false };
}

// ======== numeric ========

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_i8 {
    // Handle Vec64 input
    ($v:expr) => {
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64($v, None))
    };
    // Handle literal arrays
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;

        // Check if any element is None by trying to match patterns
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64(temp_vec, None))
    }};
    // Handle empty arrays
    () => {
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64(vec64![], None))
    };
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_i16 {
    ($v:expr) => {
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64(vec64![], None))
    };
}

#[macro_export]
macro_rules! arr_i32 {
    ($v:expr) => {
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64(vec64![], None))
    };
}

#[macro_export]
macro_rules! arr_i64 {
    ($v:expr) => {
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64(vec64![], None))
    };
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_u8 {
    ($v:expr) => {
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64(vec64![], None))
    };
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_u16 {
    ($v:expr) => {
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64(vec64![], None))
    };
}

#[macro_export]
macro_rules! arr_u32 {
    ($v:expr) => {
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64(vec64![], None))
    };
}

#[macro_export]
macro_rules! arr_u64 {
    ($v:expr) => {
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64(vec64![], None))
    };
}

// ======== Float types ========

#[macro_export]
macro_rules! arr_f32 {
    ($v:expr) => {
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64(vec64![], None))
    };
}

#[macro_export]
macro_rules! arr_f64 {
    ($v:expr) => {
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64(vec64![], None))
    };
}

// ======== Boolean ========

#[macro_export]
macro_rules! arr_bool {
    ($v:expr) => {
        $crate::Array::from_bool($crate::BooleanArray::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_bool($crate::BooleanArray::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_bool($crate::BooleanArray::from_vec64(vec64![], None))
    };
}

// ======== String ========

#[macro_export]
macro_rules! arr_str32 {
    ($v:expr) => {
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64(vec64![], None))
    };
}

#[cfg(feature = "large_string")]
#[macro_export]
macro_rules! arr_str64 {
    ($v:expr) => {
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64(vec64![], None))
    };
}

// ======== Categorical ========

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! arr_cat8 {
    ($v:expr) => {
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64(vec64![], None))
    };
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! arr_cat16 {
    ($v:expr) => {
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64(vec64![], None))
    };
}

#[macro_export]
macro_rules! arr_cat32 {
    ($v:expr) => {
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64(vec64![], None))
    };
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! arr_cat64 {
    ($v:expr) => {
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64($v, None))
    };
    ($($x:expr),+ $(,)?) => {{
        #[allow(unused_imports)]
        use $crate::Vec64;
        let temp_vec = vec64![$($x),+];
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64(temp_vec, None))
    }};
    () => {
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64(vec64![], None))
    };
}

// ======== Integer (signed) ========

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_i8_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int8($crate::IntegerArray::<i8>::from_vec64(vals, mask))
    }};
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_i16_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int16($crate::IntegerArray::<i16>::from_vec64(vals, mask))
    }};
}

#[macro_export]
macro_rules! arr_i32_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int32($crate::IntegerArray::<i32>::from_vec64(vals, mask))
    }};
}

#[macro_export]
macro_rules! arr_i64_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_int64($crate::IntegerArray::<i64>::from_vec64(vals, mask))
    }};
}

// ======== Integer (unsigned) ========

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_u8_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint8($crate::IntegerArray::<u8>::from_vec64(vals, mask))
    }};
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! arr_u16_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint16($crate::IntegerArray::<u16>::from_vec64(vals, mask))
    }};
}

#[macro_export]
macro_rules! arr_u32_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint32($crate::IntegerArray::<u32>::from_vec64(vals, mask))
    }};
}

#[macro_export]
macro_rules! arr_u64_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_uint64($crate::IntegerArray::<u64>::from_vec64(vals, mask))
    }};
}

// ======== Float ========

#[macro_export]
macro_rules! arr_f32_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_float32($crate::FloatArray::<f32>::from_vec64(vals, mask))
    }};
}

#[macro_export]
macro_rules! arr_f64_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_float64($crate::FloatArray::<f64>::from_vec64(vals, mask))
    }};
}

// ======== Boolean ========

#[macro_export]
macro_rules! arr_bool_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_option_values64($v);
        $crate::Array::from_bool($crate::BooleanArray::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_bool($crate::BooleanArray::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_option_values64(temp_vec);
        $crate::Array::from_bool($crate::BooleanArray::from_vec64(vals, mask))
    }};
}

// ======== String ========

#[macro_export]
macro_rules! arr_str32_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_string_option_values64_owned($v);
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64_owned(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_string_option_values64_owned(temp_vec);
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64_owned(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_string_option_values64_owned(temp_vec);
        $crate::Array::from_string32($crate::StringArray::<u32>::from_vec64_owned(vals, mask))
    }};
}

#[cfg(feature = "large_string")]
#[macro_export]
macro_rules! arr_str64_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_string_option_values64_owned($v);
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64_owned(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_string_option_values64_owned(temp_vec);
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64_owned(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_string_option_values64_owned(temp_vec);
        $crate::Array::from_string64($crate::StringArray::<u64>::from_vec64_owned(vals, mask))
    }};
}

// ======== Categorical ========

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! arr_cat8_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64($v);
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical8($crate::CategoricalArray::<u8>::from_vec64(vals, mask))
    }};
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! arr_cat16_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64($v);
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical16($crate::CategoricalArray::<u16>::from_vec64(vals, mask))
    }};
}

#[macro_export]
macro_rules! arr_cat32_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64($v);
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical32($crate::CategoricalArray::<u32>::from_vec64(vals, mask))
    }};
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! arr_cat64_opt {
    ($v:expr) => {{
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64($v);
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64(vals, mask))
    }};
    ($($x:expr),+ $(,)?) => {{
        let temp_vec = vec64![$($x),+];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64(vals, mask))
    }};
    () => {{
        let temp_vec = vec64![];
        let (vals, mask) = $crate::enums::array::extract_categorical_option_values64(temp_vec);
        $crate::Array::from_categorical64($crate::CategoricalArray::<u64>::from_vec64(vals, mask))
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::variants::boolean::BooleanArray;
    use crate::structs::variants::categorical::CategoricalArray;
    use crate::structs::variants::float::FloatArray;
    use crate::structs::variants::integer::IntegerArray;
    use crate::structs::variants::string::StringArray;
    use crate::traits::masked_array::MaskedArray;

    #[test]
    fn test_array_len_and_null() {
        assert_eq!(Array::Null.len(), 0);

        let arr = Array::from_int32(IntegerArray::<i32>::default());
        assert_eq!(arr.len(), 0);

        let mut arr = Array::from_int32(IntegerArray::<i32>::default());
        if let Array::NumericArray(NumericArray::Int32(ref mut a)) = arr {
            let a_mut = Arc::get_mut(a).expect("Array not uniquely owned");
            a_mut.push(7);
            a_mut.push(42);
        }
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_array_arrow_type() {
        assert_eq!(Array::Null.arrow_type(), ArrowType::Null);
        assert_eq!(
            Array::from_float32(FloatArray::<f32>::default()).arrow_type(),
            ArrowType::Float32
        );
        assert_eq!(
            Array::from_string32(StringArray::default()).arrow_type(),
            ArrowType::String
        );
        #[cfg(feature = "large_string")]
        assert_eq!(
            Array::from_string64(StringArray::default()).arrow_type(),
            ArrowType::LargeString
        );
        assert_eq!(
            Array::from_bool(BooleanArray::default()).arrow_type(),
            ArrowType::Boolean
        );

        let dict32 = Array::from_categorical32(CategoricalArray::<u32>::default());
        assert_eq!(
            dict32.arrow_type(),
            ArrowType::Dictionary(CategoricalIndexType::UInt32)
        );
    }

    #[test]
    fn test_array_is_nullable() {
        assert!(Array::Null.is_nullable());

        let arr = Array::from_int64(IntegerArray::<i64>::default());
        assert!(!arr.is_nullable());

        let mut arr = Array::from_int64(IntegerArray::<i64>::default());
        if let Array::NumericArray(NumericArray::Int64(ref mut a)) = arr {
            // only succeeds if the Arc is uniquely owned
            let a_mut = Arc::get_mut(a).expect("Array not uniquely owned");
            a_mut.push(100);
            a_mut.push(200);
            a_mut.push_null();
        }
        assert!(arr.is_nullable());

        let arr = Array::from_string32(StringArray::default());
        assert!(!arr.is_nullable());
    }

    #[test]
    fn test_data_ptr_and_len_for_primitives() {
        let mut arr = Array::from_int32(IntegerArray::<i32>::default());
        if let Array::NumericArray(NumericArray::Int32(ref mut a)) = arr {
            // Must have unique ownership to mutate the Arc contents.
            let a_mut = Arc::get_mut(a).expect("Array not uniquely owned");
            a_mut.push(123);
            a_mut.push(456);
        }
        let (ptr, len, sz) = arr.data_ptr_and_byte_len();
        assert!(!ptr.is_null());
        assert_eq!(len, 2);
        assert_eq!(sz, std::mem::size_of::<i32>());
    }

    #[test]
    fn test_data_ptr_and_len_for_str() {
        let mut str = StringArray::default();
        str.push_str("hello");
        str.push_str("world");
        let arr = Array::from_string32(str);
        let (ptr, len, sz) = arr.data_ptr_and_byte_len();
        assert!(!ptr.is_null());
        // byte length, not string count
        assert_eq!(len, 10);
        assert_eq!(sz, 1);
    }

    #[test]
    fn test_data_ptr_and_len_for_bool() {
        let mut bools = BooleanArray::default();
        for _ in 0..10 {
            bools.push(true);
        }
        let arr = Array::from_bool(bools);
        let (ptr, len, sz) = arr.data_ptr_and_byte_len();
        assert!(!ptr.is_null());
        assert_eq!(len, 10); // number of bits = logical elements
        assert_eq!(sz, 1);
    }

    #[test]
    fn test_data_ptr_and_len_for_dictionary() {
        let mut dict = CategoricalArray::<u32>::default();
        dict.push_str("a");
        dict.push_str("b");
        dict.push_str("a");
        let arr = Array::from_categorical32(dict);
        let (ptr, len, sz) = arr.data_ptr_and_byte_len();
        assert!(!ptr.is_null());
        assert_eq!(len, 3);
        assert_eq!(sz, std::mem::size_of::<u32>());
    }

    #[test]
    fn test_null_mask_ptr_and_len() {
        // Null variant has no mask
        assert!(Array::Null.null_mask_ptr_and_byte_len().is_none());

        let mut arr = IntegerArray::<i32>::default();
        arr.push(5);
        arr.push_null();
        let arr = Array::from_int32(arr);
        let mask = arr.null_mask_ptr_and_byte_len();
        assert!(mask.is_some());
        let (ptr, len) = mask.unwrap();
        assert!(!ptr.is_null());
        assert!(len > 0);

        let arr = Array::from_float64(FloatArray::<f64>::default());
        assert!(arr.null_mask_ptr_and_byte_len().is_none());
    }

    #[test]
    fn test_offsets_ptr_and_len() {
        let arr = Array::from_int32(IntegerArray::<i32>::default());
        assert!(arr.offsets_ptr_and_len().is_none());

        let mut str = StringArray::default();
        str.push_str("a");
        str.push_str("bc");
        let arr = Array::from_string32(str);
        let opt = arr.offsets_ptr_and_len();
        assert!(opt.is_some());
        let (ptr, len) = opt.unwrap();
        assert!(!ptr.is_null());
        assert_eq!(len, 12); // 3 offsets (u32) == 12 bytes
    }

    #[test]
    fn test_enum_variant_consistency() {
        let i32arr = IntegerArray::<i32>::default();
        let a = Array::from_int32(i32arr.clone());
        if let Array::NumericArray(NumericArray::Int32(ref arr2)) = a {
            assert_eq!(arr2.as_ref(), &i32arr);
        } else {
            panic!("Not the right variant");
        }
    }

    #[test]
    fn test_default_variant_is_null() {
        let a = Array::default();
        assert!(matches!(a, Array::Null));
        assert_eq!(a.len(), 0);
        assert!(a.is_nullable());
    }

    
    #[test]
    fn test_array_enum_slice() {
        use crate::{Array, ArrayVT};

        let mut bool_arr = BooleanArray::default();
        bool_arr.push(true);
        bool_arr.push(false);
        bool_arr.push(true);
        bool_arr.push(false);

        let array = Array::from_bool(bool_arr);
        let view: ArrayVT = (&array, 1, 2);

        match view.0 {
            Array::BooleanArray(inner) => {
                assert_eq!(inner.get(view.1), Some(false));
                assert_eq!(inner.get(view.1 + 1), Some(true));
            }
            _ => panic!("Expected Bool variant"),
        }
    }

    #[test]
    fn test_num_from_int_array() {
        let arr = IntegerArray::<i32>::from_slice(&[1, 2, 3]);
        let array = Array::from_int32(arr.clone());
        let out = array.num();
        match out {
            NumericArray::Int32(ref a) => assert_eq!(a.data, arr.data),
            _ => panic!("Expected Int32"),
        }
    }

    #[test]
    fn test_num_from_bool_array() {
        let mut arr = BooleanArray::default();
        arr.push(true);
        arr.push(false);
        arr.push_null();
        let array = Array::from_bool(arr.clone());
        let out = array.num();
        match out {
            NumericArray::Int32(ref a) => assert_eq!(&a.data[..], &[1, 0, 0]),
            _ => panic!("Expected Int32"),
        }
    }

    #[test]
    fn test_num_from_string_array() {
        let arr = StringArray::<u32>::from_slice(&["123", "xyz", ""]);
        let array = Array::from_string32(arr.clone());
        let out = array.num();
        match out {
            NumericArray::Int32(ref a) => {
                // "123" parses, "xyz" and "" are invalid, thus 0 and marked null.
                assert_eq!(&a.data[..], &[123, 0, 0]);
                let mask = a.null_mask.as_ref().expect("Should have a null mask");
                assert_eq!(mask.get(0), true);
                assert_eq!(mask.get(1), false);
                assert_eq!(mask.get(2), false);
            }
            _ => panic!("Expected Int32"),
        }
    }

    #[test]
    fn test_num_from_categorical_array() {
        let arr = StringArray::<u32>::from_slice(&["42", "hi", "999"]);
        let cat = arr.to_categorical_array();
        let array = Array::from_categorical32(cat.clone());
        let out = array.num();
        match out {
            NumericArray::Int32(ref a) => {
                // unique_values: ["42", "hi", "999"]
                // .data indices: [0, 0, 1, 2, ...]
                // Only "42" and "999" parse as i32
                let expected_vals: Vec<i32> = cat
                    .unique_values
                    .iter()
                    .map(|s| s.parse::<i32>().unwrap_or(0))
                    .collect();
                let expected_mask: Vec<bool> = cat
                    .unique_values
                    .iter()
                    .map(|s| s.parse::<i32>().is_ok())
                    .collect();
                // a.data contains, for each code, expected_vals[code]
                for (ix, &cat_idx) in cat.data.iter().enumerate() {
                    assert_eq!(a.data[ix], expected_vals[cat_idx as usize]);
                    let mask = a.null_mask.as_ref().unwrap();
                    assert_eq!(mask.get(ix), expected_mask[cat_idx as usize]);
                }
            }
            _ => panic!("Expected Int32"),
        }
    }

    #[test]
    fn test_str_from_bool_array() {
        let mut arr = BooleanArray::default();
        arr.push(true);
        arr.push(false);
        arr.push_null();
        let array = Array::from_bool(arr);
        let out = array.str();
        match out {
            TextArray::String32(ref s) => {
                let got: Vec<String> = (0..s.len())
                    .map(|i| s.get_str(i).unwrap_or("").to_string())
                    .collect();
                assert_eq!(&got[..], &["true", "false", ""]);
            }
            _ => panic!("Expected String32"),
        }
    }

    #[test]
    fn test_str_from_int_array() {
        let arr = IntegerArray::<i32>::from_slice(&[5, 0, -10]);
        let array = Array::from_int32(arr);
        let out = array.str();
        match out {
            TextArray::String32(ref s) => {
                let got: Vec<_> = (0..s.len())
                    .map(|i| s.get_str(i).unwrap_or("").to_string())
                    .collect();
                assert_eq!(&got[..], &["5", "0", "-10"]);
            }
            _ => panic!("Expected String32"),
        }
    }

    #[test]
    fn test_bool_from_int_array() {
        let arr = IntegerArray::<i32>::from_slice(&[0, 1, -2, 0]);
        let array = Array::from_int32(arr);
        let out = array.bool();
        let values: Vec<_> = (0..out.len).map(|i| out.get(i)).collect();
        assert_eq!(
            &values[..],
            &[Some(false), Some(true), Some(true), Some(false)]
        );
    }

    #[test]
    fn test_bool_from_string_array() {
        let arr = StringArray::<u32>::from_slice(&["True", "0", "false", "abc", ""]);
        let array = Array::from_string32(arr);
        let out = array.bool();
        let values: Vec<_> = (0..out.len).map(|i| out.get(i)).collect();
        assert_eq!(
            &values[..],
            &[
                Some(true),
                Some(false),
                Some(false),
                Some(true),
                Some(false)
            ]
        );
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_num_from_datetime_i32() {
        use crate::TimeUnit;

        let dt = DatetimeArray::<i32>::from_slice(&[123, 456, 789], Some(TimeUnit::Milliseconds));
        let array = Array::from_datetime_i32(dt.clone());
        let out = array.num();
        match out {
            NumericArray::Int32(ref a) => assert_eq!(&a.data[..], &[123, 456, 789]),
            _ => panic!("Expected Int32"),
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_dt_from_int_array() {
        let arr = IntegerArray::<i32>::from_slice(&[1000, 2000]);
        let array = Array::from_int32(arr);
        let out = array.dt();
        match out {
            TemporalArray::Datetime64(ref dt) => assert_eq!(&dt.data[..], &[1000, 2000]),
            _ => panic!("Expected DatetimeI64"),
        }
    }

    #[cfg(feature = "chrono")]
    #[test]
    fn test_dt_from_text_array_parsing() {
        let arr = StringArray::<u32>::from_slice(&[
            "2023-01-01T00:00:00Z",
            "foo",
            "2020-06-30T12:00:00Z",
        ]);
        let array = Array::from_string32(arr);
        println!("{:?}", array);
        let out = array.dt();
        println!("{:?}", out);
        match out {
            TemporalArray::Datetime64(ref dt) => {
                assert_eq!(dt.len(), 3);
                let valid = dt.null_mask.as_ref().unwrap();
                assert!(valid.get(0), "First date should be valid");
                assert!(!valid.get(1), "Second date ('foo') should be invalid/null");
                assert!(valid.get(2), "Third date should be valid");
            }
            _ => panic!("Expected DatetimeI64"),
        }
    }

    #[test]
    fn test_null_cases() {
        let array = Array::Null;
        assert_eq!(array.clone().num(), NumericArray::Null);
        assert_eq!(array.clone().str(), TextArray::Null);
        assert_eq!(array.clone().bool(), BooleanArray::default().into());
        #[cfg(feature = "datetime")]
        assert_eq!(array.dt(), TemporalArray::Null);
    }
}

#[cfg(test)]
mod macro_tests {
    use crate::{Array, Bitmask, MaskedArray, NumericArray, TextArray, Vec64, vec64};

    // helper for checking null masks
    fn assert_mask(mask: &Option<Bitmask>, expected: &[bool]) {
        if expected.iter().all(|&b| b) {
            assert!(mask.is_none(), "Expected no null mask");
        } else {
            let m = mask.as_ref().expect("Expected Some(null_mask)");
            for (i, &val) in expected.iter().enumerate() {
                assert_eq!(m.get(i), val, "Mask mismatch at position {}", i);
            }
        }
    }

    // ===== numeric types =====

    #[test]
    fn arr_i32_vec64_dense() {
        let v = vec64![1i32, 2, 3];
        let arr = arr_i32!(v);
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1, 2, 3]);
            assert_mask(&a.null_mask, &[true, true, true]);
        } else {
            panic!("arr_i32!(Vec64) wrong variant");
        }
    }

    #[test]
    fn arr_i32_vec64_opt() {
        let v: Vec64<Option<i32>> = vec64![Some(1i32), None::<i32>, Some(3)];
        let arr = arr_i32_opt!(v);
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1, 0, 3]);
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("arr_i32_opt!(Vec64<Option>) wrong variant");
        }
    }

    #[test]
    fn arr_f64_vec64_dense() {
        let v = vec64![1.1f64, 2.2, 3.3];
        let arr = arr_f64!(v);
        if let Array::NumericArray(NumericArray::Float64(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1.1, 2.2, 3.3]);
            assert_mask(&a.null_mask, &[true, true, true]);
        } else {
            panic!("arr_f64!(Vec64) wrong variant");
        }
    }

    #[test]
    fn arr_f64_vec64_opt() {
        let v = vec64![Some(1.5f64), None::<f64>, Some(-2.5)];
        let arr = arr_f64_opt!(v);
        if let Array::NumericArray(NumericArray::Float64(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1.5, 0.0, -2.5]);
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("arr_f64_opt!(Vec64<Option>) wrong variant");
        }
    }

    // ===== bool =====

    #[test]
    fn arr_bool_vec64_dense() {
        let v = vec64![true, false, true];
        let arr = arr_bool!(v);
        if let Array::BooleanArray(a) = arr {
            assert_eq!(a.get(0), Some(true));
            assert_eq!(a.get(1), Some(false));
            assert_eq!(a.get(2), Some(true));
            assert_mask(&a.null_mask, &[true, true, true]);
        } else {
            panic!("arr_bool!(Vec64) wrong variant");
        }
    }

    #[test]
    fn arr_bool_vec64_opt() {
        let v = vec64![Some(true), None::<bool>, Some(false)];
        let arr = arr_bool_opt!(v);
        if let Array::BooleanArray(a) = arr {
            assert_eq!(a.get(0), Some(true));
            assert_eq!(a.get(1), None);
            assert_eq!(a.get(2), Some(false));
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("arr_bool_opt!(Vec64<Option>) wrong variant");
        }
    }

    // ===== string =====

    #[test]
    fn arr_str32_vec64_dense() {
        let v = vec64!["a", "b", "c"];
        let arr = arr_str32!(v);
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.get_str(0), Some("a"));
            assert_eq!(a.get_str(1), Some("b"));
            assert_eq!(a.get_str(2), Some("c"));
            assert_mask(&a.null_mask, &[true, true, true]);
        } else {
            panic!("arr_str32!(Vec64) wrong variant");
        }
    }

    #[test]
    fn arr_str32_vec64_opt() {
        let v = vec64![Some("x"), None::<&str>, Some("y")];
        let arr = arr_str32_opt!(v);
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.get_str(0), Some("x"));
            assert_eq!(a.get_str(1), None);
            assert_eq!(a.get_str(2), Some("y"));
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("arr_str32_opt!(Vec64<Option>) wrong variant");
        }
    }

    // ===== categorical =====

    #[test]
    fn arr_cat32_vec64_dense() {
        let v = vec64!["red", "green", "red"];
        let arr = arr_cat32!(v);
        if let Array::TextArray(TextArray::Categorical32(a)) = arr {
            assert_eq!(a.get_str(0), Some("red"));
            assert_eq!(a.get_str(1), Some("green"));
            assert_eq!(a.get_str(2), Some("red"));
            assert_mask(&a.null_mask, &[true, true, true]);
        } else {
            panic!("arr_cat32!(Vec64) wrong variant");
        }
    }

    #[test]
    fn arr_cat32_vec64_opt() {
        let v = vec64![Some("red"), None::<&str>, Some("blue")];
        let arr = arr_cat32_opt!(v);
        if let Array::TextArray(TextArray::Categorical32(a)) = arr {
            assert_eq!(a.get_str(0), Some("red"));
            assert_eq!(a.get_str(1), None);
            assert_eq!(a.get_str(2), Some("blue"));
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("arr_cat32_opt!(Vec64<Option>) wrong variant");
        }
    }

    // ===== All numeric types =====

    #[cfg(feature = "extended_numeric_types")]
    #[test]
    fn test_all_integer_types() {
        // i8
        let arr = arr_i8!(1i8, 2, 3);
        if let Array::NumericArray(NumericArray::Int8(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1, 2, 3]);
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_i8_opt!(Some(1i8), None::<i8>, Some(3));
        if let Array::NumericArray(NumericArray::Int8(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1, 0, 3]);
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("Wrong variant");
        }

        // i16
        let arr = arr_i16!(100i16, 200, 300);
        if let Array::NumericArray(NumericArray::Int16(a)) = arr {
            assert_eq!(a.data.as_slice(), &[100, 200, 300]);
        } else {
            panic!("Wrong variant");
        }

        // u8
        let arr = arr_u8!(10u8, 20, 30);
        if let Array::NumericArray(NumericArray::UInt8(a)) = arr {
            assert_eq!(a.data.as_slice(), &[10, 20, 30]);
        } else {
            panic!("Wrong variant");
        }

        // u16
        let arr = arr_u16!(1000u16, 2000, 3000);
        if let Array::NumericArray(NumericArray::UInt16(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1000, 2000, 3000]);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_standard_integer_types() {
        // i32
        let arr = arr_i32!(vec64![1i32, 2, 3]);
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1, 2, 3]);
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_i32_opt!(vec64![Some(1i32), None::<i32>, Some(3)]);
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("Wrong variant");
        }

        // i64
        let arr = arr_i64!(vec64![100i64, 200, 300]);
        if let Array::NumericArray(NumericArray::Int64(a)) = arr {
            assert_eq!(a.data.as_slice(), &[100, 200, 300]);
        } else {
            panic!("Wrong variant");
        }

        // u32
        let arr = arr_u32!(vec64![1000u32, 2000, 3000]);
        if let Array::NumericArray(NumericArray::UInt32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1000, 2000, 3000]);
        } else {
            panic!("Wrong variant");
        }

        // u64
        let arr = arr_u64!(vec64![10000u64, 20000, 30000]);
        if let Array::NumericArray(NumericArray::UInt64(a)) = arr {
            assert_eq!(a.data.as_slice(), &[10000, 20000, 30000]);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_float_types() {
        // f32
        let arr = arr_f32!(vec64![1.1f32, 2.2, 3.3]);
        if let Array::NumericArray(NumericArray::Float32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1.1, 2.2, 3.3]);
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_f32_opt!(vec64![Some(1.5f32), None::<f32>, Some(-2.5)]);
        if let Array::NumericArray(NumericArray::Float32(a)) = arr {
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("Wrong variant");
        }

        // f64
        let arr = arr_f64!(vec64![10.1f64, 20.2, 30.3]);
        if let Array::NumericArray(NumericArray::Float64(a)) = arr {
            assert_eq!(a.data.as_slice(), &[10.1, 20.2, 30.3]);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_boolean_types() {
        let arr = arr_bool!(vec64![true, false, true]);
        if let Array::BooleanArray(a) = arr {
            assert_eq!(a.get(0), Some(true));
            assert_eq!(a.get(1), Some(false));
            assert_eq!(a.get(2), Some(true));
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_bool_opt!(vec64![Some(true), None::<bool>, Some(false)]);
        if let Array::BooleanArray(a) = arr {
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_string_types() {
        let arr = arr_str32!(vec64!["hello", "world", "test"]);
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.get_str(0), Some("hello"));
            assert_eq!(a.get_str(1), Some("world"));
            assert_eq!(a.get_str(2), Some("test"));
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_str32_opt!(vec64![Some("x"), None::<&str>, Some("y")]);
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.get_str(0), Some("x"));
            assert_eq!(a.get_str(1), None);
            assert_eq!(a.get_str(2), Some("y"));
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("Wrong variant");
        }

        #[cfg(feature = "large_string")]
        {
            let arr = arr_str64!(vec64!["large", "string", "test"]);
            if let Array::TextArray(TextArray::String64(a)) = arr {
                assert_eq!(a.get_str(0), Some("large"));
            } else {
                panic!("Wrong variant");
            }
        }
    }

    #[test]
    fn test_categorical_types() {
        let arr = arr_cat32!(vec64!["red", "green", "red"]);
        if let Array::TextArray(TextArray::Categorical32(a)) = arr {
            assert_eq!(a.get_str(0), Some("red"));
            assert_eq!(a.get_str(1), Some("green"));
            assert_eq!(a.get_str(2), Some("red"));
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_cat32_opt!(vec64![Some("red"), None::<&str>, Some("blue")]);
        if let Array::TextArray(TextArray::Categorical32(a)) = arr {
            assert_eq!(a.get_str(0), Some("red"));
            assert_eq!(a.get_str(1), None);
            assert_eq!(a.get_str(2), Some("blue"));
            assert_mask(&a.null_mask, &[true, false, true]);
        } else {
            panic!("Wrong variant");
        }

        #[cfg(feature = "extended_categorical")]
        {
            let arr = arr_cat8!(vec64!["a", "b", "a"]);
            if let Array::TextArray(TextArray::Categorical8(a)) = arr {
                assert_eq!(a.get_str(0), Some("a"));
            } else {
                panic!("Wrong variant");
            }

            let arr = arr_cat16!(vec64!["x", "y", "x"]);
            if let Array::TextArray(TextArray::Categorical16(a)) = arr {
                assert_eq!(a.get_str(0), Some("x"));
            } else {
                panic!("Wrong variant");
            }

            let arr = arr_cat64!(vec64!["alpha", "beta", "alpha"]);
            if let Array::TextArray(TextArray::Categorical64(a)) = arr {
                assert_eq!(a.get_str(0), Some("alpha"));
            } else {
                panic!("Wrong variant");
            }
        }
    }

    #[test]
    fn test_empty_arrays() {
        let arr = arr_i32!(vec64![]);
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.len(), 0);
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_str32!(vec64![]);
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.len(), 0);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_all_nulls() {
        let arr = arr_i32_opt!(vec64![None::<i32>, None::<i32>, None::<i32>]);
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.null_count(), 3);
            assert_mask(&a.null_mask, &[false, false, false]);
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_str32_opt!(vec64![None::<&str>, None::<&str>, None::<&str>]);
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.null_count(), 3);
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_literal_syntax_no_nulls() {
        // Numeric types
        let arr = arr_i32![1, 2, 3, 4];
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1, 2, 3, 4]);
            assert!(a.null_mask.is_none());
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_f64![0.5, 1.5, 2.5];
        if let Array::NumericArray(NumericArray::Float64(a)) = arr {
            assert_eq!(a.data.as_slice(), &[0.5, 1.5, 2.5]);
            assert!(a.null_mask.is_none());
        } else {
            panic!("Wrong variant");
        }

        // Boolean
        let arr = arr_bool![true, false, true];
        if let Array::BooleanArray(a) = arr {
            assert_eq!(a.get(0), Some(true));
            assert_eq!(a.get(1), Some(false));
            assert_eq!(a.get(2), Some(true));
            assert!(a.null_mask.is_none());
        } else {
            panic!("Wrong variant");
        }

        // String
        let arr = arr_str32!["a", "b", "c"];
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.get_str(0), Some("a"));
            assert_eq!(a.get_str(1), Some("b"));
            assert_eq!(a.get_str(2), Some("c"));
            assert!(a.null_mask.is_none());
        } else {
            panic!("Wrong variant");
        }

        // Categorical
        let arr = arr_cat32!["x", "y", "x", "z"];
        if let Array::TextArray(TextArray::Categorical32(a)) = arr {
            assert_eq!(a.get_str(0), Some("x"));
            assert_eq!(a.get_str(1), Some("y"));
            assert_eq!(a.get_str(2), Some("x"));
            assert_eq!(a.get_str(3), Some("z"));
            assert!(a.null_mask.is_none());
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_literal_syntax_with_nulls() {
        // Test explicit null handling macros
        let arr = arr_i32_opt![Some(1), None::<i32>, Some(3)];
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[1, 0, 3]);
            assert!(a.null_mask.is_some());
            assert_eq!(a.get(0), Some(1));
            assert_eq!(a.get(1), None);
            assert_eq!(a.get(2), Some(3));
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_str32_opt![Some("hello"), None::<&str>, Some("world")];
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.get_str(0), Some("hello"));
            assert_eq!(a.get_str(1), None);
            assert_eq!(a.get_str(2), Some("world"));
            assert!(a.null_mask.is_some());
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_bool_opt![Some(true), None::<bool>, Some(false)];
        if let Array::BooleanArray(a) = arr {
            assert_eq!(a.get(0), Some(true));
            assert_eq!(a.get(1), None);
            assert_eq!(a.get(2), Some(false));
            assert!(a.null_mask.is_some());
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_cat32_opt![Some("red"), None::<&str>, Some("blue")];
        if let Array::TextArray(TextArray::Categorical32(a)) = arr {
            assert_eq!(a.get_str(0), Some("red"));
            assert_eq!(a.get_str(1), None);
            assert_eq!(a.get_str(2), Some("blue"));
            assert!(a.null_mask.is_some());
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_single_elements() {
        // TODO: This currently needs vec64 wrapping
        let arr = arr_i32![vec64![42]];
        if let Array::NumericArray(NumericArray::Int32(a)) = arr {
            assert_eq!(a.data.as_slice(), &[42]);
            assert!(a.null_mask.is_none());
        } else {
            panic!("Wrong variant");
        }

        let arr = arr_str32![vec64!["hello"]];
        if let Array::TextArray(TextArray::String32(a)) = arr {
            assert_eq!(a.get_str(0), Some("hello"));
            assert!(a.null_mask.is_none());
        } else {
            panic!("Wrong variant");
        }
    }

    #[test]
    fn test_mixed_usage() {
        // Test that both syntaxes work
        let v = vec64![1, 2, 3];
        let arr1 = arr_i32!(v);
        let arr2 = arr_i32![1, 2, 3];

        if let (
            Array::NumericArray(NumericArray::Int32(a1)),
            Array::NumericArray(NumericArray::Int32(a2)),
        ) = (arr1, arr2)
        {
            assert_eq!(a1.data.as_slice(), a2.data.as_slice());
        } else {
            panic!("Wrong variants");
        }
    }
}
