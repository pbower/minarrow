//! # NumericArray Module
//! 
//! NumericArray unifies all integer and floating-point arrays 
//! into a single enum for standardised numeric operations.
//!   
//! ## Features
//! - direct variant access
//! - zero-cost casts when the type is known
//! - lossless conversions between integer and float types. 
//! - simplifies function signatures by accepting `impl Into<NumericArray>`
//! - centralises dispatch
//! - preserves SIMD-aligned buffers across all numeric variants.

use std::{fmt::{Display, Formatter}, sync::Arc};

use crate::enums::error::MinarrowError;
use crate::{Bitmask, FloatArray, IntegerArray, MaskedArray};
use crate::{BooleanArray, StringArray};

/// # NumericArray
/// 
/// Unifying numerical array container
/// 
/// ## Purpose
/// Exists to unify numerical operations,
/// simplify API's and streamline user ergonomics.
/// 
/// ## Usage:
/// - It is accessible from `Array` using `.num()`,
/// and provides typed variant access via for e.g.,
/// `.i64()`, so one can drill down to the required
/// granularity via `myarr.num().i64()`
/// - This streamlines function implementations,
/// and, despite the additional `enum` layer,
/// matching lanes in many real-world scenarios.
/// This is because one can for e.g., unify a 
/// function signature with `impl Into<NumericArray>`,
/// and all of the subtypes, plus `Array` and `NumericalArray`,
/// all qualify. 
/// - Additionally, you can then use one `Integer` implementation
/// on the enum dispatch arm for all `Integer` variants, or,
/// in many cases, for the entire numeric arm when they are the same.
/// 
/// ### Typecasting behaviour
/// - If the enum already holds the given type *(which should be known at compile-time)*,
/// then using accessors like `.i32()` is zero-cost, as it transfers ownership.
/// - If you want to keep the original, of course use `.clone()` beforehand.
/// - If you use an accessor to a different base type, e.g., `.f32()` when it's a
/// `.int32()` already in the enum, it will convert it. Therefore, be mindful
/// of performance when this occurs.
/// 
/// ## Also see:
/// - Under [crate::traits::type_unions] , we additionally
/// include minimal `Integer`, `Float`, `Numeric` and `Primitive` traits that 
/// for which the base Rust primitive types already qualify. 
/// These are loose wrappers over the `num-traits` crate to help improve
/// type ergonomics when traits are required, but without requiring
/// any downcasting.
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub enum NumericArray {
    #[cfg(feature = "extended_numeric_types")]
    Int8(Arc<IntegerArray<i8>>),
    #[cfg(feature = "extended_numeric_types")]
    Int16(Arc<IntegerArray<i16>>),
    Int32(Arc<IntegerArray<i32>>),
    Int64(Arc<IntegerArray<i64>>),
    #[cfg(feature = "extended_numeric_types")]
    UInt8(Arc<IntegerArray<u8>>),
    #[cfg(feature = "extended_numeric_types")]
    UInt16(Arc<IntegerArray<u16>>),
    UInt32(Arc<IntegerArray<u32>>),
    UInt64(Arc<IntegerArray<u64>>),
    Float32(Arc<FloatArray<f32>>),
    Float64(Arc<FloatArray<f64>>),
    #[default]
    Null // Default Marker for mem::take
}

impl NumericArray {
    /// Returns the logical length of the numeric array.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(arr) => arr.len(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(arr) => arr.len(),
            NumericArray::Int32(arr) => arr.len(),
            NumericArray::Int64(arr) => arr.len(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(arr) => arr.len(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(arr) => arr.len(),
            NumericArray::UInt32(arr) => arr.len(),
            NumericArray::UInt64(arr) => arr.len(),
            NumericArray::Float32(arr) => arr.len(),
            NumericArray::Float64(arr) => arr.len(),
            NumericArray::Null => 0
        }
    }

    /// Returns the underlying null mask, if any.
    #[inline]
    pub fn null_mask(&self) -> Option<&Bitmask> {
        match self {
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
            NumericArray::Null => None
        }
    }

    /// Appends all values (and null mask if present) from `other` into `self`.
    ///
    /// Panics if the two arrays are of different variants or incompatible types.
    ///
    /// This function uses copy-on-write semantics for arrays wrapped in `Arc`.
    /// If `self` is the only owner of its data, appends are performed in place without copying.
    /// If the array data is shared (`Arc` reference count > 1), the data is first cloned
    /// (so the mutation does not affect other owners), and the append is then performed on the unique copy.
    ///
    /// This ensures that calling `append_array` never mutates data referenced elsewhere,
    /// but also avoids unnecessary cloning when the data is uniquely owned.
    pub fn append_array(&mut self, other: &Self) {
        match (self, other) {
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::Int8(a), NumericArray::Int8(b)) => Arc::make_mut(a).append_array(b),
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::Int16(a), NumericArray::Int16(b)) => Arc::make_mut(a).append_array(b),
            (NumericArray::Int32(a), NumericArray::Int32(b)) => Arc::make_mut(a).append_array(b),
            (NumericArray::Int64(a), NumericArray::Int64(b)) => Arc::make_mut(a).append_array(b),

            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::UInt8(a), NumericArray::UInt8(b)) => Arc::make_mut(a).append_array(b),
            #[cfg(feature = "extended_numeric_types")]
            (NumericArray::UInt16(a), NumericArray::UInt16(b)) => Arc::make_mut(a).append_array(b),
            (NumericArray::UInt32(a), NumericArray::UInt32(b)) => Arc::make_mut(a).append_array(b),
            (NumericArray::UInt64(a), NumericArray::UInt64(b)) => Arc::make_mut(a).append_array(b),

            (NumericArray::Float32(a), NumericArray::Float32(b)) => {
                Arc::make_mut(a).append_array(b)
            }
            (NumericArray::Float64(a), NumericArray::Float64(b)) => {
                Arc::make_mut(a).append_array(b)
            }

            (NumericArray::Null, NumericArray::Null) => (),
            (lhs, rhs) => panic!("Cannot append {:?} into {:?}", rhs, lhs)
        }
    }

    /// Convert to IntegerArray<i32> using From/TryFrom as appropriate per conversion.
    pub fn i32(self) -> Result<IntegerArray<i32>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(IntegerArray::<i32>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(IntegerArray::<i32>::from(&*a)),
            NumericArray::Int32(a) => match Arc::try_unwrap(a) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            NumericArray::Int64(a) => Ok(IntegerArray::<i32>::try_from(&*a)?),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(IntegerArray::<i32>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(IntegerArray::<i32>::from(&*a)),
            NumericArray::UInt32(a) => Ok(IntegerArray::<i32>::try_from(&*a)?),
            NumericArray::UInt64(a) => Ok(IntegerArray::<i32>::try_from(&*a)?),
            NumericArray::Float32(a) => Ok(IntegerArray::<i32>::try_from(&*a)?),
            NumericArray::Float64(a) => Ok(IntegerArray::<i32>::try_from(&*a)?),
            NumericArray::Null => Err(MinarrowError::NullError { message: None })
        }
    }

    /// Convert to IntegerArray<i64> using From/TryFrom as appropriate per conversion.
    pub fn i64(self) -> Result<IntegerArray<i64>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(IntegerArray::<i64>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(IntegerArray::<i64>::from(&*a)),
            NumericArray::Int32(a) => Ok(IntegerArray::<i64>::from(&*a)),
            NumericArray::Int64(a) => match Arc::try_unwrap(a) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(IntegerArray::<i64>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(IntegerArray::<i64>::from(&*a)),
            NumericArray::UInt32(a) => Ok(IntegerArray::<i64>::from(&*a)),
            NumericArray::UInt64(a) => Ok(IntegerArray::<i64>::try_from(&*a)?),
            NumericArray::Float32(a) => Ok(IntegerArray::<i64>::try_from(&*a)?),
            NumericArray::Float64(a) => Ok(IntegerArray::<i64>::try_from(&*a)?),
            NumericArray::Null => Err(MinarrowError::NullError { message: None })
        }
    }

    /// Convert to IntegerArray<u32> using From/TryFrom as appropriate per conversion.
    pub fn u32(self) -> Result<IntegerArray<u32>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(IntegerArray::<u32>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(IntegerArray::<u32>::from(&*a)),
            NumericArray::Int32(a) => Ok(IntegerArray::<u32>::try_from(&*a)?),
            NumericArray::Int64(a) => Ok(IntegerArray::<u32>::try_from(&*a)?),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(IntegerArray::<u32>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(IntegerArray::<u32>::from(&*a)),
            NumericArray::UInt32(a) => match Arc::try_unwrap(a) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            NumericArray::UInt64(a) => Ok(IntegerArray::<u32>::try_from(&*a)?),
            NumericArray::Float32(a) => Ok(IntegerArray::<u32>::try_from(&*a)?),
            NumericArray::Float64(a) => Ok(IntegerArray::<u32>::try_from(&*a)?),
            NumericArray::Null => Err(MinarrowError::NullError { message: None })
        }
    }

    /// Convert to IntegerArray<u64> using From/TryFrom as appropriate per conversion.
    pub fn u64(self) -> Result<IntegerArray<u64>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(IntegerArray::<u64>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(IntegerArray::<u64>::from(&*a)),
            NumericArray::Int32(a) => Ok(IntegerArray::<u64>::from(&*a)),
            NumericArray::Int64(a) => Ok(IntegerArray::<u64>::try_from(&*a)?),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(IntegerArray::<u64>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(IntegerArray::<u64>::from(&*a)),
            NumericArray::UInt32(a) => Ok(IntegerArray::<u64>::from(&*a)),
            NumericArray::UInt64(a) => match Arc::try_unwrap(a) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            NumericArray::Float32(a) => Ok(IntegerArray::<u64>::try_from(&*a)?),
            NumericArray::Float64(a) => Ok(IntegerArray::<u64>::try_from(&*a)?),
            NumericArray::Null => Err(MinarrowError::NullError { message: None })
        }
    }

    /// Convert to FloatArray<f32> using From.
    pub fn f32(self) -> Result<FloatArray<f32>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(FloatArray::<f32>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(FloatArray::<f32>::from(&*a)),
            NumericArray::Int32(a) => Ok(FloatArray::<f32>::from(&*a)),
            NumericArray::Int64(a) => Ok(FloatArray::<f32>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(FloatArray::<f32>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(FloatArray::<f32>::from(&*a)),
            NumericArray::UInt32(a) => Ok(FloatArray::<f32>::from(&*a)),
            NumericArray::UInt64(a) => Ok(FloatArray::<f32>::from(&*a)),
            NumericArray::Float32(a) => match Arc::try_unwrap(a) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            NumericArray::Float64(a) => Ok(FloatArray::<f32>::from(&*a)),
            NumericArray::Null => Err(MinarrowError::NullError { message: None })
        }
    }

    /// Convert to FloatArray<f64> using From.
    pub fn f64(self) -> Result<FloatArray<f64>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(FloatArray::<f64>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(FloatArray::<f64>::from(&*a)),
            NumericArray::Int32(a) => Ok(FloatArray::<f64>::from(&*a)),
            NumericArray::Int64(a) => Ok(FloatArray::<f64>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(FloatArray::<f64>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(FloatArray::<f64>::from(&*a)),
            NumericArray::UInt32(a) => Ok(FloatArray::<f64>::from(&*a)),
            NumericArray::UInt64(a) => Ok(FloatArray::<f64>::from(&*a)),
            NumericArray::Float32(a) => Ok(FloatArray::<f64>::from(&*a)),
            NumericArray::Float64(a) => match Arc::try_unwrap(a) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            NumericArray::Null => Err(MinarrowError::NullError { message: None })
        }
    }

    /// Converts to BooleanArray<u8>. 
    /// 
    /// All non-zero values become `true`, but the null mask is preserved.
    pub fn bool(self) -> Result<BooleanArray<u8>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(BooleanArray::<u8>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(BooleanArray::<u8>::from(&*a)),
            NumericArray::Int32(a) => Ok(BooleanArray::<u8>::from(&*a)),
            NumericArray::Int64(a) => Ok(BooleanArray::<u8>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(BooleanArray::<u8>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(BooleanArray::<u8>::from(&*a)),
            NumericArray::UInt32(a) => Ok(BooleanArray::<u8>::from(&*a)),
            NumericArray::UInt64(a) => Ok(BooleanArray::<u8>::from(&*a)),
            NumericArray::Float32(a) => Ok(BooleanArray::<u8>::from(&*a)),
            NumericArray::Float64(a) => Ok(BooleanArray::<u8>::from(&*a)),
            NumericArray::Null => Err(MinarrowError::NullError { message: None })
        }
    }

    /// Converts to StringArray<u32> by formatting each value as string. 
    /// 
    /// Preserves Null mask.
    pub fn str(self) -> Result<StringArray<u32>, MinarrowError> {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => Ok(StringArray::<u32>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => Ok(StringArray::<u32>::from(&*a)),
            NumericArray::Int32(a) => Ok(StringArray::<u32>::from(&*a)),
            NumericArray::Int64(a) => Ok(StringArray::<u32>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => Ok(StringArray::<u32>::from(&*a)),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => Ok(StringArray::<u32>::from(&*a)),
            NumericArray::UInt32(a) => Ok(StringArray::<u32>::from(&*a)),
            NumericArray::UInt64(a) => Ok(StringArray::<u32>::from(&*a)),
            NumericArray::Float32(a) => Ok(StringArray::<u32>::from(&*a)),
            NumericArray::Float64(a) => Ok(StringArray::<u32>::from(&*a)),
            NumericArray::Null => Err(MinarrowError::NullError { message: None })
        }
    }
}

impl Display for NumericArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(arr) =>
                write_numeric_array_with_header(f, "Int8", arr.as_ref()),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(arr) =>
                write_numeric_array_with_header(f, "Int16", arr.as_ref()),
            NumericArray::Int32(arr) =>
                write_numeric_array_with_header(f, "Int32", arr.as_ref()),
            NumericArray::Int64(arr) =>
                write_numeric_array_with_header(f, "Int64", arr.as_ref()),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(arr) =>
                write_numeric_array_with_header(f, "UInt8", arr.as_ref()),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(arr) =>
                write_numeric_array_with_header(f, "UInt16", arr.as_ref()),
            NumericArray::UInt32(arr) =>
                write_numeric_array_with_header(f, "UInt32", arr.as_ref()),
            NumericArray::UInt64(arr) =>
                write_numeric_array_with_header(f, "UInt64", arr.as_ref()),
            NumericArray::Float32(arr) =>
                write_numeric_array_with_header(f, "Float32", arr.as_ref()),
            NumericArray::Float64(arr) =>
                write_numeric_array_with_header(f, "Float64", arr.as_ref()),
            NumericArray::Null =>
                writeln!(f, "NullNumericArray [0 values]"),
        }
    }
}

/// Writes the standard header, then delegates to the contained array's Display.
fn write_numeric_array_with_header<T>(
    f: &mut Formatter<'_>,
    dtype_name: &str,
    arr: &(impl MaskedArray<CopyType = T> + Display + ?Sized),
) -> std::fmt::Result {
    writeln!(
        f,
        "NumericArray [{dtype_name}] [{} values] (null count: {})",
        arr.len(),
        arr.null_count()
    )?;
    // Delegate row formatting
    Display::fmt(arr, f)
}