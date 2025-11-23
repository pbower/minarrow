//! # **TemporalArray Module** - *High-Level DateTimes Array Type for Unified Signature Dispatch*
//!
//! TemporalArray unifies all datetime-based arrays into a single enum for
//! standardised temporal operations.
//!   
//! ## Features:
//! - direct variant access
//! - zero-cost casts when the type is known
//! - lossless conversions between 32-bit and 64-bit datetime types.  
//! - simplifies function signatures by accepting `impl Into<TemporalArray>`
//! - centralises dispatch
//! - preserves SIMD-aligned buffers across all temporal variants.

use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use crate::{Bitmask, DatetimeArray, MaskedArray};
use crate::{
    enums::{error::MinarrowError, shape_dim::ShapeDim},
    traits::{concatenate::Concatenate, shape::Shape},
};

/// Temporal Array
///
/// Unified datetime array container
///
/// ## Purpose
/// Exists to unify datetime operations,
/// simplify API's and streamline user ergonomics.
///
/// ## Usage:
/// - It is accessible from `Array` using `.dt()`,
/// and provides typed variant access via for e.g.,
/// `.dt32()`, so one can drill down to the required
/// granularity via `myarr.dt().dt32()`
/// - This streamlines function implementations *(at least for the `NumericArray`
/// case where this pattern is the most useful)*,
/// and, despite the additional `enum` layer,
/// matching lanes in many real-world scenarios.
/// This is because one can for e.g., unify a
/// function signature with `impl Into<TemporalArray>`,
/// and all of the subtypes, plus `Array` and `TemporalArray`,
/// all qualify.
/// - Additionally, you can then use one `Temporal` implementation
/// on the enum dispatch arm for all `Temporal` variants, or,
/// in many cases, for the entire datetime arm when they are the same.
///
/// ### Handling Times, Durations, etc.
/// We use one Physical type to hold all datetime variants,
/// i.e., the *Apache Arrow* types `DATE32`, `TIME32`, `DURATION` etc.,
/// and the Logical type is stored on the `Field` as metadata, given they
/// otherwise have the same underlying data representation. To treat
/// them differently in API usage, you can use the `TimeUnit` and `IntervalUnit`,
/// along with the `ArrowType` that is stored on the `Field` in `Minarrow`,
/// and match on these for any desired behaviour. The `Field` is packaged together
/// with `Array` *(which then drill-down accesses `TemporalArray` on the fly, or
/// in dispatch routing scenarios)*.
///
/// ### Typecasting behaviour
/// - If the enum already holds the given type *(which should be known at compile-time)*,
/// then using accessors like `.dt32()` is zero-cost, as it transfers ownership.
/// - If you want to keep the original, of course use `.clone()` beforehand.
/// - If you use an accessor to a different base type, e.g., `.dt64()` when it's a
/// `.dt32()` already in the enum, it will convert it. Therefore, be mindful
/// of performance when this occurs.
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub enum TemporalArray {
    // The datetimes are chunked by their common memory layout rather than logical type
    // These can be casted to the relevant Arrow type at the FFI layer as needed
    Datetime32(Arc<DatetimeArray<i32>>), // DATE32, TIME32, DURATION(s), DURATION(ms) (32-bit)
    // DATE64, TIMESTAMP (ms/us/ns), DURATION (ms/us/ns), TIME64, DURATION(us), DURATION(ns)
    Datetime64(Arc<DatetimeArray<i64>>),
    #[default]
    Null, // Default Marker for mem::take
}

impl TemporalArray {
    /// Returns the logical length of the temporal array.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            TemporalArray::Datetime32(arr) => arr.len(),
            TemporalArray::Datetime64(arr) => arr.len(),
            TemporalArray::Null => 0,
        }
    }

    /// Returns the underlying null mask, if any.
    #[inline]
    pub fn null_mask(&self) -> Option<&Bitmask> {
        match self {
            TemporalArray::Datetime32(arr) => arr.null_mask.as_ref(),
            TemporalArray::Datetime64(arr) => arr.null_mask.as_ref(),
            TemporalArray::Null => None,
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
            (TemporalArray::Datetime32(a), TemporalArray::Datetime32(b)) => {
                Arc::make_mut(a).append_array(b)
            }
            (TemporalArray::Datetime64(a), TemporalArray::Datetime64(b)) => {
                Arc::make_mut(a).append_array(b)
            }
            (TemporalArray::Null, TemporalArray::Null) => (),
            (lhs, rhs) => panic!("Cannot append {:?} into {:?}", rhs, lhs),
        }
    }

    /// Inserts all values (and null mask if present) from `other` into `self` at the specified index.
    ///
    /// This is an **O(n)** operation.
    ///
    /// Returns an error if the two arrays are of different variants or incompatible types,
    /// or if the index is out of bounds.
    ///
    /// This function uses copy-on-write semantics for arrays wrapped in `Arc`.
    pub fn insert_rows(&mut self, index: usize, other: &Self) -> Result<(), MinarrowError> {
        match (self, other) {
            (TemporalArray::Datetime32(a), TemporalArray::Datetime32(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            (TemporalArray::Datetime64(a), TemporalArray::Datetime64(b)) => {
                Arc::make_mut(a).insert_rows(index, b)
            }
            (TemporalArray::Null, TemporalArray::Null) => Ok(()),
            (lhs, rhs) => Err(MinarrowError::TypeError {
                from: "TemporalArray",
                to: "TemporalArray",
                message: Some(format!(
                    "Cannot insert {} into {}: incompatible types",
                    temporal_variant_name(rhs),
                    temporal_variant_name(lhs)
                )),
            }),
        }
    }

    /// Splits the TemporalArray at the specified index, consuming self and returning two arrays.
    pub fn split(self, index: usize) -> Result<(Self, Self), MinarrowError> {
        use std::sync::Arc;

        match self {
            TemporalArray::Datetime32(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    TemporalArray::Datetime32(Arc::new(left)),
                    TemporalArray::Datetime32(Arc::new(right)),
                ))
            }
            TemporalArray::Datetime64(a) => {
                let (left, right) = Arc::try_unwrap(a)
                    .unwrap_or_else(|arc| (*arc).clone())
                    .split(index)?;
                Ok((
                    TemporalArray::Datetime64(Arc::new(left)),
                    TemporalArray::Datetime64(Arc::new(right)),
                ))
            }
            TemporalArray::Null => Err(MinarrowError::IndexError(
                "Cannot split Null array".to_string(),
            )),
        }
    }

    /// Returns an Arc<DatetimeArray<i32>> (casting if needed).
    pub fn dt32(self) -> Result<DatetimeArray<i32>, MinarrowError> {
        match self {
            TemporalArray::Datetime32(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            TemporalArray::Datetime64(arr) => Ok(DatetimeArray::<i32>::try_from(&*arr)?),
            TemporalArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }

    /// Returns an Arc<DatetimeArray<i64>> (casting if needed).
    pub fn dt64(self) -> Result<DatetimeArray<i64>, MinarrowError> {
        match self {
            TemporalArray::Datetime64(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone()),
            },
            TemporalArray::Datetime32(arr) => Ok(DatetimeArray::<i64>::from(&*arr)),
            TemporalArray::Null => Err(MinarrowError::NullError { message: None }),
        }
    }
}

impl Shape for TemporalArray {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Concatenate for TemporalArray {
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        match (self, other) {
            (TemporalArray::Datetime32(a), TemporalArray::Datetime32(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(TemporalArray::Datetime32(Arc::new(a.concat(b)?)))
            }
            (TemporalArray::Datetime64(a), TemporalArray::Datetime64(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(TemporalArray::Datetime64(Arc::new(a.concat(b)?)))
            }
            (TemporalArray::Null, TemporalArray::Null) => Ok(TemporalArray::Null),
            (lhs, rhs) => Err(MinarrowError::IncompatibleTypeError {
                from: "TemporalArray",
                to: "TemporalArray",
                message: Some(format!(
                    "Cannot concatenate mismatched TemporalArray variants: {:?} and {:?}",
                    temporal_variant_name(&lhs),
                    temporal_variant_name(&rhs)
                )),
            }),
        }
    }
}

/// Helper function to get the variant name for error messages
fn temporal_variant_name(arr: &TemporalArray) -> &'static str {
    match arr {
        TemporalArray::Datetime32(_) => "Datetime32",
        TemporalArray::Datetime64(_) => "Datetime64",
        TemporalArray::Null => "Null",
    }
}

impl Display for TemporalArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TemporalArray::Datetime32(arr) => {
                write_temporal_array_with_header(f, "Datetime32", arr.as_ref())
            }
            TemporalArray::Datetime64(arr) => {
                write_temporal_array_with_header(f, "Datetime64", arr.as_ref())
            }
            TemporalArray::Null => writeln!(f, "TemporalArray::Null [0 values]"),
        }
    }
}

/// Writes the standard header, then delegates to the contained array's Display.
fn write_temporal_array_with_header<T>(
    f: &mut Formatter<'_>,
    dtype: &str,
    arr: &(impl MaskedArray<CopyType = T> + Display + ?Sized),
) -> std::fmt::Result {
    writeln!(
        f,
        "TemporalArray [{dtype}] [{} values] (null count: {})",
        arr.len(),
        arr.null_count()
    )?;
    Display::fmt(arr, f)
}
