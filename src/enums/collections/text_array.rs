use std::fmt::{Display, Formatter};
use std::sync::Arc;

use crate::enums::error::MinarrowError;
use crate::{Bitmask, CategoricalArray, MaskedArray, StringArray};

/// Unifying Text array container
///
/// This exists to unify string and categorical
///  operations, simplify API's and streamline user ergonomics.
///
/// ### Usage:
/// - It is accessible from `Array` using `.str()`,
/// and provides typed variant access via for e.g.,
/// `.str64()`, so one can drill down to the required
/// granularity via `myarr.str().str64()`
/// - This streamlines function implementations,
/// and, despite the additional `enum` layer,
/// matching lanes in many real-world scenarios.
/// This is because one can for e.g., unify a
/// function signature with `impl Into<TextArray>`,
/// and all of the subtypes, plus `Array` and `TextArray`,
/// all qualify.
/// - Additionally, you can then use one `Text` implementation
/// on the enum dispatch arm for all `Text` variants, or,
/// in many cases, for the entire text arm when they are the same.
/// This is mostly useful for the `NumericArray` enum where unifying
/// floats and integers is a very common pattern, however there may be
/// are cases where it's useful for `Categorical` and `String` data too.
/// For e.g., you can do a match on `TextArray` then just handle those
/// two cases explicitly, or for e.g., accept `impl Into<TextArray>`,
/// then simply case it to `String` for some string ops when feeling
/// barbarian vibes.
///
/// ### Typecasting behaviour
/// - If the enum already holds the given type *(which should be known at compile-time)*,
/// then using accessors like `.str32()` is zero-cost, as it transfers ownership.
/// - If you want to keep the original, of course use `.clone()` beforehand.
/// - If you use an accessor to a different base type, e.g., `.cat32()` when it's a
/// `.str32()` already in the enum, it will convert it. Therefore, be mindful
/// of performance when this occurs.
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub enum TextArray {
    String32(Arc<StringArray<u32>>),
    #[cfg(feature = "large_string")]
    String64(Arc<StringArray<u64>>),
    #[cfg(feature = "extended_categorical")]
    Categorical8(Arc<CategoricalArray<u8>>),
    #[cfg(feature = "extended_categorical")]
    Categorical16(Arc<CategoricalArray<u16>>),
    Categorical32(Arc<CategoricalArray<u32>>),
    #[cfg(feature = "extended_categorical")]
    Categorical64(Arc<CategoricalArray<u64>>),
    #[default]
    Null // Default Marker for mem::take
}

impl TextArray {
    /// Returns the logical length of the text array.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            TextArray::String32(arr) => arr.len(),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => arr.len(),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical8(arr) => arr.len(),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => arr.len(),
            TextArray::Categorical32(arr) => arr.len(),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => arr.len(),
            TextArray::Null => 0
        }
    }

    /// Returns the underlying null mask, if any.
    #[inline]
    pub fn null_mask(&self) -> Option<&Bitmask> {
        match self {
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
            TextArray::Null => None
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
            (TextArray::String32(a), TextArray::String32(b)) => Arc::make_mut(a).append_array(b),
            #[cfg(feature = "large_string")]
            (TextArray::String64(a), TextArray::String64(b)) => Arc::make_mut(a).append_array(b),
            #[cfg(feature = "extended_categorical")]
            (TextArray::Categorical8(a), TextArray::Categorical8(b)) => {
                Arc::make_mut(a).append_array(b)
            }
            #[cfg(feature = "extended_categorical")]
            (TextArray::Categorical16(a), TextArray::Categorical16(b)) => {
                Arc::make_mut(a).append_array(b)
            }
            (TextArray::Categorical32(a), TextArray::Categorical32(b)) => {
                Arc::make_mut(a).append_array(b)
            }
            #[cfg(feature = "extended_categorical")]
            (TextArray::Categorical64(a), TextArray::Categorical64(b)) => {
                Arc::make_mut(a).append_array(b)
            }
            (TextArray::Null, TextArray::Null) => (),
            (lhs, rhs) => panic!("Cannot append {:?} into {:?}", rhs, lhs)
        }
    }

    /// Casts to StringArray<u32>
    /// 
    /// - Converts via TryFrom,
    /// - Uses *CloneOnWrite (COW)* when it's already a `String32`.
    pub fn str32(self) -> Result<StringArray<u32>, MinarrowError> {
        match self {
            TextArray::String32(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone())
            },
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => Ok(StringArray::<u32>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical8(arr) => Ok(StringArray::<u32>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => Ok(StringArray::<u32>::try_from(&*arr)?),
            TextArray::Categorical32(arr) => Ok(StringArray::<u32>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => Ok(StringArray::<u32>::try_from(&*arr)?),
            TextArray::Null => Err(MinarrowError::NullError { message: None })
        }
    }

    /// Casts to StringArray<u64>
    /// 
    /// - Converts via `From` or `TryFrom`, depending on the inner type
    /// - Uses *CloneOnWrite (COW)* when it's already a `String64`.
    #[cfg(feature = "large_string")]
    pub fn str64(self) -> Result<StringArray<u64>, MinarrowError> {
        match self {
            TextArray::String64(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone())
            },
            TextArray::String32(arr) => Ok(StringArray::<u64>::from(&*arr)),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical8(arr) => Ok(StringArray::<u64>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => Ok(StringArray::<u64>::try_from(&*arr)?),
            TextArray::Categorical32(arr) => Ok(StringArray::<u64>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => Ok(StringArray::<u64>::try_from(&*arr)?),
            TextArray::Null => Err(MinarrowError::NullError { message: None })
        }
    }

    /// Casts to CategoricalArray<u32>
    /// 
    /// - Converts via `From` or `TryFrom`, depending on the inner type
    /// - Uses *CloneOnWrite (COW)* when it's already a `Categorical32`.
    pub fn cat32(self) -> Result<CategoricalArray<u32>, MinarrowError> {
        match self {
            TextArray::Categorical32(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone())
            },
            TextArray::String32(arr) => Ok(CategoricalArray::<u32>::try_from(&*arr)?),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => Ok(CategoricalArray::<u32>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical8(arr) => Ok(CategoricalArray::<u32>::from(&*arr)),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => Ok(CategoricalArray::<u32>::from(&*arr)),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => Ok(CategoricalArray::<u32>::try_from(&*arr)?),
            TextArray::Null => Err(MinarrowError::NullError { message: None })
        }
    }

    /// Casts to CategoricalArray<u64>
    /// 
    /// - Converts via `From` or `TryFrom`, depending on the inner type
    /// - Uses *CloneOnWrite (COW)* when it's already a `Categorical32`.
    #[cfg(feature = "extended_categorical")]
    pub fn cat64(self) -> Result<CategoricalArray<u64>, MinarrowError> {
        match self {
            TextArray::Categorical64(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone())
            },
            TextArray::String32(arr) => Ok(CategoricalArray::<u64>::try_from(&*arr)?),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => Ok(CategoricalArray::<u64>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical8(arr) => Ok(CategoricalArray::<u64>::from(&*arr)),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => Ok(CategoricalArray::<u64>::from(&*arr)),
            TextArray::Categorical32(arr) => Ok(CategoricalArray::<u64>::from(&*arr)),
            TextArray::Null => Err(MinarrowError::NullError { message: None })
        }
    }

    /// Casts to CategoricalArray<u8>.
    /// 
    /// - Converts via `From` or `TryFrom`, depending on the inner type
    /// - Uses *CloneOnWrite (COW)* when it's already a `Categorical8`.
    #[cfg(feature = "extended_categorical")]
    pub fn cat8(self) -> Result<CategoricalArray<u8>, MinarrowError> {
        match self {
            TextArray::Categorical8(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone())
            },
            TextArray::String32(arr) => Ok(CategoricalArray::<u8>::try_from(&*arr)?),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => Ok(CategoricalArray::<u8>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) => Ok(CategoricalArray::<u8>::try_from(&*arr)?),
            TextArray::Categorical32(arr) => Ok(CategoricalArray::<u8>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => Ok(CategoricalArray::<u8>::try_from(&*arr)?),
            TextArray::Null => Err(MinarrowError::NullError { message: None })
        }
    }

    /// Casts to CategoricalArray<u16>.
    /// 
    /// - Converts via `From` or `TryFrom`, depending on the inner type
    /// - Uses *CloneOnWrite (COW)* when it's already a `Categorical16`.
    #[cfg(feature = "extended_categorical")]
    pub fn cat16(self) -> Result<CategoricalArray<u16>, MinarrowError> {
        match self {
            TextArray::Categorical16(arr) => match Arc::try_unwrap(arr) {
                Ok(inner) => Ok(inner),
                Err(shared) => Ok((*shared).clone())
            },
            TextArray::String32(arr) => Ok(CategoricalArray::<u16>::try_from(&*arr)?),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) => Ok(CategoricalArray::<u16>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical8(arr) => Ok(CategoricalArray::<u16>::from(&*arr)),
            TextArray::Categorical32(arr) => Ok(CategoricalArray::<u16>::try_from(&*arr)?),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) => Ok(CategoricalArray::<u16>::try_from(&*arr)?),
            TextArray::Null => Err(MinarrowError::NullError { message: None })
        }
    }
}

impl Display for TextArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TextArray::String32(arr) =>
                write_text_array_with_header(f, "String32", arr.as_ref()),
            #[cfg(feature = "large_string")]
            TextArray::String64(arr) =>
                write_text_array_with_header(f, "String64", arr.as_ref()),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical8(arr) =>
                write_text_array_with_header(f, "Categorical8", arr.as_ref()),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(arr) =>
                write_text_array_with_header(f, "Categorical16", arr.as_ref()),
            TextArray::Categorical32(arr) =>
                write_text_array_with_header(f, "Categorical32", arr.as_ref()),
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(arr) =>
                write_text_array_with_header(f, "Categorical64", arr.as_ref()),
            TextArray::Null =>
                writeln!(f, "TextArray::Null [0 values]"),
        }
    }
}

/// Writes the header, then delegates row printing to the array's Display.
fn write_text_array_with_header<T>(
    f: &mut Formatter<'_>,
    dtype: &str,
    arr: &(impl MaskedArray<CopyType = T> + Display + ?Sized),
) -> std::fmt::Result {
    writeln!(
        f,
        "TextArray [{dtype}] [{} values] (null count: {})",
        arr.len(),
        arr.null_count()
    )?;
    Display::fmt(arr, f)
}