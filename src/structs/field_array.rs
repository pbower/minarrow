//! # **FieldArray Module** - *De-Facto *Column* Array type w' Tagged Arrow Metadata*
//!
//! Couples a `Field` (i.e., array-level schema metadata) with an immutable `Array` of values.
//!
//! Used as the primary column representation in `Minarrow` tables, ensuring
//! schema and data remain consistent.  
//!
//! Supports creation from raw components or by inferring schema from arrays,
//! and is the unit transferred over Arrow FFI or to external libraries
//! such as Apache Arrow or Polars.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

#[cfg(feature = "cast_arrow")]
use arrow::array::ArrayRef;
#[cfg(feature = "cast_polars")]
use polars::series::Series;

#[cfg(all(feature = "select", feature = "views"))]
use crate::ArrayV;
#[cfg(feature = "views")]
use crate::aliases::FieldAVT;
use crate::enums::error::MinarrowError;
use crate::enums::shape_dim::ShapeDim;
use crate::ffi::arrow_dtype::ArrowType;
use crate::ffi::arrow_dtype::CategoricalIndexType;
use crate::traits::concatenate::Concatenate;
#[cfg(all(feature = "select", feature = "views"))]
use crate::traits::selection::{DataSelector, RowSelection};
use crate::traits::shape::Shape;
use crate::{Array, Field, NumericArray, TextArray};
#[cfg(feature = "datetime")]
use crate::{TemporalArray, TimeUnit};

/// # FieldArray
///
/// Named and typed data column with associated array values.
///
/// ## Role
/// - Combines a `Field` with an immutable `Array` instance.
/// - `FieldArray` integrates naturally into a `Table`, where immutability enforces row-length guarantees.
/// It can also serve as a self-documenting array and is required when sending `Minarrow` data
/// over FFI to `Apache Arrow`. In such cases, it's worth ensuring the correct logical `Datetime` Arrow type
/// is built when constructing the `Field`, as this determines the `Arrow` type on the receiving side.
///
/// ##  
/// ```rust
/// use minarrow::{Array, Field, FieldArray, MaskedArray};
/// use minarrow::structs::field_array::{field_array};
/// use minarrow::ffi::arrow_dtype::ArrowType;
/// use minarrow::structs::variants::integer::IntegerArray;
///
/// // Build a typed array
/// let mut ints = IntegerArray::<i32>::default();
/// ints.push(1);
/// ints.push(2);
/// let arr = Array::from_int32(ints);
///
/// // Fast constructor - infers type/nullability
/// let fa = field_array("id", arr.clone());
///
/// assert_eq!(fa.field.name, "id");
/// assert_eq!(fa.arrow_type(), ArrowType::Int32);
/// assert_eq!(fa.len(), 2);
///
/// // Take an owned slice [offset..offset+len)
/// let sub = fa.slice_clone(0, 1);
/// assert_eq!(sub.len(), 1);
///
/// // Standard constructor
///
/// // Describe it with a Field and wrap as FieldArray
/// let field = Field::new("id", ArrowType::Int32, false, None);
/// let fa = FieldArray::new(field, arr);
///
/// assert_eq!(fa.field.name, "id");
/// assert_eq!(fa.arrow_type(), ArrowType::Int32);
/// assert_eq!(fa.len(), 2);
///
/// // Take an owned slice [offset..offset+len)
/// let sub = fa.slice_clone(0, 1);
/// assert_eq!(sub.len(), 1);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FieldArray {
    /// Array metadata
    pub field: Arc<Field>,

    /// The array's inner payload is wrapped in Arc for immutability
    /// so it can safely share across threads.
    /// When part of a Table *(or higher-dimensional structure)*,
    /// immutability also upholds shape constraints.
    pub array: Array,

    /// Null count for the immutable array to support skipping null-mask
    /// operations when it's `0`, and/or related strategies.
    pub null_count: usize,
}

impl FieldArray {
    /// Constructs a new `FieldArray` from an existing `Field` and `Array`.
    pub fn new(field: Field, array: Array) -> Self {
        let null_count = array.null_count();
        FieldArray {
            field: field.into(),
            array,
            null_count,
        }
    }

    /// Constructs a new `FieldArray` from an existing `Arc<Field>` and `Array`.
    pub fn new_arc(field: Arc<Field>, array: Array) -> Self {
        let null_count = array.null_count();
        FieldArray {
            field: field,
            array,
            null_count,
        }
    }

    /// Constructs a new `FieldArray` from a name and any supported typed array,
    /// automatically wrapping as `Array` and inferring type and nullability.
    pub fn from_arr<N, A>(name: N, arr: A) -> Self
    where
        N: Into<String>,
        A: Into<Array>,
    {
        let array: Array = arr.into();
        let dtype = array.arrow_type();
        let nullable = array.is_nullable();
        let field = Field::new(name, dtype, nullable, None);
        FieldArray::new(field, array)
    }

    /// Constructs a new `FieldArray` from raw field components and an `Array`.
    pub fn from_parts<T: Into<String>>(
        field_name: T,
        dtype: ArrowType,
        nullable: Option<bool>,
        metadata: Option<BTreeMap<String, String>>,
        array: Array,
    ) -> Self {
        let null_count = array.null_count();
        let field = Field {
            name: field_name.into(),
            dtype,
            nullable: nullable.unwrap_or_else(|| array.is_nullable()),
            metadata: metadata.unwrap_or_default(),
        };
        FieldArray {
            field: field.into(),
            array: array.into(),
            null_count,
        }
    }

    pub fn len(&self) -> usize {
        self.array.len()
    }

    pub fn is_empty(&self) -> bool {
        self.array.len() == 0
    }

    pub fn arrow_type(&self) -> ArrowType {
        self.field.dtype.clone()
    }

    /// Returns a new FieldArray with updated timezone metadata.
    ///
    /// The underlying timestamp data (always UTC) remains unchanged. Only the timezone
    /// metadata in the Field's ArrowType is updated for interpretation/display purposes.
    ///
    /// # Arguments
    /// * `tz` - Timezone string in Arrow format (IANA like "America/New_York" or offset like "+05:00")
    ///
    /// # Errors
    /// Returns an error if the array is not a Timestamp type.
    #[cfg(feature = "datetime")]
    pub fn tz(&self, tz: &str) -> Result<Self, MinarrowError> {
        match &self.field.dtype {
            ArrowType::Timestamp(unit, _) => {
                let mut new_field = (*self.field).clone();
                new_field.dtype = ArrowType::Timestamp(*unit, Some(tz.to_string()));
                Ok(FieldArray {
                    field: Arc::new(new_field),
                    array: self.array.clone(),
                    null_count: self.null_count,
                })
            }
            _ => Err(MinarrowError::TypeError {
                from: "FieldArray",
                to: "Timestamp",
                message: Some("tz() requires a Timestamp type".to_string()),
            }),
        }
    }

    /// Returns a new FieldArray with timezone metadata set to "UTC".
    ///
    /// The underlying timestamp data (always UTC) remains unchanged. Only the timezone
    /// metadata in the Field's ArrowType is updated.
    ///
    /// # Errors
    /// Returns an error if the array is not a Timestamp type.
    #[cfg(feature = "datetime")]
    pub fn utc(&self) -> Result<Self, MinarrowError> {
        self.tz("UTC")
    }

    /// Returns a zero-copy view (`FieldArraySlice`) into the window `[offset, offset+len)`.
    ///
    /// The returned object holds references into the original `FieldArray`.
    ///
    /// The `(&Array, Offset, WindowLength), &Field)` `FieldArraySlice` pattern here
    /// is a once-off we avoid recommending.
    #[cfg(feature = "views")]
    #[inline]
    pub fn view(&self, offset: usize, len: usize) -> FieldAVT<'_> {
        ((&self.array, offset, len), &self.field)
    }

    /// Returns a new owned FieldArray with array sliced `[offset, offset+len)`.
    pub fn slice_clone(&self, offset: usize, len: usize) -> Self {
        let array: Array = self.array.slice_clone(offset, len).into();
        let null_count = array.null_count();
        FieldArray {
            field: self.field.clone(),
            array: array.into(),
            null_count,
        }
    }

    /// Updates the cached null_count from the underlying array.
    /// Should be called after any mutation of the array that could change null count.
    #[inline]
    pub fn refresh_null_count(&mut self) {
        self.null_count = self.array.null_count();
    }

    /// Returns the cached null count.
    /// This is kept in sync with the underlying array via refresh_null_count().
    #[inline]
    pub fn null_count(&self) -> usize {
        self.null_count
    }

    /// Concatenates another FieldArray's data into this one using copy-on-write semantics.
    /// If this FieldArray's array has Arc reference count > 1, the data is cloned first.
    /// Both FieldArrays must have compatible types. Updates the cached null_count.
    pub fn concat_field_array(&mut self, other: &FieldArray) {
        self.array.concat_array(&other.array);
        self.refresh_null_count();
    }

    /// Provides mutable access to the underlying array with automatic null_count refresh.
    /// Uses copy-on-write semantics - clones array data if Arc reference count > 1.
    /// Use this for operations that may change the null count.
    pub fn with_array_mut<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Array) -> R,
    {
        let result = f(&mut self.array);
        self.refresh_null_count();
        result
    }

    /// Export this field+array over FFI and import into arrow-rs.
    #[cfg(feature = "cast_arrow")]
    #[inline]
    pub fn to_apache_arrow(&self) -> ArrayRef {
        self.array.to_apache_arrow_with_field(&self.field)
    }

    // ** The below polars function is tested tests/polars.rs **

    /// Casts the FieldArray to a polars Series
    #[cfg(feature = "cast_polars")]
    pub fn to_polars(&self) -> Series {
        let name = self.field.name.as_str();
        self.array.to_polars_with_field(name, &self.field)
    }
}

/// Creates a new basic field array based on a name and an existing array
pub fn field_array<T: Into<String>>(name: T, array: Array) -> FieldArray {
    let dtype = array.arrow_type();
    let nullable = array.is_nullable();
    let field = Field::new(name, dtype, nullable, None);
    FieldArray::new(field, array)
}

/// Helper to create a proper Field for an Array with correct type, mask, and metadata
pub fn create_field_for_array(
    name: &str,
    array: &Array,
    other_array: Option<&Array>,
    metadata: Option<std::collections::BTreeMap<String, String>>,
) -> Field {
    let arrow_type = match array {
        Array::NumericArray(num_arr) => match num_arr {
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
        Array::TextArray(text_arr) => match text_arr {
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
        Array::TemporalArray(temp_arr) => match temp_arr {
            TemporalArray::Datetime32(dt_arr) => match &dt_arr.time_unit {
                TimeUnit::Days => ArrowType::Date32,
                unit => ArrowType::Time32(unit.clone()),
            },
            TemporalArray::Datetime64(dt_arr) => match &dt_arr.time_unit {
                TimeUnit::Milliseconds => ArrowType::Date64,
                TimeUnit::Microseconds | TimeUnit::Nanoseconds => {
                    ArrowType::Time64(dt_arr.time_unit.clone())
                }
                unit => ArrowType::Timestamp(unit.clone(), None), // TODO: extract timezone from metadata
            },
            TemporalArray::Null => ArrowType::Null,
        },
        Array::BooleanArray(_) => ArrowType::Boolean,
        Array::Null => ArrowType::Null,
    };

    let has_mask = array.null_mask().is_some()
        || other_array.map_or(false, |other| other.null_mask().is_some());

    Field::new(name, arrow_type, has_mask, metadata)
}

impl Display for FieldArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "\nFieldArray \"{}\" [{} values] (dtype: {:?})",
            self.field.name,
            self.array.len(),
            self.field.dtype
        )?;

        // For Timestamp types with timezone, use custom printing
        #[cfg(feature = "datetime")]
        if let ArrowType::Timestamp(_unit, Some(ref tz)) = self.field.dtype {
            return format_field_array_with_timezone(f, self, tz);
        }

        self.array.fmt(f)
    }
}

#[cfg(feature = "datetime")]
fn format_field_array_with_timezone(
    f: &mut Formatter<'_>,
    field_array: &FieldArray,
    timezone: &str,
) -> std::fmt::Result {
    use crate::traits::print::MAX_PREVIEW;
    use crate::{Array, TemporalArray};

    let arr = &field_array.array;
    let len = arr.len();
    let nulls = arr.null_count();

    if let Array::TemporalArray(TemporalArray::Datetime64(dt)) = arr {
        writeln!(
            f,
            "DatetimeArray [{} values] (dtype: datetime[{:?}], timezone: {}, nulls: {})",
            len, dt.time_unit, timezone, nulls
        )?;

        write!(f, "[")?;
        for i in 0..usize::min(len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", format_datetime_with_tz(dt.as_ref(), i, timezone))?;
        }
        if len > MAX_PREVIEW {
            write!(f, ", ...")?;
        }
        writeln!(f, "]")
    } else if let Array::TemporalArray(TemporalArray::Datetime32(dt)) = arr {
        writeln!(
            f,
            "DatetimeArray [{} values] (dtype: datetime[{:?}], timezone: {}, nulls: {})",
            len, dt.time_unit, timezone, nulls
        )?;

        write!(f, "[")?;
        for i in 0..usize::min(len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", format_datetime_with_tz(dt.as_ref(), i, timezone))?;
        }
        if len > MAX_PREVIEW {
            write!(f, ", ...")?;
        }
        writeln!(f, "]")
    } else {
        field_array.array.fmt(f)
    }
}

#[cfg(feature = "datetime")]
fn format_datetime_with_tz<T>(arr: &crate::DatetimeArray<T>, idx: usize, timezone: &str) -> String
where
    T: crate::Integer + std::fmt::Display,
{
    use crate::traits::print::format_datetime_value;
    format_datetime_value(arr, idx, Some(timezone))
}

impl Shape for FieldArray {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Concatenate for FieldArray {
    /// Concatenates two FieldArrays, consuming both.
    ///
    /// # Requirements
    /// - Both FieldArrays must have matching field metadata:
    ///   - Same name
    ///   - Same dtype
    ///   - Same nullability
    ///
    /// # Returns
    /// A new FieldArray with the concatenated array data
    ///
    /// # Errors
    /// - `IncompatibleTypeError` if field metadata doesn't match
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // Validate field compatibility
        if self.field.name != other.field.name {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "FieldArray",
                to: "FieldArray",
                message: Some(format!(
                    "Field name mismatch: '{}' vs '{}'",
                    self.field.name, other.field.name
                )),
            });
        }

        if self.field.dtype != other.field.dtype {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "FieldArray",
                to: "FieldArray",
                message: Some(format!(
                    "Field '{}' dtype mismatch: {:?} vs {:?}",
                    self.field.name, self.field.dtype, other.field.dtype
                )),
            });
        }

        if self.field.nullable != other.field.nullable {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "FieldArray",
                to: "FieldArray",
                message: Some(format!(
                    "Field '{}' nullable mismatch: {} vs {}",
                    self.field.name, self.field.nullable, other.field.nullable
                )),
            });
        }

        // Concatenate the underlying arrays
        let concatenated_array = self.array.concat(other.array)?;
        let null_count = concatenated_array.null_count();

        // Create result FieldArray with the same field metadata
        Ok(FieldArray {
            field: self.field,
            array: concatenated_array,
            null_count,
        })
    }
}

// ===== Selection Trait Implementation =====

#[cfg(all(feature = "select", feature = "views"))]
impl RowSelection for FieldArray {
    type View = ArrayV;

    fn r<S: DataSelector>(&self, selection: S) -> ArrayV {
        if selection.is_contiguous() {
            // Contiguous selection (ranges): adjust offset and len
            let indices = selection.resolve_indices(self.array.len());
            if indices.is_empty() {
                return ArrayV::new(self.array.clone(), 0, 0);
            }
            ArrayV::new(self.array.clone(), indices[0], indices.len())
        } else {
            // Non-contiguous selection (index arrays): gather into new array
            let view = ArrayV::from(self.array.clone());
            let indices = selection.resolve_indices(self.array.len());
            let gathered_array = view.gather_indices(&indices);
            ArrayV::new(gathered_array, 0, indices.len())
        }
    }

    fn get_row_count(&self) -> usize {
        self.array.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::variants::integer::IntegerArray;
    use crate::traits::masked_array::MaskedArray;

    #[test]
    fn test_field_array_basic_construction() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        let array = Array::from_int32(arr);

        let field = Field::new("my_col", ArrowType::Int32, false, None);
        let field_array = FieldArray::new(field.clone(), array.clone());

        assert_eq!(field_array.len(), 2);
        assert_eq!(field_array.field, field.into());
        assert_eq!(field_array.array, array.into());
    }

    #[test]
    fn test_field_array_from_parts_infers_nullability() {
        let mut arr = IntegerArray::<i64>::default();
        arr.push(10);
        arr.push_null(); // makes it nullable
        let array = Array::from_int64(arr);

        let field_array =
            FieldArray::from_parts("nullable_col", ArrowType::Int64, None, None, array.clone());

        assert_eq!(field_array.field.name, "nullable_col");
        assert_eq!(field_array.field.dtype, ArrowType::Int64);
        assert_eq!(field_array.field.nullable, true);
        assert_eq!(field_array.len(), 2);
        assert_eq!(field_array.array, array.into());
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_field_array_slice() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(10);
        arr.push(20);
        arr.push(30);

        let fa = field_array("x", Array::from_int32(arr));
        let view = fa.view(1, 2);
        assert_eq!(view.1.name, "x");
        assert_eq!(view.0.2, 2);
        assert_eq!(view.0.1, 1);
        assert_eq!(view.0.2, 2);
        assert_eq!(view.0.0.len(), 3);
    }

    #[test]
    fn test_null_count_cache_sync_concat() {
        // Create first FieldArray with nulls
        let mut arr1 = IntegerArray::<i32>::default();
        arr1.push(1);
        arr1.push_null();
        arr1.push(3);
        let mut fa1 = field_array("test", Array::from_int32(arr1));
        assert_eq!(fa1.null_count(), 1);

        // Create second FieldArray with nulls
        let mut arr2 = IntegerArray::<i32>::default();
        arr2.push_null();
        arr2.push(5);
        let fa2 = field_array("test", Array::from_int32(arr2));
        assert_eq!(fa2.null_count(), 1);

        // Concatenate and verify null_count cache is updated
        fa1.concat_field_array(&fa2);
        assert_eq!(fa1.len(), 5);
        assert_eq!(fa1.null_count(), 2); // Should be 2 nulls total
    }

    #[test]
    fn test_null_count_cache_sync_with_array_mut() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        let mut fa = field_array("test", Array::from_int32(arr));
        assert_eq!(fa.null_count(), 0);

        // Mutate through with_array_mut to add nulls
        fa.with_array_mut(|array| {
            array.concat_array(&Array::from_int32({
                let mut new_arr = IntegerArray::<i32>::default();
                new_arr.push_null();
                new_arr.push_null();
                new_arr
            }));
        });

        assert_eq!(fa.len(), 4);
        assert_eq!(fa.null_count(), 2); // Cache should be refreshed automatically
    }

    #[test]
    fn test_refresh_null_count() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        let mut fa = field_array("test", Array::from_int32(arr));
        assert_eq!(fa.null_count(), 0);

        // Manually mutate underlying array (simulating external mutation)
        if let Array::NumericArray(crate::NumericArray::Int32(int_arr)) = &mut fa.array {
            use crate::traits::masked_array::MaskedArray;
            std::sync::Arc::make_mut(int_arr).push_null();
        }

        // Cache is now stale
        assert_eq!(fa.null_count, 0); // Cached value still 0
        assert_eq!(fa.array.null_count(), 1); // Actual value is 1

        // Refresh the cache
        fa.refresh_null_count();
        assert_eq!(fa.null_count(), 1); // Cache now updated
    }
}

#[cfg(test)]
mod concat_tests {
    use super::*;
    use crate::structs::variants::integer::IntegerArray;
    use crate::traits::concatenate::Concatenate;
    use crate::traits::masked_array::MaskedArray;

    #[test]
    fn test_field_array_concat_basic() {
        let arr1 = IntegerArray::<i32>::from_slice(&[1, 2, 3]);
        let fa1 = field_array("numbers", Array::from_int32(arr1));

        let arr2 = IntegerArray::<i32>::from_slice(&[4, 5, 6]);
        let fa2 = field_array("numbers", Array::from_int32(arr2));

        let result = fa1.concat(fa2).unwrap();

        assert_eq!(result.len(), 6);
        assert_eq!(result.field.name, "numbers");
        assert_eq!(result.field.dtype, ArrowType::Int32);

        if let Array::NumericArray(crate::NumericArray::Int32(arr)) = result.array {
            assert_eq!(arr.len(), 6);
            assert_eq!(arr.get(0), Some(1));
            assert_eq!(arr.get(5), Some(6));
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_field_array_concat_with_nulls() {
        let mut arr1 = IntegerArray::<i32>::with_capacity(3, true);
        arr1.push(10);
        arr1.push_null();
        arr1.push(30);
        let fa1 = FieldArray::from_parts(
            "data",
            ArrowType::Int32,
            Some(true),
            None,
            Array::from_int32(arr1),
        );

        let mut arr2 = IntegerArray::<i32>::with_capacity(2, true);
        arr2.push_null();
        arr2.push(50);
        let fa2 = FieldArray::from_parts(
            "data",
            ArrowType::Int32,
            Some(true),
            None,
            Array::from_int32(arr2),
        );

        let result = fa1.concat(fa2).unwrap();

        assert_eq!(result.len(), 5);
        assert_eq!(result.null_count(), 2);

        if let Array::NumericArray(crate::NumericArray::Int32(arr)) = result.array {
            assert_eq!(arr.get(0), Some(10));
            assert_eq!(arr.get(1), None);
            assert_eq!(arr.get(2), Some(30));
            assert_eq!(arr.get(3), None);
            assert_eq!(arr.get(4), Some(50));
        } else {
            panic!("Expected Int32 array");
        }
    }

    #[test]
    fn test_field_array_concat_name_mismatch() {
        let arr1 = IntegerArray::<i32>::from_slice(&[1, 2]);
        let fa1 = field_array("col_a", Array::from_int32(arr1));

        let arr2 = IntegerArray::<i32>::from_slice(&[3, 4]);
        let fa2 = field_array("col_b", Array::from_int32(arr2));

        let result = fa1.concat(fa2);
        assert!(result.is_err());

        if let Err(MinarrowError::IncompatibleTypeError { message, .. }) = result {
            assert!(message.unwrap().contains("Field name mismatch"));
        } else {
            panic!("Expected IncompatibleTypeError");
        }
    }

    #[test]
    fn test_field_array_concat_dtype_mismatch() {
        let arr1 = IntegerArray::<i32>::from_slice(&[1, 2]);
        let fa1 = field_array("data", Array::from_int32(arr1));

        let arr2 = crate::FloatArray::<f64>::from_slice(&[3.0, 4.0]);
        let fa2 = field_array("data", Array::from_float64(arr2));

        let result = fa1.concat(fa2);
        assert!(result.is_err());

        if let Err(MinarrowError::IncompatibleTypeError { message, .. }) = result {
            assert!(message.unwrap().contains("dtype mismatch"));
        } else {
            panic!("Expected IncompatibleTypeError");
        }
    }

    #[test]
    fn test_field_array_concat_nullable_mismatch() {
        let arr1 = IntegerArray::<i32>::from_slice(&[1, 2]);
        let fa1 = FieldArray::from_parts(
            "data",
            ArrowType::Int32,
            Some(false),
            None,
            Array::from_int32(arr1),
        );

        let mut arr2 = IntegerArray::<i32>::with_capacity(2, true);
        arr2.push(3);
        arr2.push(4);
        let fa2 = FieldArray::from_parts(
            "data",
            ArrowType::Int32,
            Some(true),
            None,
            Array::from_int32(arr2),
        );

        let result = fa1.concat(fa2);
        assert!(result.is_err());

        if let Err(MinarrowError::IncompatibleTypeError { message, .. }) = result {
            assert!(message.unwrap().contains("nullable mismatch"));
        } else {
            panic!("Expected IncompatibleTypeError");
        }
    }
}
