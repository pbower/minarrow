//! # FieldArray Module - *De-Facto *Column* Array type w' Tagged Arrow Metadata*
//!
//! Couples a `Field` (array-level schema metadata) with an immutable `Array` of values.
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

#[cfg(feature = "views")]
use crate::aliases::FieldAVT;
use crate::ffi::arrow_dtype::ArrowType;
use crate::{Array, Field};


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
    pub null_count: usize
}

impl FieldArray {
    /// Constructs a new `FieldArray` from an existing `Field` and `Array`.
    pub fn new(field: Field, array: Array) -> Self {
        let null_count = array.null_count();
        FieldArray { field: field.into(), array, null_count }
    }

    /// Constructs a new `FieldArray` from an existing `Arc<Field>` and `Array`.
    pub fn new_arc(field: Arc<Field>, array: Array) -> Self {
        let null_count = array.null_count();
        FieldArray { field: field, array, null_count }
    }

    /// Constructs a new `FieldArray` from a name and any supported typed array,
    /// automatically wrapping as `Array` and inferring type and nullability.
    pub fn from_inner<N, A>(name: N, arr: A) -> Self
    where
        N: Into<String>,
        A: Into<Array>
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
        array: Array
    ) -> Self {
        let null_count = array.null_count();
        let field = Field {
            name: field_name.into(),
            dtype,
            nullable: nullable.unwrap_or_else(|| array.is_nullable()),
            metadata: metadata.unwrap_or_default()
        };
        FieldArray {
            field: field.into(),
            array: array.into(),
            null_count
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

    /// Returns a zero-copy view (`FieldArraySlice`) into the window `[offset, offset+len)`.
    ///
    /// The returned object holds references into the original `FieldArray`.
    ///
    /// The `(&Array, Offset, WindowLength), &Field)` `FieldArraySlice` pattern here
    /// is a once-off we avoid recommending.
    #[cfg(feature = "views")]
    #[inline]
    pub fn to_window(&self, offset: usize, len: usize) -> FieldAVT {
        ((&self.array, offset, len), &self.field)
    }

    /// Returns a new owned FieldArray with array sliced `[offset, offset+len)`.
    pub fn slice_clone(&self, offset: usize, len: usize) -> Self {
        let array: Array = self.array.slice_clone(offset, len).into();
        let null_count = array.null_count();
        FieldArray {
            field: self.field.clone(),
            array: array.into(),
            null_count
        }
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

impl Display for FieldArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "\nFieldArray \"{}\" [{} values] (dtype: {:?})",
            self.field.name,
            self.array.len(),
            self.field.dtype
        )?;
        self.array.fmt(f)
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
        let view = fa.to_window(1, 2);
        assert_eq!(view.1.name, "x");
        assert_eq!(view.0.2, 2);
        assert_eq!(view.0.1, 1);
        assert_eq!(view.0.2, 2);
        assert_eq!(view.0.0.len(), 3);
    }
}
