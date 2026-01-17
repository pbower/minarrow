//! # **Value Module** - *Single *Whole Type Universe* Value Container*
//!
//! Contains the `Value` enum, a unified container for any Minarrow-supported data structure.
//!
//! ## Description
//! -Encapsulates scalars, arrays, tables, views, chunked collections, bitmasks, fields,
//! matrices, cubes, nested values, and custom user-defined types.
//!
//! ## Purpose
//! Used to create a global type universe for function signatures and dispatch, enabling
//! constructs like `Result<Value, MinarrowError>` without restricting the contained type.
//!
//! ## Supports:
//! - recursive containers (boxed, arced, tuples, vectors)
//! - `From`/`TryFrom` conversions for safe extraction
//! - equality comparison across all variants, including custom values
//! - custom extension types if needed

#[cfg(feature = "cube")]
use crate::Cube;
#[cfg(feature = "datetime")]
use crate::DatetimeArray;
#[cfg(feature = "matrix")]
use crate::Matrix;
#[cfg(feature = "scalar_type")]
use crate::Scalar;
use crate::{
    Array, BooleanArray, FieldArray, FloatArray, IntegerArray, StringArray, Table,
    enums::error::MinarrowError, enums::shape_dim::ShapeDim, traits::concatenate::Concatenate,
    traits::custom_value::CustomValue, traits::shape::Shape,
};
use std::convert::TryFrom;
use std::{convert::From, sync::Arc};

#[cfg(feature = "chunked")]
use crate::{SuperArray, SuperTable};

#[cfg(feature = "views")]
use crate::{ArrayV, TableV};

#[cfg(all(feature = "chunked", feature = "views"))]
use crate::{SuperArrayV, SuperTableV};

/// # Value
///
/// Unified value enum representing any supported data structure.
///
/// ## Details
/// - Wraps scalar values, arrays, array windows, full tables, or table windows
/// under a single type for function signatures and downstream dispatch.
/// - This can be useful when you need a global *type universe*.
/// - It is not part of the `Arrow` specification, but is useful
/// because of the flexibility it adds unifying all types to a single one.
/// For example, to return `Result<Value, Error>`, particularly in engine contexts.
/// - It's enabled optionally via the `value_type` feature.
///
/// ## Usage
/// You can also use it to hold a custom type under the `Custom` entry.
/// As long as the object implements `Debug`, `Clone`, and `PartialEq`,
/// remains `Send + Sync`, and implements `Any` it can be stored in `Value::Custom`.
/// `Any` is implemented automatically for all Rust types with a `'static` lifetime.
#[derive(Debug, Clone)]
pub enum Value {
    #[cfg(feature = "scalar_type")]
    Scalar(Scalar),
    Array(Arc<Array>),
    #[cfg(feature = "views")]
    ArrayView(Arc<ArrayV>),
    FieldArray(Arc<FieldArray>),
    Table(Arc<Table>),
    #[cfg(feature = "views")]
    TableView(Arc<TableV>),
    #[cfg(feature = "chunked")]
    SuperArray(Arc<SuperArray>),
    #[cfg(all(feature = "chunked", feature = "views"))]
    SuperArrayView(Arc<SuperArrayV>),
    #[cfg(feature = "chunked")]
    SuperTable(Arc<SuperTable>),
    #[cfg(all(feature = "chunked", feature = "views"))]
    SuperTableView(Arc<SuperTableV>),
    #[cfg(feature = "matrix")]
    Matrix(Arc<Matrix>),
    #[cfg(feature = "cube")]
    Cube(Arc<Cube>),
    VecValue(Arc<Vec<Value>>),
    // For recursive
    BoxValue(Box<Value>),
    ArcValue(Arc<Value>),
    Tuple2(Arc<(Value, Value)>),
    Tuple3(Arc<(Value, Value, Value)>),
    Tuple4(Arc<(Value, Value, Value, Value)>),
    Tuple5(Arc<(Value, Value, Value, Value, Value)>),
    Tuple6(Arc<(Value, Value, Value, Value, Value, Value)>),

    /// Arbitrary user or library-defined payload.
    ///
    /// As long as the object implements `Debug`, `Clone`, and `PartialEq`,
    /// remains `Send + Sync`, and implements `Any` it can be stored in `Value::Custom`.
    /// `Any` is implemented automatically for all Rust types with a `'static` lifetime.
    ///
    /// Borrowed values **cannot** be used directly.
    /// These must be wrapped in `Arc` or otherwise promoted to `'static` to
    /// store inside `Value`.
    ///
    /// It's recommended that creators also implement `From` and `TryFrom`.
    Custom(Arc<dyn CustomValue>),
}

impl Value {
    /// Computes the logical row/element count for the batch’s input `Value`.
    ///
    /// This normalises the various `Value` representations so callers can consistently pass a
    /// `[start, len)` range to `execute_fn`.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            #[cfg(feature = "scalar_type")]
            Value::Scalar(_) => 1,

            Value::Table(t) => t.n_rows,

            #[cfg(feature = "views")]
            Value::TableView(tv) => tv.len,

            Value::Array(a) => a.len(),

            #[cfg(feature = "views")]
            Value::ArrayView(av) => av.array.len(),

            Value::FieldArray(fa) => fa.array.len(),

            #[cfg(feature = "chunked")]
            Value::SuperArray(sa) => sa.len(),

            #[cfg(all(feature = "chunked", feature = "views"))]
            Value::SuperArrayView(sav) => sav.len(),

            #[cfg(feature = "chunked")]
            Value::SuperTable(st) => st.len(),

            #[cfg(all(feature = "chunked", feature = "views"))]
            Value::SuperTableView(stv) => stv.len,

            #[cfg(feature = "matrix")]
            Value::Matrix(m) => m.len(),

            #[cfg(feature = "cube")]
            Value::Cube(c) => c.len(),

            // A vector of `Value`s is treated as a logical concatenation.
            Value::VecValue(vv) => vv.iter().map(|x| x.len()).sum(),

            // Recursive wrappers: delegate to the inner `Value`.
            Value::BoxValue(bv) => bv.len(),
            Value::ArcValue(av) => av.len(),

            // Tuples are treated as a logical concatenation of their elements.
            Value::Tuple2(t2) => t2.0.len() + t2.1.len(),
            Value::Tuple3(t3) => t3.0.len() + t3.1.len() + t3.2.len(),
            Value::Tuple4(t4) => t4.0.len() + t4.1.len() + t4.2.len() + t4.3.len(),
            Value::Tuple5(t5) => t5.0.len() + t5.1.len() + t5.2.len() + t5.3.len() + t5.4.len(),
            Value::Tuple6(t6) => {
                t6.0.len() + t6.1.len() + t6.2.len() + t6.3.len() + t6.4.len() + t6.5.len()
            }

            // Defer to the custom payload's notion of length (per `CustomValue` contract).
            Value::Custom(_cv) => panic!("Length is not implemented for custom value type."),
        }
    }

    /// Returns true if the value is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Implements `PartialEq` for `Value`
///
/// This includes special handling for the `Custom` type.
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        use Value::*;
        match (self, other) {
            #[cfg(feature = "scalar_type")]
            (Scalar(a), Scalar(b)) => a == b,
            (Array(a), Array(b)) => **a == **b,
            #[cfg(feature = "views")]
            (ArrayView(a), ArrayView(b)) => **a == **b,
            (Table(a), Table(b)) => **a == **b,
            #[cfg(feature = "views")]
            (TableView(a), TableView(b)) => **a == **b,
            #[cfg(feature = "chunked")]
            (SuperArray(a), SuperArray(b)) => a == b,
            #[cfg(all(feature = "chunked", feature = "views"))]
            (SuperArrayView(a), SuperArrayView(b)) => a == b,
            #[cfg(feature = "chunked")]
            (SuperTable(a), SuperTable(b)) => **a == **b,
            (FieldArray(a), FieldArray(b)) => **a == **b,
            #[cfg(feature = "matrix")]
            (Matrix(a), Matrix(b)) => a == b,
            #[cfg(feature = "cube")]
            (Cube(a), Cube(b)) => **a == **b,
            (Custom(a), Custom(b)) => a.eq_box(&**b),
            (VecValue(a), VecValue(b)) => **a == **b,
            (BoxValue(a), BoxValue(b)) => a == b,
            (ArcValue(a), ArcValue(b)) => a == b,
            (Tuple2(a), Tuple2(b)) => **a == **b,
            (Tuple3(a), Tuple3(b)) => **a == **b,
            (Tuple4(a), Tuple4(b)) => **a == **b,
            (Tuple5(a), Tuple5(b)) => **a == **b,
            (Tuple6(a), Tuple6(b)) => **a == **b,
            _ => false,
        }
    }
}

/// Implements `Eq` for `Value`
///
/// Since PartialEq is reflexive, symmetric, and transitive for Value,
/// we can safely implement Eq.
impl Eq for Value {}

/// Macro to implement `From` for `Value` variants.
macro_rules! impl_value_from {
    ($variant:ident: $t:ty) => {
        impl From<$t> for Value {
            #[inline]
            fn from(v: $t) -> Self {
                Value::$variant(v)
            }
        }
    };
}

/// Macro to implement `TryFrom<Value>` for `Value` variants.
macro_rules! impl_tryfrom_value {
    ($variant:ident: $t:ty) => {
        impl TryFrom<Value> for $t {
            type Error = MinarrowError;
            fn try_from(v: Value) -> Result<Self, Self::Error> {
                match v {
                    Value::$variant(inner) => Ok(inner),
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: stringify!($t),
                        message: Some("Value type mismatch".to_owned()),
                    }),
                }
            }
        }
    };
}

// Scalars
#[cfg(feature = "scalar_type")]
impl_value_from!(Scalar: Scalar);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value!(Scalar: Scalar);

// TryFrom<Value> for primitive numeric types
//
// Enables DataStream<T>::collect() -> Result<T, BlockError> for numeric types.
// Extracts Value::Scalar variant and converts to the target numeric type.

#[cfg(feature = "scalar_type")]
macro_rules! impl_tryfrom_value_numeric {
    ($t:ty, $method:ident) => {
        impl TryFrom<Value> for $t {
            type Error = MinarrowError;
            #[inline]
            fn try_from(v: Value) -> Result<Self, Self::Error> {
                match v {
                    Value::Scalar(s) => match s {
                        Scalar::Null => Err(MinarrowError::TypeError {
                            from: "Value::Scalar(Null)",
                            to: stringify!($t),
                            message: Some("Cannot convert Null to numeric type".to_owned()),
                        }),
                        _ => Ok(s.$method()),
                    },
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: stringify!($t),
                        message: Some("Expected Value::Scalar variant".to_owned()),
                    }),
                }
            }
        }
    };
}

#[cfg(feature = "scalar_type")]
impl_tryfrom_value_numeric!(f64, f64);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_numeric!(f32, f32);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_numeric!(i64, i64);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_numeric!(i32, i32);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_numeric!(u64, u64);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_numeric!(u32, u32);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_numeric!(bool, bool);

// Option<T> for nullable results - Null becomes None
#[cfg(feature = "scalar_type")]
macro_rules! impl_tryfrom_value_option {
    ($t:ty, $method:ident) => {
        impl TryFrom<Value> for Option<$t> {
            type Error = MinarrowError;
            #[inline]
            fn try_from(v: Value) -> Result<Self, Self::Error> {
                match v {
                    Value::Scalar(s) => match s {
                        Scalar::Null => Ok(None),
                        _ => Ok(Some(s.$method())),
                    },
                    _ => Err(MinarrowError::TypeError {
                        from: "Value",
                        to: concat!("Option<", stringify!($t), ">"),
                        message: Some("Expected Value::Scalar variant".to_owned()),
                    }),
                }
            }
        }
    };
}

#[cfg(feature = "scalar_type")]
impl_tryfrom_value_option!(f64, f64);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_option!(f32, f32);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_option!(i64, i64);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_option!(i32, i32);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_option!(u64, u64);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_option!(u32, u32);
#[cfg(feature = "scalar_type")]
impl_tryfrom_value_option!(bool, bool);

// Array-like types - Arc-wrapped (large types)
impl From<Array> for Value {
    #[inline]
    fn from(v: Array) -> Self {
        Value::Array(Arc::new(v))
    }
}

#[cfg(feature = "views")]
impl From<ArrayV> for Value {
    #[inline]
    fn from(v: ArrayV) -> Self {
        Value::ArrayView(Arc::new(v))
    }
}

impl From<Table> for Value {
    #[inline]
    fn from(v: Table) -> Self {
        Value::Table(Arc::new(v))
    }
}

#[cfg(feature = "views")]
impl From<TableV> for Value {
    #[inline]
    fn from(v: TableV) -> Self {
        Value::TableView(Arc::new(v))
    }
}

impl From<FieldArray> for Value {
    #[inline]
    fn from(v: FieldArray) -> Self {
        Value::FieldArray(Arc::new(v))
    }
}

#[cfg(feature = "chunked")]
impl From<SuperTable> for Value {
    #[inline]
    fn from(v: SuperTable) -> Self {
        Value::SuperTable(Arc::new(v))
    }
}

#[cfg(feature = "cube")]
impl From<Cube> for Value {
    #[inline]
    fn from(v: Cube) -> Self {
        Value::Cube(Arc::new(v))
    }
}

#[cfg(feature = "chunked")]
impl From<SuperArray> for Value {
    #[inline]
    fn from(v: SuperArray) -> Self {
        Value::SuperArray(Arc::new(v))
    }
}

#[cfg(all(feature = "chunked", feature = "views"))]
impl From<SuperArrayV> for Value {
    #[inline]
    fn from(v: SuperArrayV) -> Self {
        Value::SuperArrayView(Arc::new(v))
    }
}

#[cfg(feature = "matrix")]
impl From<Matrix> for Value {
    #[inline]
    fn from(v: Matrix) -> Self {
        Value::Matrix(Arc::new(v))
    }
}

// TryFrom for Array-like types - Arc-wrapped (unwrap or clone)
impl TryFrom<Value> for Array {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Array(inner) => Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "Array",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(feature = "views")]
impl TryFrom<Value> for ArrayV {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::ArrayView(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "ArrayV",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for Table {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Table(inner) => Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "Table",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(feature = "views")]
impl TryFrom<Value> for TableV {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::TableView(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "TableV",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for FieldArray {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::FieldArray(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "FieldArray",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(feature = "chunked")]
impl TryFrom<Value> for SuperTable {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::SuperTable(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "SuperTable",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(feature = "cube")]
impl TryFrom<Value> for Cube {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Cube(inner) => Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "Cube",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(feature = "chunked")]
impl TryFrom<Value> for SuperArray {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::SuperArray(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "SuperArray",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(all(feature = "chunked", feature = "views"))]
impl TryFrom<Value> for SuperArrayV {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::SuperArrayView(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "SuperArrayV",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

#[cfg(feature = "matrix")]
impl TryFrom<Value> for Matrix {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Matrix(inner) => Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "Matrix",
                message: Some("Value type mismatch".to_owned()),
            }),
        }
    }
}

// Recursive containers
impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Self {
        Value::VecValue(Arc::new(v))
    }
}

impl From<(Value, Value)> for Value {
    fn from(v: (Value, Value)) -> Self {
        Value::Tuple2(Arc::new(v))
    }
}

impl From<(Value, Value, Value)> for Value {
    fn from(v: (Value, Value, Value)) -> Self {
        Value::Tuple3(Arc::new(v))
    }
}

impl From<(Value, Value, Value, Value)> for Value {
    fn from(v: (Value, Value, Value, Value)) -> Self {
        Value::Tuple4(Arc::new(v))
    }
}

impl From<(Value, Value, Value, Value, Value)> for Value {
    fn from(v: (Value, Value, Value, Value, Value)) -> Self {
        Value::Tuple5(Arc::new(v))
    }
}

impl From<(Value, Value, Value, Value, Value, Value)> for Value {
    fn from(v: (Value, Value, Value, Value, Value, Value)) -> Self {
        Value::Tuple6(Arc::new(v))
    }
}

// TryFrom for recursive containers
impl TryFrom<Value> for Vec<Value> {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::VecValue(inner) => {
                Ok(Arc::try_unwrap(inner).unwrap_or_else(|arc| (*arc).clone()))
            }
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "Vec<Value>",
                message: Some("Expected VecValue variant".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for (Value, Value) {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Tuple2(tuple) => Ok(Arc::try_unwrap(tuple).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "(Value, Value)",
                message: Some("Expected Tuple2 variant".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for (Value, Value, Value) {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Tuple3(tuple) => Ok(Arc::try_unwrap(tuple).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "(Value, Value, Value)",
                message: Some("Expected Tuple3 variant".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for (Value, Value, Value, Value) {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Tuple4(tuple) => Ok(Arc::try_unwrap(tuple).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "(Value, Value, Value, Value)",
                message: Some("Expected Tuple4 variant".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for (Value, Value, Value, Value, Value) {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Tuple5(tuple) => Ok(Arc::try_unwrap(tuple).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "(Value, Value, Value, Value, Value)",
                message: Some("Expected Tuple5 variant".to_owned()),
            }),
        }
    }
}

impl TryFrom<Value> for (Value, Value, Value, Value, Value, Value) {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::Tuple6(tuple) => Ok(Arc::try_unwrap(tuple).unwrap_or_else(|arc| (*arc).clone())),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "(Value, Value, Value, Value, Value, Value)",
                message: Some("Expected Tuple6 variant".to_owned()),
            }),
        }
    }
}

impl Shape for Value {
    fn shape(&self) -> ShapeDim {
        match self {
            #[cfg(feature = "scalar_type")]
            Value::Scalar(_) => ShapeDim::Rank0(1),
            Value::Array(array) => array.shape(),
            #[cfg(feature = "views")]
            Value::ArrayView(array_view) => array_view.shape(),
            Value::Table(table) => table.shape(),
            #[cfg(feature = "views")]
            Value::TableView(table_view) => table_view.shape(),
            #[cfg(feature = "chunked")]
            Value::SuperArray(chunked_array) => ShapeDim::Rank1(chunked_array.len()),
            #[cfg(all(feature = "chunked", feature = "views"))]
            Value::SuperArrayView(chunked_view) => ShapeDim::Rank1(chunked_view.len()),
            #[cfg(feature = "chunked")]
            Value::SuperTable(chunked_table) => ShapeDim::Rank2 {
                rows: chunked_table.n_rows(),
                cols: chunked_table.n_cols(),
            },
            #[cfg(all(feature = "chunked", feature = "views"))]
            Value::SuperTableView(chunked_view) => ShapeDim::Rank2 {
                rows: chunked_view.n_rows(),
                cols: chunked_view.n_cols(),
            },
            Value::FieldArray(field_array) => field_array.shape(),
            #[cfg(feature = "matrix")]
            Value::Matrix(matrix) => matrix.shape(),
            #[cfg(feature = "cube")]
            Value::Cube(cube) => cube.shape(),
            Value::VecValue(vec_value) => {
                let shapes: Vec<ShapeDim> = vec_value.iter().map(|v| v.shape()).collect();
                ShapeDim::Collection(shapes)
            }
            Value::BoxValue(boxed_value) => boxed_value.shape(),
            Value::ArcValue(arc_value) => arc_value.shape(),
            Value::Tuple2(tuple) => ShapeDim::Collection(vec![tuple.0.shape(), tuple.1.shape()]),
            Value::Tuple3(tuple) => {
                ShapeDim::Collection(vec![tuple.0.shape(), tuple.1.shape(), tuple.2.shape()])
            }
            Value::Tuple4(tuple) => ShapeDim::Collection(vec![
                tuple.0.shape(),
                tuple.1.shape(),
                tuple.2.shape(),
                tuple.3.shape(),
            ]),
            Value::Tuple5(tuple) => ShapeDim::Collection(vec![
                tuple.0.shape(),
                tuple.1.shape(),
                tuple.2.shape(),
                tuple.3.shape(),
                tuple.4.shape(),
            ]),
            Value::Tuple6(tuple) => ShapeDim::Collection(vec![
                tuple.0.shape(),
                tuple.1.shape(),
                tuple.2.shape(),
                tuple.3.shape(),
                tuple.4.shape(),
                tuple.5.shape(),
            ]),
            Value::Custom(_) => ShapeDim::Unknown,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Value Creation Macros
// ═══════════════════════════════════════════════════════════════════════════
//
// Macros for creating `Value` instances from arrays and scalars.
// These wrap the existing `arr_*` macros and `Scalar` constructors.

// ─────────────────────────────────────────────────────────────────────────────
// Signed Integer Array Values
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_i8 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_i8![$($x)*])
    };
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_i16 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_i16![$($x)*])
    };
}

#[macro_export]
macro_rules! val_i32 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_i32![$($x)*])
    };
}

#[macro_export]
macro_rules! val_i64 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_i64![$($x)*])
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Unsigned Integer Array Values
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_u8 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_u8![$($x)*])
    };
}

#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_u16 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_u16![$($x)*])
    };
}

#[macro_export]
macro_rules! val_u32 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_u32![$($x)*])
    };
}

#[macro_export]
macro_rules! val_u64 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_u64![$($x)*])
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Floating Point Array Values
// ─────────────────────────────────────────────────────────────────────────────

#[macro_export]
macro_rules! val_f32 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_f32![$($x)*])
    };
}

#[macro_export]
macro_rules! val_f64 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_f64![$($x)*])
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Boolean Array Values
// ─────────────────────────────────────────────────────────────────────────────

#[macro_export]
macro_rules! val_bool {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_bool![$($x)*])
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// String Array Values
// ─────────────────────────────────────────────────────────────────────────────

#[macro_export]
macro_rules! val_str32 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_str32![$($x)*])
    };
}

#[cfg(feature = "large_string")]
#[macro_export]
macro_rules! val_str64 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_str64![$($x)*])
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Categorical Array Values
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! val_cat8 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_cat8![$($x)*])
    };
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! val_cat16 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_cat16![$($x)*])
    };
}

#[macro_export]
macro_rules! val_cat32 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_cat32![$($x)*])
    };
}

#[cfg(feature = "extended_categorical")]
#[macro_export]
macro_rules! val_cat64 {
    ($($x:tt)*) => {
        $crate::Value::from($crate::arr_cat64![$($x)*])
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Scalar Values
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "scalar_type")]
#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_scalar_i8 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::Int8($v))
    };
}

#[cfg(feature = "scalar_type")]
#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_scalar_i16 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::Int16($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_i32 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::Int32($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_i64 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::Int64($v))
    };
}

#[cfg(feature = "scalar_type")]
#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_scalar_u8 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::UInt8($v))
    };
}

#[cfg(feature = "scalar_type")]
#[cfg(feature = "extended_numeric_types")]
#[macro_export]
macro_rules! val_scalar_u16 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::UInt16($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_u32 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::UInt32($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_u64 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::UInt64($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_f32 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::Float32($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_f64 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::Float64($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_bool {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::Boolean($v))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_str32 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::String32($v.to_string()))
    };
}

#[cfg(feature = "scalar_type")]
#[cfg(feature = "large_string")]
#[macro_export]
macro_rules! val_scalar_str64 {
    ($v:expr) => {
        $crate::Value::from($crate::Scalar::String64($v.to_string()))
    };
}

#[cfg(feature = "scalar_type")]
#[macro_export]
macro_rules! val_scalar_null {
    () => {
        $crate::Value::from($crate::Scalar::Null)
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// Concatenate Implementation
// ═══════════════════════════════════════════════════════════════════════════

impl Concatenate for Value {
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        use Value::*;
        match (self, other) {
            // Scalar + Scalar -> Array (length 2)
            #[cfg(feature = "scalar_type")]
            (Scalar(a), Scalar(b)) => {
                use crate::Scalar::*;
                match (a, b) {
                    // Integer types
                    #[cfg(feature = "extended_numeric_types")]
                    (Int8(a), Int8(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_int8(arr))))
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    (Int16(a), Int16(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_int16(arr))))
                    }
                    (Int32(a), Int32(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_int32(arr))))
                    }
                    (Int64(a), Int64(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_int64(arr))))
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    (UInt8(a), UInt8(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_uint8(arr))))
                    }
                    #[cfg(feature = "extended_numeric_types")]
                    (UInt16(a), UInt16(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_uint16(arr))))
                    }
                    (UInt32(a), UInt32(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_uint32(arr))))
                    }
                    (UInt64(a), UInt64(b)) => {
                        let arr = IntegerArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_uint64(arr))))
                    }
                    // Float types
                    (Float32(a), Float32(b)) => {
                        let arr = FloatArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_float32(arr))))
                    }
                    (Float64(a), Float64(b)) => {
                        let arr = FloatArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_float64(arr))))
                    }
                    // Boolean
                    (Boolean(a), Boolean(b)) => {
                        let arr = BooleanArray::from_slice(&[a, b]);
                        Ok(Value::Array(Arc::new(crate::Array::from_bool(arr))))
                    }
                    // String types
                    (String32(a), String32(b)) => {
                        let arr = StringArray::from_slice(&[a.as_str(), b.as_str()]);
                        Ok(Value::Array(Arc::new(crate::Array::from_string32(arr))))
                    }
                    #[cfg(feature = "large_string")]
                    (String64(a), String64(b)) => {
                        let arr = StringArray::from_slice(&[a.as_str(), b.as_str()]);
                        Ok(Value::Array(Arc::new(crate::Array::from_string64(arr))))
                    }
                    // Datetime types
                    #[cfg(feature = "datetime")]
                    (Datetime32(a), Datetime32(b)) => {
                        let arr = DatetimeArray::from_slice(&[a, b], None);
                        Ok(Value::Array(Arc::new(crate::Array::from_datetime_i32(arr))))
                    }
                    #[cfg(feature = "datetime")]
                    (Datetime64(a), Datetime64(b)) => {
                        let arr = DatetimeArray::from_slice(&[a, b], None);
                        Ok(Value::Array(Arc::new(crate::Array::from_datetime_i64(arr))))
                    }
                    // Null + Null
                    (Null, Null) => Ok(Value::Array(Arc::new(crate::Array::Null))),
                    // Mismatched scalar types
                    (lhs, rhs) => Err(MinarrowError::IncompatibleTypeError {
                        from: "Scalar",
                        to: "Array",
                        message: Some(format!(
                            "Cannot concatenate mismatched Scalar types: {:?} and {:?}",
                            scalar_variant_name(&lhs),
                            scalar_variant_name(&rhs)
                        )),
                    }),
                }
            }

            // Array + Array -> Array
            (Array(a), Array(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::Array(Arc::new(a.concat(b)?)))
            }

            // Table + Table -> Table
            (Table(a), Table(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::Table(Arc::new(a.concat(b)?)))
            }

            // Matrix + Matrix -> Matrix
            #[cfg(feature = "matrix")]
            (Matrix(a), Matrix(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::Matrix(Arc::new(a.concat(b)?)))
            }

            // Cube + Cube -> Cube
            #[cfg(feature = "cube")]
            (Cube(a), Cube(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::Cube(Arc::new(a.concat(b)?)))
            }

            // Chunked types
            #[cfg(feature = "chunked")]
            (SuperArray(a), SuperArray(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::SuperArray(Arc::new(a.concat(b)?)))
            }

            #[cfg(feature = "chunked")]
            (SuperTable(a), SuperTable(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::SuperTable(Arc::new(a.concat(b)?)))
            }

            // Tuples (element-wise concatenation, recursive)
            (Tuple2(a_arc), Tuple2(b_arc)) => {
                let (a1, a2) = Arc::try_unwrap(a_arc).unwrap_or_else(|arc| (*arc).clone());
                let (b1, b2) = Arc::try_unwrap(b_arc).unwrap_or_else(|arc| (*arc).clone());
                let c1 = a1.concat(b1)?;
                let c2 = a2.concat(b2)?;
                Ok(Value::Tuple2(Arc::new((c1, c2))))
            }

            (Tuple3(a_arc), Tuple3(b_arc)) => {
                let (a1, a2, a3) = Arc::try_unwrap(a_arc).unwrap_or_else(|arc| (*arc).clone());
                let (b1, b2, b3) = Arc::try_unwrap(b_arc).unwrap_or_else(|arc| (*arc).clone());
                let c1 = a1.concat(b1)?;
                let c2 = a2.concat(b2)?;
                let c3 = a3.concat(b3)?;
                Ok(Value::Tuple3(Arc::new((c1, c2, c3))))
            }

            (Tuple4(a_arc), Tuple4(b_arc)) => {
                let (a1, a2, a3, a4) = Arc::try_unwrap(a_arc).unwrap_or_else(|arc| (*arc).clone());
                let (b1, b2, b3, b4) = Arc::try_unwrap(b_arc).unwrap_or_else(|arc| (*arc).clone());
                let c1 = a1.concat(b1)?;
                let c2 = a2.concat(b2)?;
                let c3 = a3.concat(b3)?;
                let c4 = a4.concat(b4)?;
                Ok(Value::Tuple4(Arc::new((c1, c2, c3, c4))))
            }

            (Tuple5(a_arc), Tuple5(b_arc)) => {
                let (a1, a2, a3, a4, a5) =
                    Arc::try_unwrap(a_arc).unwrap_or_else(|arc| (*arc).clone());
                let (b1, b2, b3, b4, b5) =
                    Arc::try_unwrap(b_arc).unwrap_or_else(|arc| (*arc).clone());
                let c1 = a1.concat(b1)?;
                let c2 = a2.concat(b2)?;
                let c3 = a3.concat(b3)?;
                let c4 = a4.concat(b4)?;
                let c5 = a5.concat(b5)?;
                Ok(Value::Tuple5(Arc::new((c1, c2, c3, c4, c5))))
            }

            (Tuple6(a_arc), Tuple6(b_arc)) => {
                let (a1, a2, a3, a4, a5, a6) =
                    Arc::try_unwrap(a_arc).unwrap_or_else(|arc| (*arc).clone());
                let (b1, b2, b3, b4, b5, b6) =
                    Arc::try_unwrap(b_arc).unwrap_or_else(|arc| (*arc).clone());
                let c1 = a1.concat(b1)?;
                let c2 = a2.concat(b2)?;
                let c3 = a3.concat(b3)?;
                let c4 = a4.concat(b4)?;
                let c5 = a5.concat(b5)?;
                let c6 = a6.concat(b6)?;
                Ok(Value::Tuple6(Arc::new((c1, c2, c3, c4, c5, c6))))
            }

            // Recursive containers (Box, Arc)
            (BoxValue(a), BoxValue(b)) => {
                let result = (*a).concat(*b)?;
                Ok(Value::BoxValue(Box::new(result)))
            }

            (ArcValue(a), ArcValue(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                let result = a.concat(b)?;
                Ok(Value::ArcValue(Arc::new(result)))
            }

            // Views (materialise to owned, concat, wrap back in view)
            #[cfg(feature = "views")]
            (ArrayView(a), ArrayView(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::ArrayView(Arc::new(a.concat(b)?)))
            }

            #[cfg(feature = "views")]
            (TableView(a), TableView(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::TableView(Arc::new(a.concat(b)?)))
            }

            #[cfg(all(feature = "chunked", feature = "views"))]
            (SuperArrayView(a), SuperArrayView(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::SuperArrayView(Arc::new(a.concat(b)?)))
            }

            #[cfg(all(feature = "chunked", feature = "views"))]
            (SuperTableView(a), SuperTableView(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::SuperTableView(Arc::new(a.concat(b)?)))
            }

            // FieldArray + FieldArray => FieldArray
            (FieldArray(a), FieldArray(b)) => {
                let a = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());
                Ok(Value::FieldArray(Arc::new(a.concat(b)?)))
            }

            // VecValue - element-wise concatenation (recursive)
            (VecValue(a), VecValue(b)) => {
                // Unwrap Arcs
                let a_vec = Arc::try_unwrap(a).unwrap_or_else(|arc| (*arc).clone());
                let b_vec = Arc::try_unwrap(b).unwrap_or_else(|arc| (*arc).clone());

                // Validate same length
                if a_vec.len() != b_vec.len() {
                    return Err(MinarrowError::IncompatibleTypeError {
                        from: "VecValue",
                        to: "VecValue",
                        message: Some(format!(
                            "Cannot concatenate VecValue of different lengths: {} vs {}",
                            a_vec.len(),
                            b_vec.len()
                        )),
                    });
                }

                // Element-wise concatenation
                let mut result = Vec::with_capacity(a_vec.len());
                for (val_a, val_b) in a_vec.into_iter().zip(b_vec.into_iter()) {
                    result.push(val_a.concat(val_b)?);
                }

                Ok(Value::VecValue(Arc::new(result)))
            }

            // Custom values cannot be concatenated (no generic way to do it)
            (Custom(_), Custom(_)) => Err(MinarrowError::IncompatibleTypeError {
                from: "Custom",
                to: "Custom",
                message: Some("Cannot concatenate Custom values".to_string()),
            }),

            // Mismatched types
            (lhs, rhs) => Err(MinarrowError::IncompatibleTypeError {
                from: "Value",
                to: "Value",
                message: Some(format!(
                    "Cannot concatenate mismatched Value types: {} and {}",
                    value_variant_name(&lhs),
                    value_variant_name(&rhs)
                )),
            }),
        }
    }
}

/// Helper function to get scalar variant name for error messages
#[cfg(feature = "scalar_type")]
fn scalar_variant_name(scalar: &crate::Scalar) -> &'static str {
    use crate::Scalar::*;
    match scalar {
        Null => "Null",
        Boolean(_) => "Boolean",
        #[cfg(feature = "extended_numeric_types")]
        Int8(_) => "Int8",
        #[cfg(feature = "extended_numeric_types")]
        Int16(_) => "Int16",
        Int32(_) => "Int32",
        Int64(_) => "Int64",
        #[cfg(feature = "extended_numeric_types")]
        UInt8(_) => "UInt8",
        #[cfg(feature = "extended_numeric_types")]
        UInt16(_) => "UInt16",
        UInt32(_) => "UInt32",
        UInt64(_) => "UInt64",
        Float32(_) => "Float32",
        Float64(_) => "Float64",
        String32(_) => "String32",
        #[cfg(feature = "large_string")]
        String64(_) => "String64",
        #[cfg(feature = "datetime")]
        Datetime32(_) => "Datetime32",
        #[cfg(feature = "datetime")]
        Datetime64(_) => "Datetime64",
        #[cfg(feature = "datetime")]
        Interval => "Interval",
    }
}

/// Helper function to get value variant name for error messages
fn value_variant_name(value: &Value) -> &'static str {
    match value {
        #[cfg(feature = "scalar_type")]
        Value::Scalar(_) => "Scalar",
        Value::Array(_) => "Array",
        #[cfg(feature = "views")]
        Value::ArrayView(_) => "ArrayView",
        Value::Table(_) => "Table",
        #[cfg(feature = "views")]
        Value::TableView(_) => "TableView",
        #[cfg(feature = "chunked")]
        Value::SuperArray(_) => "SuperArray",
        #[cfg(all(feature = "chunked", feature = "views"))]
        Value::SuperArrayView(_) => "SuperArrayView",
        #[cfg(feature = "chunked")]
        Value::SuperTable(_) => "SuperTable",
        #[cfg(all(feature = "chunked", feature = "views"))]
        Value::SuperTableView(_) => "SuperTableView",
        Value::FieldArray(_) => "FieldArray",
        #[cfg(feature = "matrix")]
        Value::Matrix(_) => "Matrix",
        #[cfg(feature = "cube")]
        Value::Cube(_) => "Cube",
        Value::VecValue(_) => "VecValue",
        Value::BoxValue(_) => "BoxValue",
        Value::ArcValue(_) => "ArcValue",
        Value::Tuple2(_) => "Tuple2",
        Value::Tuple3(_) => "Tuple3",
        Value::Tuple4(_) => "Tuple4",
        Value::Tuple5(_) => "Tuple5",
        Value::Tuple6(_) => "Tuple6",
        Value::Custom(_) => "Custom",
    }
}

#[cfg(test)]
mod concat_tests {
    use super::*;
    use crate::MaskedArray;
    use crate::structs::field_array::field_array;
    use crate::structs::variants::integer::IntegerArray;

    #[test]
    fn test_value_size() {
        use std::mem::size_of;
        println!("\n=== Value Enum Size Analysis ===");
        println!("Total Value enum size: {} bytes", size_of::<Value>());
        println!("\nIndividual type sizes:");
        println!("  Array: {} bytes", size_of::<crate::Array>());
        println!("  Table: {} bytes", size_of::<crate::Table>());
        println!("  Bitmask: {} bytes", size_of::<crate::Bitmask>());
        println!("  FieldArray: {} bytes", size_of::<crate::FieldArray>());
        println!("  Field: {} bytes", size_of::<crate::Field>());
        #[cfg(feature = "matrix")]
        println!("  Matrix: {} bytes", size_of::<crate::Matrix>());
        #[cfg(feature = "cube")]
        println!("  Cube: {} bytes", size_of::<crate::Cube>());
        #[cfg(feature = "chunked")]
        println!("  SuperArray: {} bytes", size_of::<crate::SuperArray>());
        #[cfg(feature = "chunked")]
        println!("  SuperTable: {} bytes", size_of::<crate::SuperTable>());
        #[cfg(feature = "views")]
        println!("  ArrayView: {} bytes", size_of::<crate::ArrayV>());
        #[cfg(feature = "views")]
        println!("  TableView: {} bytes", size_of::<crate::TableV>());
        #[cfg(all(feature = "views", feature = "chunked"))]
        println!(
            "  SuperArrayView: {} bytes",
            size_of::<crate::SuperArrayV>()
        );
        #[cfg(all(feature = "views", feature = "chunked"))]
        println!(
            "  SuperTableView: {} bytes",
            size_of::<crate::SuperTableV>()
        );
        println!("  Box<Vec<Value>>: {} bytes", size_of::<Box<Vec<Value>>>());
        println!("  Vec<Value>: {} bytes", size_of::<Vec<Value>>());
    }

    #[test]
    fn test_value_concat_field_array() {
        // Create two FieldArrays with matching metadata
        let arr1 = IntegerArray::<i32>::from_slice(&[1, 2, 3]);
        let fa1 = field_array("data", Array::from_int32(arr1));
        let val1 = Value::FieldArray(Arc::new(fa1));

        let arr2 = IntegerArray::<i32>::from_slice(&[4, 5, 6]);
        let fa2 = field_array("data", Array::from_int32(arr2));
        let val2 = Value::FieldArray(Arc::new(fa2));

        let result = val1.concat(val2).unwrap();

        if let Value::FieldArray(fa_arc) = result {
            let fa = Arc::unwrap_or_clone(fa_arc);
            assert_eq!(fa.len(), 6);
            assert_eq!(fa.field.name, "data");
            if let Array::NumericArray(crate::NumericArray::Int32(arr)) = fa.array {
                assert_eq!(arr.get(0), Some(1));
                assert_eq!(arr.get(5), Some(6));
            } else {
                panic!("Expected Int32 array");
            }
        } else {
            panic!("Expected FieldArray value");
        }
    }

    #[test]
    fn test_value_concat_vec_value() {
        // Create two VecValues with same length and matching types
        let arr1_1 = IntegerArray::<i32>::from_slice(&[1, 2]);
        let arr1_2 = IntegerArray::<i32>::from_slice(&[10, 20]);
        let val1 = Value::VecValue(Arc::new(vec![
            Value::Array(Arc::new(Array::from_int32(arr1_1))),
            Value::Array(Arc::new(Array::from_int32(arr1_2))),
        ]));

        let arr2_1 = IntegerArray::<i32>::from_slice(&[3, 4]);
        let arr2_2 = IntegerArray::<i32>::from_slice(&[30, 40]);
        let val2 = Value::VecValue(Arc::new(vec![
            Value::Array(Arc::new(Array::from_int32(arr2_1))),
            Value::Array(Arc::new(Array::from_int32(arr2_2))),
        ]));

        let result = val1.concat(val2).unwrap();

        if let Value::VecValue(vec) = result {
            assert_eq!(vec.len(), 2);

            // Check first element
            if let Value::Array(arc) = &vec[0] {
                if let Array::NumericArray(crate::NumericArray::Int32(arr)) = arc.as_ref() {
                    assert_eq!(arr.len(), 4);
                    assert_eq!(arr.get(0), Some(1));
                    assert_eq!(arr.get(1), Some(2));
                    assert_eq!(arr.get(2), Some(3));
                    assert_eq!(arr.get(3), Some(4));
                } else {
                    panic!("Expected Int32 array in first element");
                }
            } else {
                panic!("Expected Array value in first element");
            }

            // Check second element
            if let Value::Array(arc) = &vec[1] {
                if let Array::NumericArray(crate::NumericArray::Int32(arr)) = arc.as_ref() {
                    assert_eq!(arr.len(), 4);
                    assert_eq!(arr.get(0), Some(10));
                    assert_eq!(arr.get(1), Some(20));
                    assert_eq!(arr.get(2), Some(30));
                    assert_eq!(arr.get(3), Some(40));
                } else {
                    panic!("Expected Int32 array in second element");
                }
            } else {
                panic!("Expected Array value in second element");
            }
        } else {
            panic!("Expected VecValue");
        }
    }

    #[test]
    fn test_value_concat_vec_value_length_mismatch() {
        let arr1 = IntegerArray::<i32>::from_slice(&[1, 2]);
        let val1 = Value::VecValue(Arc::new(vec![Value::Array(Arc::new(Array::from_int32(
            arr1,
        )))]));

        let arr2_1 = IntegerArray::<i32>::from_slice(&[3, 4]);
        let arr2_2 = IntegerArray::<i32>::from_slice(&[5, 6]);
        let val2 = Value::VecValue(Arc::new(vec![
            Value::Array(Arc::new(Array::from_int32(arr2_1))),
            Value::Array(Arc::new(Array::from_int32(arr2_2))),
        ]));

        let result = val1.concat(val2);
        assert!(result.is_err());

        if let Err(MinarrowError::IncompatibleTypeError { message, .. }) = result {
            assert!(message.unwrap().contains("different lengths"));
        } else {
            panic!("Expected IncompatibleTypeError");
        }
    }

    #[test]
    fn test_value_concat_vec_value_type_mismatch() {
        // Element types don't match - first element is Int32, second is Float64
        let arr1_1 = IntegerArray::<i32>::from_slice(&[1, 2]);
        let arr1_2 = IntegerArray::<i32>::from_slice(&[10, 20]);
        let val1 = Value::VecValue(Arc::new(vec![
            Value::Array(Arc::new(Array::from_int32(arr1_1))),
            Value::Array(Arc::new(Array::from_int32(arr1_2))),
        ]));

        let arr2_1 = IntegerArray::<i32>::from_slice(&[3, 4]);
        let arr2_2 = crate::FloatArray::<f64>::from_slice(&[30.0, 40.0]);
        let val2 = Value::VecValue(Arc::new(vec![
            Value::Array(Arc::new(Array::from_int32(arr2_1))),
            Value::Array(Arc::new(Array::from_float64(arr2_2))),
        ]));

        let result = val1.concat(val2);
        assert!(result.is_err());

        // Should fail when trying to concat the second elements
        if let Err(MinarrowError::IncompatibleTypeError { .. }) = result {
            // Expected
        } else {
            panic!("Expected IncompatibleTypeError");
        }
    }
}
