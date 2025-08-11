use crate::{enums::error::MinarrowError, traits::custom_value::CustomValue, Array, ArrayV, Bitmask, BitmaskV, SuperArray, SuperArrayV, SuperTable, SuperTableV, Field, FieldArray, Table, TableV};
#[cfg(feature = "scalar_type")]
use crate::Scalar;
#[cfg(feature = "cube")]
use crate::Cube;
use std::{convert::From, sync::Arc};
#[cfg(feature = "matrix")]
use crate::Matrix;
use std::convert::TryFrom;

#[cfg(feature = "collection_views")]
use crate::{ TextArrayV, NumericArrayV};

#[cfg(feature = "collection_views")]
#[cfg(feature = "datetime")]
use crate::TemporalArrayV;

/// Unified value enum representing any supported data structure.
///
/// # Details
/// - Wraps scalar values, arrays, array windows, full tables, or table windows
/// under a single type for function signatures and downstream dispatch.
/// - This can be useful when you need a global *type universe*.
/// - It is not part of the `Arrow` specification, but is useful
/// because of the flexibility it adds unifying all types to a single one.
/// For example, to return `Result<Value, Error>`, particularly in engine contexts.
/// - It's enabled optionally via the `value_type` feature.
/// 
/// # Usage
/// You can also use it to hold a custom type under the `Custom` entry.
/// As long as the object implements `Debug`, `Clone`, and `PartialEq`,
/// remains `Send + Sync`, and implements `Any` it can be stored in `Value::Custom`.
/// `Any` is implemented automatically for all Rust types with a `'static` lifetime. 
#[derive(Debug, Clone)]
pub enum Value {
    #[cfg(feature = "scalar_type")]
    Scalar(Scalar),
    Array(Array),
    ArrayView(ArrayV),
    Table(Table),
    TableView(TableV),    
    #[cfg(feature = "collection_views")]
    NumericArrayView(NumericArrayV),
    #[cfg(feature = "collection_views")]
    TextArrayView(TextArrayV),
    #[cfg(feature = "collection_views")]
    #[cfg(feature = "datetime")]
    TemporalArrayView(TemporalArrayV),
    Bitmask(Bitmask),
    BitmaskView(BitmaskV),
    #[cfg(feature = "chunked")]
    ChunkedArray(SuperArray),
    #[cfg(feature = "slicing_extras")]
    #[cfg(feature = "chunked")]
    ChunkedArrayView(SuperArrayV),
    #[cfg(feature = "chunked")]
    ChunkedTable(SuperTable),
    #[cfg(feature = "slicing_extras")]
    #[cfg(feature = "chunked")]
    ChunkedTableView(SuperTableV),
    FieldArray(FieldArray),
    Field(Field),
    #[cfg(feature = "matrix")]
    Matrix(Matrix),
    #[cfg(feature = "cube")]
    Cube(Cube),
    VecValue(Vec<Value>), 
    // For recursive
    BoxValue(Box<Value>),
    ArcValue(Arc<Value>),
    Tuple2((Box<Value>, Box<Value>)),
    Tuple3((Box<Value>, Box<Value>, Box<Value>)),
    Tuple4((Box<Value>, Box<Value>, Box<Value>, Box<Value>)),
    Tuple5((Box<Value>, Box<Value>, Box<Value>, Box<Value>, Box<Value>)),
    Tuple6((Box<Value>, Box<Value>, Box<Value>, Box<Value>, Box<Value>, Box<Value>)),
    
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

/// Implements `PartialEq` for `Value`
/// 
/// This includes special handling for the `Custom` type.
impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        use Value::*;
        match (self, other) {
            #[cfg(feature = "scalar_type")]
            (Scalar(a), Scalar(b)) => a == b,
            (Array(a), Array(b)) => a == b,
            (ArrayView(a), ArrayView(b)) => a == b,
            (Table(a), Table(b)) => a == b,
            (TableView(a), TableView(b)) => a == b,
            #[cfg(feature = "collection_views")]
            (NumericArrayView(a), NumericArrayView(b)) => a == b,
            #[cfg(feature = "collection_views")]
            (TextArrayView(a), TextArrayView(b)) => a == b,
            #[cfg(feature = "collection_views")]
            #[cfg(feature = "datetime")]
            (TemporalArrayView(a), TemporalArrayView(b)) => a == b,
            (Bitmask(a), Bitmask(b)) => a == b,
            (BitmaskView(a), BitmaskView(b)) => a == b,
            #[cfg(feature = "chunked")]
            (ChunkedArray(a), ChunkedArray(b)) => a == b,
            #[cfg(feature = "chunked")]
            (ChunkedArrayView(a), ChunkedArrayView(b)) => a == b,
            (FieldArray(a), FieldArray(b)) => a == b,
            (Field(a), Field(b)) => a == b,
            #[cfg(feature = "matrix")]
            (Matrix(a), Matrix(b)) => a == b,
            #[cfg(feature = "cube")]
            (Cube(a), Cube(b)) => a == b,
            (Custom(a), Custom(b)) => a.eq_box(&**b),
            (VecValue(a), VecValue(b)) => a == b,
            (BoxValue(a), BoxValue(b)) => a == b,
            (ArcValue(a), ArcValue(b)) => a == b,
            (Tuple2(a), Tuple2(b)) => a.0 == b.0 && a.1 == b.1,
            (Tuple3(a), Tuple3(b)) => a.0 == b.0 && a.1 == b.1 && a.2 == b.2,
            (Tuple4(a), Tuple4(b)) => a.0 == b.0 && a.1 == b.1 && a.2 == b.2 && a.3 == b.3,
            (Tuple5(a), Tuple5(b)) => a.0 == b.0 && a.1 == b.1 && a.2 == b.2 && a.3 == b.3 && a.4 == b.4,
            (Tuple6(a), Tuple6(b)) => a.0 == b.0 && a.1 == b.1 && a.2 == b.2 && a.3 == b.3 && a.4 == b.4 && a.5 == b.5,
            _ => false,
        }
    }
}



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

// Array-like types
impl_value_from!(Array: Array);
impl_value_from!(ArrayView: ArrayV);
impl_value_from!(Table: Table);
impl_value_from!(TableView: TableV);
impl_value_from!(Bitmask: Bitmask);
impl_value_from!(BitmaskView: BitmaskV);
impl_value_from!(FieldArray: FieldArray);
impl_value_from!(Field: Field);

#[cfg(feature = "collection_views")]
impl_value_from!(NumericArrayView: NumericArrayV);
#[cfg(feature = "collection_views")]
impl_value_from!(TextArrayView: TextArrayV);
#[cfg(feature = "collection_views")]
#[cfg(feature = "datetime")]
impl_value_from!(TemporalArrayView: TemporalArrayV);

#[cfg(feature = "chunked")]
impl_value_from!(ChunkedArray: SuperArray);
#[cfg(feature = "chunked")]
impl_value_from!(ChunkedArrayView: SuperArrayV);

#[cfg(feature = "matrix")]
impl_value_from!(Matrix: Matrix);
#[cfg(feature = "cube")]
impl_value_from!(Cube: Cube);

// TryFrom for Array-like types
impl_tryfrom_value!(Array: Array);
impl_tryfrom_value!(ArrayView: ArrayV);
impl_tryfrom_value!(Table: Table);
impl_tryfrom_value!(TableView: TableV);
impl_tryfrom_value!(Bitmask: Bitmask);
impl_tryfrom_value!(BitmaskView: BitmaskV);
impl_tryfrom_value!(FieldArray: FieldArray);
impl_tryfrom_value!(Field: Field);

#[cfg(feature = "collection_views")]
impl_tryfrom_value!(NumericArrayView: NumericArrayV);
#[cfg(feature = "collection_views")]
impl_tryfrom_value!(TextArrayView: TextArrayV);
#[cfg(feature = "collection_views")]
#[cfg(feature = "datetime")]
impl_tryfrom_value!(TemporalArrayView: TemporalArrayV);

#[cfg(feature = "chunked")]
impl_tryfrom_value!(ChunkedArray: SuperArray);
#[cfg(feature = "chunked")]
impl_tryfrom_value!(ChunkedArrayView: SuperArrayV);

#[cfg(feature = "matrix")]
impl_tryfrom_value!(Matrix: Matrix);
#[cfg(feature = "cube")]
impl_tryfrom_value!(Cube: Cube);

// Recursive containers
impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Self {
        Value::VecValue(v)
    }
}

impl From<(Value, Value)> for Value {
    fn from(v: (Value, Value)) -> Self {
        Value::Tuple2((Box::new(v.0), Box::new(v.1)))
    }
}

impl From<(Value, Value, Value)> for Value {
    fn from(v: (Value, Value, Value)) -> Self {
        Value::Tuple3((Box::new(v.0), Box::new(v.1), Box::new(v.2)))
    }
}

impl From<(Value, Value, Value, Value)> for Value {
    fn from(v: (Value, Value, Value, Value)) -> Self {
        Value::Tuple4((Box::new(v.0), Box::new(v.1), Box::new(v.2), Box::new(v.3)))
    }
}

impl From<(Value, Value, Value, Value, Value)> for Value {
    fn from(v: (Value, Value, Value, Value, Value)) -> Self {
        Value::Tuple5((
            Box::new(v.0),
            Box::new(v.1),
            Box::new(v.2),
            Box::new(v.3),
            Box::new(v.4),
        ))
    }
}

impl From<(Value, Value, Value, Value, Value, Value)> for Value {
    fn from(v: (Value, Value, Value, Value, Value, Value)) -> Self {
        Value::Tuple6((
            Box::new(v.0),
            Box::new(v.1),
            Box::new(v.2),
            Box::new(v.3),
            Box::new(v.4),
            Box::new(v.5),
        ))
    }
}

// TryFrom for recursive containers
impl TryFrom<Value> for Vec<Value> {
    type Error = MinarrowError;
    fn try_from(v: Value) -> Result<Self, Self::Error> {
        match v {
            Value::VecValue(inner) => Ok(inner),
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
            Value::Tuple2((a, b)) => Ok((*a, *b)),
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
            Value::Tuple3((a, b, c)) => Ok((*a, *b, *c)),
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
            Value::Tuple4((a, b, c, d)) => Ok((*a, *b, *c, *d)),
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
            Value::Tuple5((a, b, c, d, e)) => Ok((*a, *b, *c, *d, *e)),
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
            Value::Tuple6((a, b, c, d, e, f)) => Ok((*a, *b, *c, *d, *e, *f)),
            _ => Err(MinarrowError::TypeError {
                from: "Value",
                to: "(Value, Value, Value, Value, Value, Value)",
                message: Some("Expected Tuple6 variant".to_owned()),
            }),
        }
    }
}