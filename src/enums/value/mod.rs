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

mod conversions;

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
use std::sync::Arc;

#[cfg(feature = "chunked")]
use crate::{SuperArray, SuperTable};

#[cfg(feature = "views")]
use crate::{ArrayV, TableV};

#[cfg(all(feature = "chunked", feature = "views"))]
use crate::{SuperArrayV, SuperTableV};

mod impls;

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
    // Length and Shape

    /// Computes the logical row/element count for the batch's input `Value`.
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

// Also see typed accessors in ./conversions.rs and trait impls in ./impls.rs
