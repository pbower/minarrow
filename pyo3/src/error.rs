//! # Error Module for minarrow-pyo3
//!
//! Provides error types and conversions between MinArrow errors and Python exceptions.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use thiserror::Error;

/// Error type for minarrow-pyo3 operations.
#[derive(Error, Debug)]
pub enum PyMinarrowError {
    /// Error from MinArrow core library.
    #[error("MinArrow error: {0}")]
    Minarrow(#[from] minarrow::enums::error::MinarrowError),

    /// Error during FFI operations.
    #[error("FFI error: {0}")]
    Ffi(String),

    /// Type conversion error.
    #[error("Type error: {0}")]
    Type(String),

    /// Unsupported Arrow type.
    #[error("Unsupported type: {0}")]
    UnsupportedType(String),

    /// PyArrow import error.
    #[error("PyArrow error: {0}")]
    PyArrow(String),
}

impl From<PyMinarrowError> for PyErr {
    fn from(err: PyMinarrowError) -> PyErr {
        match err {
            PyMinarrowError::Type(msg) => PyTypeError::new_err(msg),
            PyMinarrowError::UnsupportedType(msg) => PyTypeError::new_err(msg),
            PyMinarrowError::Ffi(msg) => PyRuntimeError::new_err(msg),
            PyMinarrowError::PyArrow(msg) => PyValueError::new_err(msg),
            PyMinarrowError::Minarrow(e) => {
                // Map MinArrow errors to appropriate Python exceptions
                match e {
                    minarrow::enums::error::MinarrowError::TypeError { .. } => {
                        PyTypeError::new_err(e.to_string())
                    }
                    minarrow::enums::error::MinarrowError::IncompatibleTypeError { .. } => {
                        PyTypeError::new_err(e.to_string())
                    }
                    minarrow::enums::error::MinarrowError::IndexError(_) => {
                        pyo3::exceptions::PyIndexError::new_err(e.to_string())
                    }
                    minarrow::enums::error::MinarrowError::Overflow { .. } => {
                        pyo3::exceptions::PyOverflowError::new_err(e.to_string())
                    }
                    _ => PyRuntimeError::new_err(e.to_string()),
                }
            }
        }
    }
}

/// Result type alias for minarrow-pyo3 operations.
pub type PyMinarrowResult<T> = Result<T, PyMinarrowError>;
