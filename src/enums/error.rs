//! # Error Module - Custom *Minarrow* Error Type
//! 
//! Defines the unified error type for Minarrow.
//! 
//! ## Features 
//! - Covers array length mismatches, overflow, lossy casts, null handling,
//! type incompatibility, and invalid conversions.  
//! - Implements `Display` for readable output and `Error` for integration
//! with standard Rust error handling.

use std::fmt;
use std::error::Error;

/// Catch all error type for `Minarrow`
#[derive(Debug, PartialEq)]
pub enum MinarrowError {
    ColumnLengthMismatch {
        col: usize,
        expected: usize,
        found: usize,
    },
    Overflow {
        value: String,
        target: &'static str,
    },
    LossyCast {
        value: String,
        target: &'static str,
    },
    TypeError {
        from: &'static str,
        to: &'static str,
        message: Option<String>,
    },
    NullError {
        message: Option<String>,
    },
    IncompatibleTypeError {
        from: &'static str,
        to: &'static str,
        message: Option<String>,
    },
}

impl fmt::Display for MinarrowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MinarrowError::ColumnLengthMismatch { col, expected, found } => {
                write!(
                    f,
                    "Column length mismatch in column {}: expected {}, found {}.",
                    col, expected, found
                )
            }
            MinarrowError::Overflow { value, target } => {
                write!(f, "Overflow: value '{}' cannot be represented in type '{}'.", value, target)
            }
            MinarrowError::LossyCast { value, target } => {
                write!(f, "Lossy cast: value '{}' loses precision or cannot be exactly represented as '{}'.", value, target)
            }
            MinarrowError::TypeError { from, to, message } => {
                if let Some(msg) = message {
                    write!(f, "Type error: cannot cast from '{}' to '{}': {}", from, to, msg)
                } else {
                    write!(f, "Type error: cannot cast from '{}' to '{}'.", from, to)
                }
            }
            MinarrowError::NullError { message } => {
                if let Some(msg) = message {
                    write!(f, "Null error: {}", msg)
                } else {
                    write!(f, "Null error: nulls cannot be represented in target type.")
                }
            }
            MinarrowError::IncompatibleTypeError { from, to, message } => {
                if let Some(msg) = message {
                    write!(f, "Incompatible type error: cannot convert from '{}' to '{}': {}", from, to, msg)
                } else {
                    write!(f, "Incompatible type error: cannot convert from '{}' to '{}'.", from, to)
                }
            }
        }
    }
}

impl Error for MinarrowError {}
