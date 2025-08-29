//! # **Error Module** - Custom *Minarrow* Error Type
//! 
//! Defines the unified error type for Minarrow.
//! 
//! Also includes a KernelError type for this crate and downstream SIMD-kernels
//! 
//! ## Covers 
//! - Array length mismatches, overflow, lossy casts, null handling,
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
    KernelError(Option<String>),
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
            MinarrowError::KernelError(message) =>                         
            if let Some(msg) = message {
                write!(f, "Kernel error: {}", msg)
            } else {
                write!(f, "Kernel error")
            }
        }
    }
}

impl Error for MinarrowError {}


/// Error type for all kernel operations.
///
/// Each variant includes a contextual message string providing specific details
/// about the error condition, enabling precise debugging and error reporting.
#[derive(Debug, Clone)]
pub enum KernelError {
    /// Data type mismatch between operands or unsupported type combinations.
    TypeMismatch(String),
    
    /// Array length mismatch between operands.
    LengthMismatch(String),
    
    /// Invalid operator for the given operands or context.
    OperatorMismatch(String),
    
    /// Unsupported data type for the requested operation.
    UnsupportedType(String),
    
    /// Column or field not found in structured data.
    ColumnNotFound(String),
    
    /// Invalid arguments provided to kernel function.
    InvalidArguments(String),
    
    /// Planning or configuration error.
    Plan(String),
    
    /// Array index or memory access out of bounds.
    OutOfBounds(String),
    
    /// Division by zero or similar mathematical errors.
    DivideByZero(String),
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::TypeMismatch(msg) => write!(f, "Type mismatch: {}", msg),
            KernelError::LengthMismatch(msg) => write!(f, "Length mismatch: {}", msg),
            KernelError::OperatorMismatch(msg) => write!(f, "Operator mismatch: {}", msg),
            KernelError::UnsupportedType(msg) => write!(f, "Unsupported type: {}", msg),
            KernelError::ColumnNotFound(msg) => write!(f, "Column not found: {}", msg),
            KernelError::InvalidArguments(msg) => write!(f, "Invalid arguments: {}", msg),
            KernelError::Plan(msg) => write!(f, "Planning error: {}", msg),
            KernelError::OutOfBounds(msg) => write!(f, "Out of bounds: {}", msg),
            KernelError::DivideByZero(msg) => write!(f, "Divide by Zero error: {}", msg),
        }
    }
}

impl Error for KernelError {}

/// Creates a formatted error message for length mismatches between left-hand side (LHS) and right-hand side (RHS) arrays.
///
/// # Arguments
/// * `fname` - Function name where the mismatch occurred
/// * `lhs` - Length of the left-hand side array
/// * `rhs` - Length of the right-hand side array
///
/// # Returns
/// A formatted error message string
pub fn log_length_mismatch(fname: String, lhs: usize, rhs: usize) -> String {
    return format!("{} => Length mismatch: LHS {} RHS {}", fname, lhs, rhs);
}
