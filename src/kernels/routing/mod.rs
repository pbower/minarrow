// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under MIT License.

//! # Routing Module
//!
//! Intelligent routing and dispatching for kernel operations
//! including broadcasting, type promotion, and operation dispatch.

pub mod arithmetic;
pub mod broadcast;

pub use arithmetic::resolve_binary_arithmetic;
pub use broadcast::{broadcast_length_1_array, maybe_broadcast_scalar_array};
