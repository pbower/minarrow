//! # Consolidate Trait Module
//!
//! Provides uniform consolidation of chunked types into contiguous storage.
//!
//! ## Overview
//! The `Consolidate` trait materialises chunked/segmented data into a single
//! contiguous buffer, enabling efficient operations and compatibility
//! with APIs that require contiguous memory.
//!
//! - **SuperArray** -> **Array**: Merges all chunks into one contiguous array
//! - **SuperTable** -> **Table**: Merges all batches into one contiguous table
//! - **SuperArrayView** -> **Array**: Copies view chunks into owned array
//! - **SuperTableView** -> **Table**: Copies view batches into owned table
//!
//! ## Use Case
//! Sometimes processing returns chunked results to retain zero-copy and/or avoid buffer thrashing.
//! Call `.consolidate()` when you need contiguous memory.
//!
//! ## Example
//! ```ignore
//! use minarrow::{SuperArray, Consolidate};
//!
//! let chunked = SuperArray::from_chunks(vec![chunk1, chunk2, chunk3]);
//! // chunked has 3 separate memory regions
//!
//! let contiguous = chunked.consolidate();
//! // contiguous is a single Array with one buffer
//! ```

use crate::structs::bitmask::Bitmask;

/// Trait for consolidating chunked types into contiguous storage.
///
/// # Output Type
/// The `Output` associated type defines what the consolidated result is:
/// - `SuperArray::Output = Array`
/// - `SuperTable::Output = Table`
///
/// # When to Use
/// - After parallel batch processing that returns chunked results
/// - Before operations requiring contiguous memory (e.g., FFI, certain operations)
/// - When you need to serialise to formats requiring single buffers
///
/// # Naming
/// The consolidated result uses the name from the source. Call `.rename()` or
/// equivalent if you need a different name.
pub trait Consolidate {
    /// The type produced after consolidation.
    type Output;

    /// Consolidates chunked data into contiguous storage.
    ///
    /// Consumes `self` and returns a consolidated `Output`.
    fn consolidate(self) -> Self::Output;
}

// Helper Functions for Consolidation

/// Extends a result null mask from a source mask's range.
///
/// Handles all four cases of (result has mask, source has mask) combinations:
/// - Both have masks: extend from source range
/// - Result has mask, source doesn't: mark new bits as valid
/// - Source has mask, result doesn't: create new mask with previous bits valid
/// - Neither has mask: no-op
pub fn extend_null_mask(
    result_mask: &mut Option<Bitmask>,
    result_len: usize,
    source_mask: Option<&Bitmask>,
    offset: usize,
    len: usize,
) {
    match (result_mask.as_mut(), source_mask) {
        (Some(mask), Some(src)) => {
            mask.extend((offset..offset + len).map(|i| src.get(i)));
        }
        (Some(mask), None) => {
            // Source has no nulls, set all bits valid
            for _ in 0..len {
                mask.set(mask.len(), true);
            }
        }
        (None, Some(src)) => {
            // Create mask, all previous values valid, then copy from source
            let mut mask = Bitmask::new_set_all(result_len, true);
            mask.extend((offset..offset + len).map(|i| src.get(i)));
            *result_mask = Some(mask);
        }
        (None, None) => {}
    }
}

/// Macro for consolidating integer array slices by variant.
/// Directly extends from raw data buffers.
#[macro_export]
macro_rules! consolidate_int_variant {
    ($slices:expr, $variant:ident, $ty:ty) => {{
        use $crate::traits::consolidate::extend_null_mask;
        use $crate::traits::masked_array::MaskedArray;
        use $crate::enums::collections::numeric_array::NumericArray;
        use $crate::enums::array::Array;
        use $crate::structs::variants::integer::IntegerArray;
        use $crate::structs::bitmask::Bitmask;
        use std::sync::Arc;

        let total_len: usize = $slices.iter().map(|s| s.len()).sum();
        let has_nulls = $slices.iter().any(|s| {
            if let Array::NumericArray(NumericArray::$variant(arr)) = &s.array {
                arr.null_mask().is_some()
            } else {
                false
            }
        });

        let mut result = IntegerArray::<$ty>::with_capacity(total_len, has_nulls);
        let mut result_mask: Option<Bitmask> = if has_nulls {
            Some(Bitmask::default())
        } else {
            None
        };
        let mut current_len = 0;

        for slice in $slices {
            if let Array::NumericArray(NumericArray::$variant(arr)) = &slice.array {
                let data: &[$ty] = &arr.data[slice.offset..slice.offset + slice.len()];
                result.extend_from_slice(data);
                extend_null_mask(&mut result_mask, current_len, arr.null_mask(), slice.offset, slice.len());
                current_len += slice.len();
            }
        }

        if let Some(mask) = result_mask {
            result.set_null_mask(Some(mask));
        }
        Array::NumericArray(NumericArray::$variant(Arc::new(result)))
    }};
}

/// Macro for consolidating float array slices by variant.
/// Directly extends from raw data buffers.
#[macro_export]
macro_rules! consolidate_float_variant {
    ($slices:expr, $variant:ident, $ty:ty) => {{
        use $crate::traits::consolidate::extend_null_mask;
        use $crate::traits::masked_array::MaskedArray;
        use $crate::enums::collections::numeric_array::NumericArray;
        use $crate::enums::array::Array;
        use $crate::structs::variants::float::FloatArray;
        use $crate::structs::bitmask::Bitmask;
        use std::sync::Arc;

        let total_len: usize = $slices.iter().map(|s| s.len()).sum();
        let has_nulls = $slices.iter().any(|s| {
            if let Array::NumericArray(NumericArray::$variant(arr)) = &s.array {
                arr.null_mask().is_some()
            } else {
                false
            }
        });

        let mut result = FloatArray::<$ty>::with_capacity(total_len, has_nulls);
        let mut result_mask: Option<Bitmask> = if has_nulls {
            Some(Bitmask::default())
        } else {
            None
        };
        let mut current_len = 0;

        for slice in $slices {
            if let Array::NumericArray(NumericArray::$variant(arr)) = &slice.array {
                let data: &[$ty] = &arr.data[slice.offset..slice.offset + slice.len()];
                result.extend_from_slice(data);
                extend_null_mask(&mut result_mask, current_len, arr.null_mask(), slice.offset, slice.len());
                current_len += slice.len();
            }
        }

        if let Some(mask) = result_mask {
            result.set_null_mask(Some(mask));
        }
        Array::NumericArray(NumericArray::$variant(Arc::new(result)))
    }};
}

/// Macro for consolidating string array slices by variant.
/// Directly copies offsets and data buffers.
#[macro_export]
macro_rules! consolidate_string_variant {
    ($slices:expr, $variant:ident, $offset_ty:ty) => {{
        use $crate::traits::consolidate::extend_null_mask;
        use $crate::traits::masked_array::MaskedArray;
        use $crate::enums::collections::text_array::TextArray;
        use $crate::enums::array::Array;
        use $crate::structs::variants::string::StringArray;
        use $crate::structs::bitmask::Bitmask;
        use $crate::Vec64;
        use std::sync::Arc;

        let total_len: usize = $slices.iter().map(|s| s.len()).sum();

        let has_nulls = $slices.iter().any(|s| {
            if let Array::TextArray(TextArray::$variant(arr)) = &s.array {
                arr.null_mask().is_some()
            } else {
                false
            }
        });

        let mut result_offsets: Vec64<$offset_ty> = Vec64::with_capacity(total_len + 1);
        let mut result_data: Vec64<u8> = Vec64::new();
        let mut result_mask: Option<Bitmask> = if has_nulls {
            Some(Bitmask::default())
        } else {
            None
        };
        let mut current_len = 0;

        result_offsets.push(0 as $offset_ty);

        for slice in $slices {
            if let Array::TextArray(TextArray::$variant(arr)) = &slice.array {
                let offsets: &[$offset_ty] = arr.offsets.as_slice();
                let data: &[u8] = arr.data.as_slice();

                let start_idx = slice.offset;
                let end_idx = slice.offset + slice.len();

                let byte_start = offsets[start_idx] as usize;
                let byte_end = offsets[end_idx] as usize;
                let current_data_len = *result_offsets.last().unwrap();

                result_data.extend_from_slice(&data[byte_start..byte_end]);

                for i in (start_idx + 1)..=end_idx {
                    let adjusted_offset = (offsets[i] - offsets[start_idx]) + current_data_len;
                    result_offsets.push(adjusted_offset);
                }

                extend_null_mask(&mut result_mask, current_len, arr.null_mask(), slice.offset, slice.len());
                current_len += slice.len();
            }
        }

        let result = StringArray::<$offset_ty>::from_parts(result_offsets, result_data, result_mask);
        Array::TextArray(TextArray::$variant(Arc::new(result)))
    }};
}

/// Macro for consolidating temporal (datetime) array slices by variant.
/// Directly extends from raw data buffers.
/// Preserves the time_unit from the first slice.
#[cfg(feature = "datetime")]
#[macro_export]
macro_rules! consolidate_temporal_variant {
    ($slices:expr, $variant:ident, $ty:ty) => {{
        use $crate::traits::consolidate::extend_null_mask;
        use $crate::traits::masked_array::MaskedArray;
        use $crate::enums::collections::temporal_array::TemporalArray;
        use $crate::enums::array::Array;
        use $crate::structs::variants::datetime::DatetimeArray;
        use $crate::structs::bitmask::Bitmask;
        use std::sync::Arc;

        let total_len: usize = $slices.iter().map(|s| s.len()).sum();

        // Extract time_unit from first slice
        let time_unit = if let Array::TemporalArray(TemporalArray::$variant(arr)) = &$slices[0].array {
            arr.time_unit.clone()
        } else {
            panic!("Expected TemporalArray::{}", stringify!($variant))
        };

        let has_nulls = $slices.iter().any(|s| {
            if let Array::TemporalArray(TemporalArray::$variant(arr)) = &s.array {
                arr.null_mask().is_some()
            } else {
                false
            }
        });

        let mut result = DatetimeArray::<$ty>::with_capacity(total_len, has_nulls, Some(time_unit));
        let mut result_mask: Option<Bitmask> = if has_nulls {
            Some(Bitmask::default())
        } else {
            None
        };
        let mut current_len = 0;

        for slice in $slices {
            if let Array::TemporalArray(TemporalArray::$variant(arr)) = &slice.array {
                let data: &[$ty] = &arr.data[slice.offset..slice.offset + slice.len()];
                result.extend_from_slice(data);
                extend_null_mask(&mut result_mask, current_len, arr.null_mask(), slice.offset, slice.len());
                current_len += slice.len();
            }
        }

        if let Some(mask) = result_mask {
            result.set_null_mask(Some(mask));
        }
        Array::TemporalArray(TemporalArray::$variant(Arc::new(result)))
    }};
}
