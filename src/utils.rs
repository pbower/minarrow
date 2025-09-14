//! # **Utilities** - *Internal Helper Utilities*
//!
//! A small collection of internal utilities that support validation, parsing, and text conversion
//! elsewhere within the crate.

#[cfg(feature = "fast_hash")]
use ahash::AHashSet as HashSet;
#[cfg(not(feature = "fast_hash"))]
use std::collections::HashSet;
use std::simd::{LaneCount, Mask, MaskElement, SupportedLaneCount};
use std::{fmt::Display, sync::Arc};

use crate::enums::error::KernelError;
#[cfg(feature = "chunked")]
use crate::enums::error::MinarrowError;
#[cfg(feature = "chunked")]
use crate::structs::field_array::create_field_for_array;
use crate::traits::masked_array::MaskedArray;
#[cfg(feature = "chunked")]
use crate::{Array, FieldArray, SuperArray};
use crate::{
    Bitmask, CategoricalArray, Float, FloatArray, Integer, IntegerArray, StringArray, TextArray,
};

#[inline(always)]
pub fn validate_null_mask_len(data_len: usize, null_mask: &Option<Bitmask>) {
    if let Some(mask) = null_mask {
        assert_eq!(
            mask.len(),
            data_len,
            "Validation Error: Null mask length ({}) does not match data length ({})",
            mask.len(),
            data_len
        );
    }
}

/// Parses a string into a timestamp in milliseconds since the Unix epoch.
/// Returns `Some(i64)` on success, or `None` if the string could not be parsed.
///
/// Attempts common ISO8601/RFC3339 and `%Y-%m-%d` formats if the `chrono`
/// feature is enabled.
pub fn parse_datetime_str(s: &str) -> Option<i64> {
    // Empty string is always None/null
    if s.is_empty() {
        return None;
    }

    #[cfg(feature = "chrono")]
    {
        use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};

        // Try to parse as RFC3339/ISO8601 string
        if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
            return Some(dt.timestamp_millis());
        }
        // Try parsing as full date-time (no timezone)
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
            // Assume UTC
            return Some(Utc.from_utc_datetime(&dt).timestamp_millis());
        }
        // Try parsing as date only
        if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            let dt = date.and_hms_opt(0, 0, 0)?;
            return Some(Utc.from_utc_datetime(&dt).timestamp_millis());
        }
        // Try parsing as time only (today's date)
        if let Ok(time) = NaiveTime::parse_from_str(s, "%H:%M:%S") {
            let today = Utc::now().date_naive();
            let dt = NaiveDateTime::new(today, time);
            return Some(Utc.from_utc_datetime(&dt).timestamp_millis());
        }
    }

    // Fallback: parse as i64 integer (milliseconds since epoch)
    if let Ok(ms) = s.parse::<i64>() {
        return Some(ms);
    }

    None
}

/// Converts an integer array to a String32 TextArray, preserving nulls.
pub fn int_to_text_array<T: Display + Integer>(arr: &Arc<IntegerArray<T>>) -> TextArray {
    let mut strings: Vec<String> = Vec::with_capacity(arr.len());
    for i in 0..arr.len() {
        if arr.is_null(i) {
            strings.push(String::new()); // This "" keeps the correct length
        } else {
            strings.push(format!("{}", arr.data[i]));
        }
    }
    let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
    let string_array = StringArray::<u32>::from_vec(refs, arr.null_mask.clone());
    TextArray::String32(Arc::new(string_array))
}

/// Converts a float array to a String32 TextArray, preserving nulls.
pub fn float_to_text_array<T: Display + Float>(arr: &Arc<FloatArray<T>>) -> TextArray {
    let mut strings: Vec<String> = Vec::with_capacity(arr.len());
    for i in 0..arr.len() {
        if arr.is_null(i) {
            strings.push(String::new()); // This "" keeps the correct length
        } else {
            strings.push(format!("{}", arr.data[i]));
        }
    }
    let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
    let string_array = StringArray::<u32>::from_vec(refs, arr.null_mask.clone());
    TextArray::String32(Arc::new(string_array))
}

/// Validates that two lengths are equal for binary kernel operations.
///
/// Critical validation function ensuring input arrays have matching lengths before performing
/// binary operations like comparisons, arithmetic, or logical operations. Prevents undefined
/// behaviour and provides clear error diagnostics when length mismatches occur.
///
/// # Parameters
/// - `label`: Descriptive context label for error reporting (e.g., "compare numeric")
/// - `a`: Length of the first input array or data structure
/// - `b`: Length of the second input array or data structure
///
/// # Returns
/// `Ok(())` if lengths are equal, otherwise `KernelError::LengthMismatch` with diagnostic details.
#[inline(always)]
pub fn confirm_equal_len(label: &str, a: usize, b: usize) -> Result<(), KernelError> {
    if a != b {
        return Err(KernelError::LengthMismatch(format!(
            "{}: length mismatch (lhs: {}, rhs: {})",
            label, a, b
        )));
    }
    Ok(())
}

/// SIMD Alignment check. Returns true if the slice is properly
/// 64-byte aligned for SIMD operations, false otherwise.
#[inline(always)]
pub fn is_simd_aligned<T>(slice: &[T]) -> bool {
    if slice.is_empty() {
        true
    } else {
        (slice.as_ptr() as usize) % 64 == 0
    }
}

/// Creates a SIMD mask from a bitmask window for vectorised conditional operations.
///
/// Converts a contiguous section of a bitmask into a SIMD mask.
/// The resulting mask can be used to selectively enable/disable SIMD lanes during
/// computation, providing efficient support for sparse or conditional operations.
///
/// # Type Parameters
/// - `T`: Mask element type implementing `MaskElement` (typically i8, i16, i32, or i64)
/// - `N`: Number of SIMD lanes, must match the SIMD vector width for the target operation
///
/// # Parameters
/// - `mask`: Source bitmask containing validity information
/// - `offset`: Starting bit offset within the bitmask
/// - `len`: Maximum number of bits to consider (bounds checking)
///
/// # Returns
/// A `Mask<T, N>` where each lane corresponds to the validity of the corresponding input element.
/// Lanes beyond `len` are set to false for safety.
///
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::utils::simd_mask;
///
/// // Create 8-lane mask for conditional SIMD operations  
/// let mask: Mask<i32, 8> = simd_mask(&bitmask, 0, 64);
/// let result = simd_vector.select(mask, default_vector);
/// ```
#[inline(always)]
pub fn simd_mask<T: MaskElement, const N: usize>(
    mask: &Bitmask,
    offset: usize,
    len: usize,
) -> Mask<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let mut bits = [false; N];
    for l in 0..N {
        let idx = offset + l;
        bits[l] = idx < len && unsafe { mask.get_unchecked(idx) };
    }
    Mask::from_array(bits)
}

/// Checks the mask capacity is large enough
/// Used so we can avoid bounds checks in the hot loop
#[inline(always)]
pub fn confirm_mask_capacity(cmp_len: usize, mask: Option<&Bitmask>) -> Result<(), KernelError> {
    if let Some(m) = mask {
        confirm_capacity("mask (Bitmask)", m.capacity(), cmp_len)?;
    }
    Ok(())
}

/// Validates that actual capacity matches expected capacity for kernel operations.
///
/// Validation function used throughout the kernel library to ensure data structure
/// capacities are correct before performing operations. Prevents buffer overruns and ensures
/// memory safety by catching capacity mismatches early with descriptive error messages.
///
/// # Parameters
/// - `label`: Descriptive label for the validation context (used in error messages)
/// - `actual`: The actual capacity of the data structure being validated
/// - `expected`: The expected capacity required for the operation
///
/// # Returns
/// `Ok(())` if capacities match, otherwise `KernelError::InvalidArguments` with detailed message.
///
/// # Error Conditions
/// Returns `KernelError::InvalidArguments` when `actual != expected`, providing a clear
/// error message indicating the mismatch and context.
#[inline(always)]
pub fn confirm_capacity(label: &str, actual: usize, expected: usize) -> Result<(), KernelError> {
    if actual != expected {
        return Err(KernelError::InvalidArguments(format!(
            "{}: capacity mismatch (expected {}, got {})",
            label, expected, actual
        )));
    }
    Ok(())
}

/// Estimate cardinality ratio on a sample from a CategoricalArray.
/// Used to quickly figure out the optimal strategy when comparing
/// StringArray and CategoricalArrays.
#[inline(always)]
pub fn estimate_categorical_cardinality(cat: &CategoricalArray<u32>, sample_size: usize) -> f64 {
    let len = cat.data.len();
    if len == 0 {
        return 0.0;
    }
    let mut seen = HashSet::with_capacity(sample_size.min(len));
    let step = (len / sample_size.max(1)).max(1);
    for i in (0..len).step_by(step) {
        let s = unsafe { cat.get_str_unchecked(i) };
        seen.insert(s);
        if seen.len() >= sample_size {
            break;
        }
    }
    (seen.len() as f64) / (sample_size.min(len) as f64)
}

/// Estimate cardinality ratio on a sample from a StringArray.
/// Used to quickly figure out the optimal strategy when comparing
/// StringArray and CategoricalArrays.
#[inline(always)]
pub fn estimate_string_cardinality<T: Integer>(arr: &StringArray<T>, sample_size: usize) -> f64 {
    let len = arr.len();
    if len == 0 {
        return 0.0;
    }
    let mut seen = HashSet::with_capacity(sample_size.min(len));
    let step = (len / sample_size.max(1)).max(1);
    for i in (0..len).step_by(step) {
        let s = unsafe { arr.get_str_unchecked(i) };
        seen.insert(s);
        if seen.len() >= sample_size {
            break;
        }
    }
    (seen.len() as f64) / (sample_size.min(len) as f64)
}

#[cfg(feature = "chunked")]
/// Helper function to handle mask union between Array and SuperArray
fn union_array_superarray_masks(
    array: &Array,
    super_array: &SuperArray,
) -> Result<Option<Bitmask>, MinarrowError> {
    let array_mask = array.null_mask();
    let super_array_masks: Vec<_> = super_array
        .chunks()
        .iter()
        .map(|chunk| chunk.array.null_mask())
        .collect();

    let super_array_concatenated_mask = if super_array_masks.iter().any(|m| m.is_some()) {
        let mut concatenated_bits = Vec::new();
        for (chunk, mask_opt) in super_array.chunks().iter().zip(super_array_masks.iter()) {
            if let Some(mask) = mask_opt {
                concatenated_bits.extend((0..mask.len()).map(|i| mask.get(i)));
            } else {
                concatenated_bits.extend(std::iter::repeat(true).take(chunk.array.len()));
            }
        }
        Some(Bitmask::from_bools(&concatenated_bits))
    } else {
        None
    };

    match (array_mask, super_array_concatenated_mask) {
        (Some(arr_mask), Some(super_mask)) => {
            if arr_mask.len() == super_mask.len() {
                Ok(Some(arr_mask.union(&super_mask)))
            } else {
                Err(MinarrowError::ShapeError {
                    message: format!(
                        "Mask lengths must match for union: {} vs {}",
                        arr_mask.len(),
                        super_mask.len()
                    ),
                })
            }
        }
        (Some(mask), None) => Ok(Some(mask.clone())),
        (None, Some(mask)) => Ok(Some(mask)),
        (None, None) => Ok(None),
    }
}

#[cfg(feature = "chunked")]
/// Helper function to create aligned chunks from Array to match SuperArray chunk structure
pub fn create_aligned_chunks_from_array(
    array: Array,
    super_array: &SuperArray,
    field_name: &str,
) -> Result<SuperArray, MinarrowError> {
    // Check total lengths match
    if array.len() != super_array.len() {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "Array and SuperArray must have same total length for broadcasting: {} vs {}",
                array.len(),
                super_array.len()
            ),
        });
    }

    // Union the masks
    let full_mask = union_array_superarray_masks(&array, super_array)?;

    // Extract chunk lengths from SuperArray
    let chunk_lengths: Vec<usize> = super_array
        .chunks()
        .into_iter()
        .map(|chunk| chunk.array.len())
        .collect();

    // Create aligned chunks from Array using view function
    let mut start = 0;
    let mut mask_start = 0;
    let chunks: Result<Vec<_>, _> = chunk_lengths
        .iter()
        .map(|&chunk_len| {
            let end = start + chunk_len;
            if end > array.len() {
                return Err(MinarrowError::ShapeError {
                    message: format!(
                        "Chunk alignment failed: index {} out of bounds for length {}",
                        end,
                        array.len()
                    ),
                });
            }
            let view = array.view(start, chunk_len);
            let mut array_chunk = view.array.slice_clone(view.offset, view.len());

            // Apply portion of full_mask to this chunk
            if let Some(ref mask) = full_mask {
                let mask_end = mask_start + chunk_len;
                let chunk_mask_bits: Vec<bool> =
                    (mask_start..mask_end).map(|i| mask.get(i)).collect();
                let chunk_mask = Bitmask::from_bools(&chunk_mask_bits);
                array.set_null_mask(&mut array_chunk, chunk_mask);
                mask_start = mask_end;
            }

            start = end;
            let first_super_chunk = &super_array.chunks()[0].array;
            let field =
                create_field_for_array(field_name, &array_chunk, Some(first_super_chunk), None);
            Ok(FieldArray::new(field, array_chunk))
        })
        .collect();

    Ok(SuperArray::from_chunks(chunks?))
}
