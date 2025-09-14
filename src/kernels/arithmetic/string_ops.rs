// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under MIT License.

//! Basic string operations for arithmetic module

use crate::enums::error::KernelError;
use crate::traits::type_unions::Integer;
use crate::{MaskedArray, StringArray, Vec64};
use num_traits::NumCast;

/// Generic string concatenation for both String32 and String64 arrays
pub fn apply_str_str<T: Integer>(
    lhs: &std::sync::Arc<StringArray<T>>,
    rhs: &std::sync::Arc<StringArray<T>>,
) -> Result<StringArray<T>, KernelError> {
    if lhs.len() != rhs.len() {
        return Err(KernelError::LengthMismatch(format!(
            "String concatenation length mismatch: {} vs {}",
            lhs.len(),
            rhs.len()
        )));
    }

    let mut result_data = Vec64::new();
    let mut result_offsets = Vec64::with_capacity(lhs.len() + 1);
    result_offsets.push(NumCast::from(0).unwrap());

    let mut current_offset = NumCast::from(0).unwrap();

    for i in 0..lhs.len() {
        // Get strings from both arrays
        let left_str = lhs.get_str(i).unwrap_or("");
        let right_str = rhs.get_str(i).unwrap_or("");

        // Concatenate
        let concatenated = format!("{}{}", left_str, right_str);
        let bytes = concatenated.as_bytes();

        // Add to result data
        result_data.extend_from_slice(bytes);
        current_offset = current_offset + NumCast::from(bytes.len()).unwrap();
        result_offsets.push(current_offset);
    }

    Ok(StringArray {
        data: result_data.into(),
        offsets: result_offsets.into(),
        null_mask: None, // TODO: Handle null masks properly
    })
}
