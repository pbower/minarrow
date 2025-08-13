//! # Utilities - *Internal Helper Utilities*
//! 
//! A small collection of internal utilities that support validation, parsing, and text conversion
//! elsewhere within the crate.


use std::{fmt::Display, sync::Arc};

use crate::{Bitmask, Float, FloatArray, Integer, IntegerArray, StringArray, TextArray};
use crate::traits::masked_array::MaskedArray;

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
        use chrono::{NaiveDateTime, NaiveDate, NaiveTime, DateTime, Utc, TimeZone};

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
pub fn int_to_text_array<T: Display + Integer>(
    arr: &Arc<IntegerArray<T>>,
) -> TextArray {
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
pub fn float_to_text_array<T: Display + Float>(
    arr: &Arc<FloatArray<T>>,
) -> TextArray {
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
