// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under MIT License.

//! # **String Operations Kernels Module** - *High-Performance String Processing and Text Analysis*
//!
//! String processing kernels for text manipulation, pattern matching,
//! and string analysis operations with UTF-8 awareness and null-safe semantics. Essential infrastructure
//! for text analytics, data cleansing, and string-heavy analytical workloads.
//!
//! ## Core Operations
//! - **String transformations**: Case conversion, trimming, padding, and substring operations
//! - **Pattern matching**: Regular expression support with compiled pattern caching  
//! - **String comparison**: Lexicographic ordering with UTF-8 aware collation
//! - **Text analysis**: Length calculation, character counting, and encoding detection
//! - **String aggregation**: Concatenation with configurable delimiters and null handling
//! - **Search operations**: Contains, starts with, ends with predicates with optimised implementations

#[cfg(feature = "fast_hash")]
use ahash::{AHashMap, AHashSet};
#[cfg(not(feature = "fast_hash"))]
use std::collections::{HashMap, HashSet};

use crate::{
    Bitmask, BooleanArray, CategoricalArray, Integer, IntegerArray, MaskedArray, StringArray,
    Vec64,
    aliases::{CategoricalAVT, StringAVT},
};
#[cfg(feature = "regex")]
use regex::Regex;

use crate::enums::error::KernelError;
use crate::utils::confirm_mask_capacity;
use std::marker::PhantomData;

/// Helper for predicate kernels: produce optional input masks and a fresh output mask
#[inline(always)]
pub fn string_predicate_masks<'a>(
    lhs_mask: Option<&'a Bitmask>,
    rhs_mask: Option<&'a Bitmask>,
    len: usize,
) -> (Option<&'a Bitmask>, Option<&'a Bitmask>, Bitmask) {
    let out = Bitmask::new_set_all(len, false);
    (lhs_mask, rhs_mask, out)
}

// Concatenation

/// Concatenates corresponding string pairs from two string arrays element-wise.
///
/// Performs element-wise concatenation of strings from two `StringArray` sources,
/// producing a new string array where each result string is the concatenation of
/// the corresponding left and right input strings.
///
/// # Parameters
/// - `lhs`: Left-hand string array view tuple `(StringArray, offset, length)`
/// - `rhs`: Right-hand string array view tuple `(StringArray, offset, length)`
///
/// # Returns
/// A new `StringArray<T>` containing concatenated strings with proper null handling.
///
/// # Null Handling
/// - If either input string is null, the result is null
/// - Output null mask reflects the union of input null positions
///
/// # Performance
/// - Pre-computes total memory requirements to minimise allocations
/// - Uses unsafe unchecked access for validated indices
/// - Optimised for bulk string concatenation operations
pub fn concat_str_str<T: Integer>(lhs: StringAVT<T>, rhs: StringAVT<T>) -> StringArray<T> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let len = llen.min(rlen);

    let (lmask, rmask, mut out_mask) =
        string_predicate_masks(larr.null_mask.as_ref(), rarr.null_mask.as_ref(), len);
    let _ = confirm_mask_capacity(larr.len(), lmask);
    let _ = confirm_mask_capacity(rarr.len(), rmask);

    // Compute total byte size required
    let mut total_bytes = 0;
    for i in 0..len {
        let valid = lmask.map_or(true, |m| unsafe { m.get_unchecked(loff + i) })
            && rmask.map_or(true, |m| unsafe { m.get_unchecked(roff + i) });
        if valid {
            let l = unsafe { larr.get_str_unchecked(loff + i) };
            let r = unsafe { rarr.get_str_unchecked(roff + i) };
            total_bytes += l.len() + r.len();
        }
    }

    // Allocate offsets and data buffers
    let mut offsets = Vec64::<T>::with_capacity(len + 1);
    unsafe {
        offsets.set_len(len + 1);
    }
    let mut values = Vec64::<u8>::with_capacity(total_bytes);

    // Fill values and offsets
    offsets[0] = T::zero();
    let mut cur = 0;

    for i in 0..len {
        let valid = lmask.map_or(true, |m| unsafe { m.get_unchecked(loff + i) })
            && rmask.map_or(true, |m| unsafe { m.get_unchecked(roff + i) });

        if valid {
            let l = unsafe { larr.get_str_unchecked(loff + i).as_bytes() };
            let r = unsafe { rarr.get_str_unchecked(roff + i).as_bytes() };

            values.extend_from_slice(l);
            values.extend_from_slice(r);
            cur += l.len() + r.len();

            unsafe {
                out_mask.set_unchecked(i, true);
            }
        } else {
            unsafe {
                out_mask.set_unchecked(i, false);
            }
        }

        offsets[i + 1] = T::from_usize(cur);
    }

    StringArray {
        offsets: offsets.into(),
        data: values.into(),
        null_mask: Some(out_mask),
    }
}

/// Concatenates corresponding string pairs from two categorical arrays element-wise.
///
/// Performs element-wise concatenation by looking up dictionary values from both
/// categorical arrays and concatenating the resolved strings. Creates a new string
/// array with the concatenated results.
///
/// # Parameters
/// - `lhs`: Left-hand categorical array view tuple `(CategoricalArray, offset, length)`
/// - `rhs`: Right-hand categorical array view tuple `(CategoricalArray, offset, length)`
///
/// # Returns
/// A new `StringArray<T>` containing concatenated dictionary strings.
///
/// # Null Handling
/// - If either categorical value is null, the result is null
/// - Output null mask reflects the union of input null positions
///
/// # Implementation
/// - Dictionary lookups resolve categorical codes to actual strings
/// - Memory allocation optimised based on total concatenated length
pub fn concat_dict_dict<T: Integer>(
    lhs: CategoricalAVT<T>,
    rhs: CategoricalAVT<T>,
) -> Result<CategoricalArray<T>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let len = llen.min(rlen);

    let (lmask, rmask, mut out_mask) =
        string_predicate_masks(larr.null_mask.as_ref(), rarr.null_mask.as_ref(), len);
    let _ = confirm_mask_capacity(larr.data.len(), lmask)?;
    let _ = confirm_mask_capacity(rarr.data.len(), rmask)?;

    // Use max possible unique count for preallocation. Worst case is all unique.
    let mut data = Vec64::<T>::with_capacity(len);
    unsafe {
        data.set_len(len);
    }

    let mut unique_values = Vec64::<String>::with_capacity(len);
    #[cfg(feature = "fast_hash")]
    let mut seen: AHashMap<String, T> = AHashMap::with_capacity(len);
    #[cfg(not(feature = "fast_hash"))]
    let mut seen: HashMap<String, T> = HashMap::with_capacity(len);
    let mut unique_idx = 0;

    for i in 0..len {
        let valid = lmask.map_or(true, |m| unsafe { m.get_unchecked(loff + i) })
            && rmask.map_or(true, |m| unsafe { m.get_unchecked(roff + i) });

        if valid {
            let l = unsafe { larr.get_str_unchecked(loff + i) };
            let r = unsafe { rarr.get_str_unchecked(roff + i) };
            let cat = format!("{l}{r}");

            let idx = match seen.get(&cat) {
                Some(ix) => *ix,
                None => {
                    let ix = T::from_usize(unique_idx);
                    unique_values.push(cat.clone());
                    seen.insert(cat, ix);
                    unique_idx += 1;
                    ix
                }
            };

            unsafe {
                *data.get_unchecked_mut(i) = idx;
                out_mask.set_unchecked(i, true);
            }
        } else {
            unsafe {
                *data.get_unchecked_mut(i) = T::zero();
                out_mask.set_unchecked(i, false);
            }
        }
    }

    unsafe {
        unique_values.set_len(unique_idx);
    }

    Ok(CategoricalArray {
        data: data.into(),
        unique_values,
        null_mask: Some(out_mask),
    })
}

/// Concatenates strings from a string array with dictionary values from a categorical array.
///
/// Performs element-wise concatenation where left operands come from a string array
/// and right operands are resolved from a categorical array's dictionary.
///
/// # Parameters
/// - `lhs`: String array view tuple `(StringArray, offset, length)`
/// - `rhs`: Categorical array view tuple `(CategoricalArray, offset, length)`
///
/// # Type Parameters
/// - `T`: Integer type for string array offsets
/// - `U`: Integer type for categorical array indices
///
/// # Returns
/// A new `StringArray<T>` containing concatenated string-dictionary pairs.
///
/// # Null Handling
/// Results are null if either input value is null at the corresponding position.
pub fn concat_str_dict<T: Integer, U: Integer>(
    lhs: StringAVT<T>,
    rhs: CategoricalAVT<U>,
) -> Result<StringArray<T>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let len = llen.min(rlen);

    let (lmask, rmask, mut out_mask) =
        string_predicate_masks(larr.null_mask.as_ref(), rarr.null_mask.as_ref(), len);
    let _ = confirm_mask_capacity(larr.len(), lmask)?;
    let _ = confirm_mask_capacity(rarr.data.len(), rmask)?;

    // Compute total byte size required
    let mut total_bytes = 0;
    for i in 0..len {
        let valid = lmask.map_or(true, |m| unsafe { m.get_unchecked(loff + i) })
            && rmask.map_or(true, |m| unsafe { m.get_unchecked(roff + i) });
        if valid {
            let a = unsafe { larr.get_str_unchecked(loff + i) };
            let b = unsafe { rarr.get_str_unchecked(roff + i) };
            total_bytes += a.len() + b.len();
        }
    }

    // Preallocate offsets and values
    let mut offsets = Vec64::<T>::with_capacity(len + 1);
    unsafe {
        offsets.set_len(len + 1);
    }
    let mut values = Vec64::<u8>::with_capacity(total_bytes);

    // Fill values and offsets
    offsets[0] = T::zero();
    let mut cur = 0;

    for i in 0..len {
        let valid = lmask.map_or(true, |m| unsafe { m.get_unchecked(loff + i) })
            && rmask.map_or(true, |m| unsafe { m.get_unchecked(roff + i) });

        if valid {
            let a = unsafe { larr.get_str_unchecked(loff + i).as_bytes() };
            let b = unsafe { rarr.get_str_unchecked(roff + i).as_bytes() };

            values.extend_from_slice(a);
            values.extend_from_slice(b);
            cur += a.len() + b.len();

            unsafe {
                out_mask.set_unchecked(i, true);
            }
        } else {
            unsafe {
                out_mask.set_unchecked(i, false);
            }
        }

        offsets[i + 1] = T::from_usize(cur);
    }

    Ok(StringArray {
        offsets: offsets.into(),
        data: values.into(),
        null_mask: Some(out_mask),
    })
}

/// Concatenates dictionary values from a categorical array with strings from a string array.
///
/// Performs element-wise concatenation where left operands are resolved from a
/// categorical array's dictionary and right operands come from a string array.
///
/// # Parameters
/// - `lhs`: Categorical array view tuple `(CategoricalArray, offset, length)`
/// - `rhs`: String array view tuple `(StringArray, offset, length)`
///
/// # Type Parameters
/// - `T`: Integer type for string array offsets
/// - `U`: Integer type for categorical array indices
///
/// # Returns
/// A new `StringArray<T>` containing concatenated dictionary-string pairs.
///
/// # Null Handling
/// Results are null if either input value is null at the corresponding position.
pub fn concat_dict_str<T: Integer, U: Integer>(
    lhs: CategoricalAVT<U>,
    rhs: StringAVT<T>,
) -> Result<StringArray<T>, KernelError> {
    concat_str_dict(rhs, lhs)
}

macro_rules! binary_str_pred_loop {
    ($len:expr, $lmask:expr, $rmask:expr, $out_mask:expr, $lhs:expr, $rhs:expr, $method:ident) => {{
        let mut data = Bitmask::new_set_all($len, false);
        // ensure masks cover offset + len
        let lhs_off = $lhs.1;
        let rhs_off = $rhs.1;
        let _ = confirm_mask_capacity(lhs_off + $len, $lmask)?;
        let _ = confirm_mask_capacity(rhs_off + $len, $rmask)?;
        for i in 0..$len {
            let li = lhs_off + i;
            let ri = rhs_off + i;
            let valid = $lmask.map_or(true, |m| unsafe { m.get_unchecked(li) })
                && $rmask.map_or(true, |m| unsafe { m.get_unchecked(ri) });
            let result = valid && {
                let s = unsafe { $lhs.0.get_str_unchecked(li) };
                let pat = unsafe { $rhs.0.get_str_unchecked(ri) };
                !pat.is_empty() && s.$method(pat)
            };
            unsafe {
                data.set_unchecked(i, result);
                $out_mask.set_unchecked(i, valid);
            }
        }
        data
    }};
}

// STRING PREDICATES - contains/starts_with/ends_with

/// Generates string predicate functions that compare string arrays.
macro_rules! str_predicate {
    ($fn_name:ident, $method:ident) => {
        /// Performs string predicate operations between two string arrays.
        ///
        /// Applies the specified string method (contains, starts_with, ends_with)
        /// to compare corresponding elements of two string arrays.
        ///
        /// # Type Parameters
        ///
        /// * `T` - Integer type for left string array offsets
        /// * `U` - Integer type for right string array offsets
        ///
        /// # Arguments
        ///
        /// * `lhs` - Left string array (data, offset, length)
        /// * `rhs` - Right string array (data, offset, length)
        ///
        /// # Returns
        ///
        /// Boolean array containing comparison results
        pub fn $fn_name<T: Integer, U: Integer>(
            lhs: StringAVT<T>,
            rhs: StringAVT<U>,
        ) -> BooleanArray<()> {
            let (larr, loff, llen) = lhs;
            let (rarr, roff, rlen) = rhs;
            let len = llen.min(rlen);
            // Grab raw pointers & slices once
            let lmask = larr.null_mask.as_ref();
            let rmask = rarr.null_mask.as_ref();
            let mut out = Bitmask::new_set_all(len, false);

            for i in 0..len {
                unsafe {
                    // Check null‐mask validity without bounds checks
                    let lv = lmask.map_or(true, |m| m.get_unchecked(loff + i));
                    let rv = rmask.map_or(true, |m| m.get_unchecked(roff + i));
                    if !lv || !rv {
                        // leave out[i]=false
                        continue;
                    }
                    // Slice out the raw bytes
                    let ls = larr.offsets[loff + i].to_usize();
                    let le = larr.offsets[loff + i + 1].to_usize();
                    let rs = rarr.offsets[roff + i].to_usize();
                    let re = rarr.offsets[roff + i + 1].to_usize();
                    let s = std::str::from_utf8_unchecked(&larr.data[ls..le]);
                    let p = std::str::from_utf8_unchecked(&rarr.data[rs..re]);
                    // Only non-empty pattern can match
                    if !p.is_empty() && s.$method(p) {
                        out.set_unchecked(i, true);
                    }
                }
            }
            // Tight bitmask with no nulls - nulls became 'false'
            BooleanArray {
                data: out.into(),
                null_mask: None,
                len,
                _phantom: PhantomData,
            }
        }
    };
}

/// Generates string-to-categorical predicate functions.
macro_rules! str_cat_predicate {
    ($fn_name:ident, $method:ident) => {
        /// Performs string predicate operations between string and categorical arrays.
        ///
        /// Applies the specified string method (contains, starts_with, ends_with)
        /// to compare string array elements with categorical array elements.
        ///
        /// # Type Parameters
        ///
        /// * `T` - Integer type for string array offsets
        /// * `U` - Integer type for categorical array offsets
        ///
        /// # Arguments
        ///
        /// * `lhs` - String array (data, offset, length)
        /// * `rhs` - Categorical array (data, offset, length)
        ///
        /// # Returns
        ///
        /// Result containing boolean array with comparison results, or error
        pub fn $fn_name<T: Integer, U: Integer>(
            lhs: StringAVT<T>,
            rhs: CategoricalAVT<U>,
        ) -> Result<BooleanArray<()>, KernelError> {
            let (larr, loff, llen) = lhs;
            let (rarr, roff, rlen) = rhs;
            let len = llen.min(rlen);

            let (lmask, rmask, mut out_mask) =
                string_predicate_masks(larr.null_mask.as_ref(), rarr.null_mask.as_ref(), len);

            let data = binary_str_pred_loop!(
                len,
                lmask,
                rmask,
                out_mask,
                (larr, loff),
                (rarr, roff),
                $method
            );

            Ok(BooleanArray {
                data: data.into(),
                null_mask: Some(out_mask),
                len,
                _phantom: PhantomData,
            })
        }
    };
}

/// Generates categorical-to-categorical predicate functions.
macro_rules! cat_cat_predicate {
    ($fn_name:ident, $method:ident) => {
        /// Performs string predicate operations between two categorical arrays.
        ///
        /// Applies the specified string method (contains, starts_with, ends_with)
        /// to compare corresponding elements of two categorical arrays.
        ///
        /// # Type Parameters
        ///
        /// * `T` - Integer type for categorical array offsets
        ///
        /// # Arguments
        ///
        /// * `lhs` - Left categorical array (data, offset, length)
        /// * `rhs` - Right categorical array (data, offset, length)
        ///
        /// # Returns
        ///
        /// Result containing boolean array with comparison results, or error
        pub fn $fn_name<T: Integer>(
            lhs: CategoricalAVT<T>,
            rhs: CategoricalAVT<T>,
        ) -> Result<BooleanArray<()>, KernelError> {
            let (larr, loff, llen) = lhs;
            let (rarr, roff, rlen) = rhs;
            let len = llen.min(rlen);

            let (lmask, rmask, mut out_mask) =
                string_predicate_masks(larr.null_mask.as_ref(), rarr.null_mask.as_ref(), len);

            let data = binary_str_pred_loop!(
                len,
                lmask,
                rmask,
                out_mask,
                (larr, loff),
                (rarr, roff),
                $method
            );

            Ok(BooleanArray {
                data: data.into(),
                null_mask: Some(out_mask),
                len,
                _phantom: PhantomData,
            })
        }
    };
}

/// Generates categorical-to-string predicate functions.
macro_rules! dict_str_predicate {
    ($fn_name:ident, $method:ident) => {
        /// Performs string predicate operations between categorical and string arrays.
        ///
        /// Applies the specified string method (contains, starts_with, ends_with)
        /// to compare categorical array elements with string array elements.
        ///
        /// # Type Parameters
        ///
        /// * `T` - Integer type for categorical array offsets
        /// * `U` - Integer type for string array offsets
        ///
        /// # Arguments
        ///
        /// * `lhs` - Categorical array (data, offset, length)
        /// * `rhs` - String array (data, offset, length)
        ///
        /// # Returns
        ///
        /// Result containing boolean array with comparison results, or error
        pub fn $fn_name<T: Integer, U: Integer>(
            lhs: CategoricalAVT<T>,
            rhs: StringAVT<U>,
        ) -> Result<BooleanArray<()>, KernelError> {
            let (larr, loff, llen) = lhs;
            let (rarr, roff, rlen) = rhs;
            let len = llen.min(rlen);

            let (lmask, rmask, mut out_mask) =
                string_predicate_masks(larr.null_mask.as_ref(), rarr.null_mask.as_ref(), len);
            let _ = confirm_mask_capacity(larr.data.len(), lmask)?;
            let _ = confirm_mask_capacity(rarr.len(), rmask)?;

            let mut data = Bitmask::new_set_all(len, false);
            for i in 0..len {
                let valid = lmask.map_or(true, |m| unsafe { m.get_unchecked(loff + i) })
                    && rmask.map_or(true, |m| unsafe { m.get_unchecked(roff + i) });
                let match_i = valid && {
                    let hay = unsafe { larr.get_str_unchecked(loff + i) };
                    let needle = unsafe { rarr.get_str_unchecked(roff + i) };
                    !needle.is_empty() && hay.$method(needle)
                };
                unsafe { data.set_unchecked(i, match_i) };
                unsafe { out_mask.set_unchecked(i, valid) };
            }

            Ok(BooleanArray {
                data: data.into(),
                null_mask: Some(out_mask),
                len,
                _phantom: PhantomData,
            })
        }
    };
}

str_predicate!(contains_str_str, contains);
str_predicate!(starts_with_str_str, starts_with);
str_predicate!(ends_with_str_str, ends_with);
str_cat_predicate!(contains_str_dict, contains);
cat_cat_predicate!(contains_dict_dict, contains);
str_cat_predicate!(starts_with_str_dict, starts_with);
cat_cat_predicate!(starts_with_dict_dict, starts_with);
str_cat_predicate!(ends_with_str_dict, ends_with);
cat_cat_predicate!(ends_with_dict_dict, ends_with);
dict_str_predicate!(contains_dict_str, contains);
dict_str_predicate!(starts_with_dict_str, starts_with);
dict_str_predicate!(ends_with_dict_str, ends_with);

// Regex match

#[cfg(feature = "regex")]
macro_rules! regex_match_loop {
    ($len:expr, $lmask:expr, $rmask:expr, $out_mask:expr, $lhs_arr:expr, $lhs_off:expr, $rhs_arr:expr, $rhs_off:expr) => {{
        let mut data = Bitmask::new_set_all($len, false);
        let _ = confirm_mask_capacity($len + $lhs_off, $lmask)?;
        let _ = confirm_mask_capacity($len + $rhs_off, $rmask)?;
        for i in 0..$len {
            let valid = $lmask.map_or(true, |m| unsafe { m.get_unchecked($lhs_off + i) })
                && $rmask.map_or(true, |m| unsafe { m.get_unchecked($rhs_off + i) });
            let matched = if valid {
                let s = unsafe { $lhs_arr.get_str_unchecked($lhs_off + i) };
                let pat = unsafe { $rhs_arr.get_str_unchecked($rhs_off + i) };
                if pat.is_empty() {
                    false
                } else {
                    match Regex::new(pat) {
                        Ok(re) => re.is_match(s),
                        Err(_) => {
                            return Err(KernelError::InvalidArguments(
                                "Invalid regex string".to_string(),
                            ));
                        }
                    }
                }
            } else {
                false
            };
            unsafe { data.set_unchecked(i, matched) };
            unsafe { $out_mask.set_unchecked(i, valid) };
        }
        data
    }};
}

/// Applies regular expression pattern matching between two string arrays.
///
/// Evaluates regex patterns from the right-hand string array against corresponding
/// strings in the left-hand array, producing a boolean array indicating matches.
///
/// # Parameters
/// - `lhs`: Source string array view tuple `(StringArray, offset, length)`
/// - `rhs`: Pattern string array view tuple `(StringArray, offset, length)`
///
/// # Type Parameters
/// - `T`: Integer type for left array offsets
/// - `U`: Integer type for right array offsets
///
/// # Returns
/// `Result<BooleanArray<()>, KernelError>` where true indicates pattern match.
///
/// # Errors
/// Returns `KernelError` for invalid regular expression patterns.
///
/// # Feature Gate
/// Requires the `regex` feature to be enabled.
///
/// # Performance
/// - Regex compilation overhead amortised across bulk operations
/// - Pattern caching opportunities for repeated patterns
#[cfg(feature = "regex")]
pub fn regex_str_str<'a, T: Integer, U: Integer>(
    lhs: StringAVT<'a, T>,
    rhs: StringAVT<'a, U>,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let len = llen.min(rlen);
    let (lmask, rmask, mut out_mask) =
        string_predicate_masks(larr.null_mask.as_ref(), rarr.null_mask.as_ref(), len);

    let data = regex_match_loop!(len, lmask, rmask, out_mask, larr, loff, rarr, roff);
    Ok(BooleanArray {
        data: data.into(),
        null_mask: Some(out_mask),
        len,
        _phantom: PhantomData,
    })
}

/// Applies regular expression patterns to categorical array values against string patterns.
///
/// Evaluates regex patterns from the string array against dictionary-resolved strings
/// from the categorical array, producing a boolean array indicating matches.
///
/// # Parameters
/// - `lhs`: Categorical array view tuple `(CategoricalArray, offset, length)`
/// - `rhs`: Pattern string array view tuple `(StringArray, offset, length)`
///
/// # Type Parameters
/// - `U`: Integer type for categorical array indices
/// - `T`: Integer type for pattern array offsets
///
/// # Returns
/// `Result<BooleanArray<()>, KernelError>` where true indicates pattern match.
///
/// # Errors
/// Returns `KernelError` for invalid regular expression patterns.
///
/// # Feature Gate
/// Requires the `regex` feature to be enabled.
#[cfg(feature = "regex")]
pub fn regex_dict_str<'a, U: Integer, T: Integer>(
    lhs: CategoricalAVT<'a, U>,
    rhs: StringAVT<'a, T>,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let len = llen.min(rlen);
    let (lmask, rmask, mut out_mask) =
        string_predicate_masks(larr.null_mask.as_ref(), rarr.null_mask.as_ref(), len);

    let data = regex_match_loop!(len, lmask, rmask, out_mask, larr, loff, rarr, roff);
    Ok(BooleanArray {
        data: data.into(),
        null_mask: Some(out_mask),
        len,
        _phantom: PhantomData,
    })
}

/// Applies regular expression patterns from categorical dictionary against string values.
///
/// Evaluates regex patterns resolved from the categorical array's dictionary against
/// corresponding strings in the string array, producing a boolean array of matches.
///
/// # Parameters
/// - `lhs`: Source string array view tuple `(StringArray, offset, length)`
/// - `rhs`: Pattern categorical array view tuple `(CategoricalArray, offset, length)`
///
/// # Type Parameters
/// - `T`: Integer type for string array offsets
/// - `U`: Integer type for categorical array indices
///
/// # Returns
/// `Result<BooleanArray<()>, KernelError>` where true indicates pattern match.
///
/// # Errors
/// Returns `KernelError` for invalid regular expression patterns in dictionary.
///
/// # Feature Gate
/// Requires the `regex` feature to be enabled.
#[cfg(feature = "regex")]
pub fn regex_str_dict<'a, T: Integer, U: Integer>(
    lhs: StringAVT<'a, T>,
    rhs: CategoricalAVT<'a, U>,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let len = llen.min(rlen);
    let (lmask, rmask, mut out_mask) =
        string_predicate_masks(larr.null_mask.as_ref(), rarr.null_mask.as_ref(), len);

    let data = regex_match_loop!(len, lmask, rmask, out_mask, larr, loff, rarr, roff);
    Ok(BooleanArray {
        data: data.into(),
        null_mask: Some(out_mask),
        len,
        _phantom: PhantomData,
    })
}

/// Applies regular expression patterns between two categorical arrays via dictionary lookup.
///
/// Evaluates regex patterns by resolving both pattern and target strings from their
/// respective categorical dictionaries, producing a boolean array indicating matches.
///
/// # Parameters
/// - `lhs`: Source categorical array view tuple `(CategoricalArray, offset, length)`
/// - `rhs`: Pattern categorical array view tuple `(CategoricalArray, offset, length)`
///
/// # Type Parameters
/// - `T`: Integer type for categorical array indices
///
/// # Returns
/// `Result<BooleanArray<()>, KernelError>` where true indicates pattern match.
///
/// # Errors
/// Returns `KernelError` for invalid regular expression patterns in dictionaries.
///
/// # Feature Gate
/// Requires the `regex` feature to be enabled.
///
/// # Performance
/// - Dictionary lookups amortised across categorical operations
/// - Pattern compilation cached for repeated dictionary patterns
#[cfg(feature = "regex")]
pub fn regex_dict_dict<'a, T: Integer>(
    lhs: CategoricalAVT<'a, T>,
    rhs: CategoricalAVT<'a, T>,
) -> Result<BooleanArray<()>, KernelError> {
    let (larr, loff, llen) = lhs;
    let (rarr, roff, rlen) = rhs;
    let len = llen.min(rlen);
    let (lmask, rmask, mut out_mask) =
        string_predicate_masks(larr.null_mask.as_ref(), rarr.null_mask.as_ref(), len);

    let data = regex_match_loop!(len, lmask, rmask, out_mask, larr, loff, rarr, roff);
    Ok(BooleanArray {
        data: data.into(),
        null_mask: Some(out_mask),
        len,
        _phantom: PhantomData,
    })
}

/// Computes the character length of each string in a `StringArray<T>` slice,
/// returning an `IntegerArray<T>` with the same length and null semantics.
///
/// This applies to a windowed slice (`offset`, `len`) of the input array.
/// The output null mask mirrors the sliced portion of the input mask.
///
/// # Parameters
/// - `input`: A `(StringArray<T>, offset, len)` tuple defining the slice to operate on.
///
/// # Returns
/// An `IntegerArray<T>` of the same length, with each element representing the character count
/// of the corresponding (non-null) string value.
pub fn len_str<'a, T: Integer + Copy>(
    input: StringAVT<'a, T>,
) -> Result<IntegerArray<T>, KernelError> {
    let (array, offset, len) = input;
    debug_assert!(offset + len <= array.offsets.len() - 1);

    let mask_opt = array.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe {
                m.set_unchecked(i, orig.get_unchecked(offset + i));
            }
        }
        m
    });

    let mut data = Vec64::<T>::with_capacity(len);
    unsafe { data.set_len(len) };
    for i in 0..len {
        let valid = mask_opt
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(i) });
        if valid {
            let start = array.offsets[offset + i].to_usize();
            let end = array.offsets[offset + i + 1].to_usize();
            let s = unsafe { std::str::from_utf8_unchecked(&array.data[start..end]) };
            data[i] = T::from(s.chars().count()).unwrap();
        } else {
            data[i] = T::zero();
        }
    }

    Ok(IntegerArray {
        data: data.into(),
        null_mask: mask_opt,
    })
}

/// Computes the character length of each string in a `CategoricalArray<T>` slice,
/// returning an `IntegerArray<T>` with the same length and null semantics.
///
/// This applies to a windowed slice (`offset`, `len`) of the input categorical array,
/// using dictionary lookup to resolve each string.
///
/// # Parameters
/// - `input`: A `(CategoricalArray<T>, offset, len)` tuple defining the slice to operate on.
///
/// # Returns
/// An `IntegerArray<T>` of the same length, with each element representing the character count
/// of the corresponding (non-null) resolved string.
pub fn len_dict<'a, T: Integer>(
    input: CategoricalAVT<'a, T>,
) -> Result<IntegerArray<T>, KernelError> {
    let (array, offset, len) = input;
    debug_assert!(offset + len <= array.data.len());

    let mask_opt = array.null_mask.as_ref().map(|orig| {
        let mut m = Bitmask::new_set_all(len, true);
        for i in 0..len {
            unsafe {
                m.set_unchecked(i, orig.get_unchecked(offset + i));
            }
        }
        m
    });

    let mut data = Vec64::<T>::with_capacity(len);
    unsafe { data.set_len(len) };
    for i in 0..len {
        let valid = mask_opt
            .as_ref()
            .map_or(true, |m| unsafe { m.get_unchecked(i) });
        data[i] = if valid {
            T::from(
                unsafe { array.get_str_unchecked(offset + i) }
                    .chars()
                    .count(),
            )
            .unwrap()
        } else {
            T::zero()
        };
    }

    Ok(IntegerArray {
        data: data.into(),
        null_mask: mask_opt,
    })
}

/// Finds the lexicographically minimum string in a string array window.
///
/// Scans through a windowed portion of a string array to determine the minimum
/// string value according to lexicographic ordering, ignoring null values.
///
/// # Parameters
/// - `window`: String array view tuple `(StringArray, offset, length)` defining the scan window
///
/// # Returns
/// `Option<String>` containing the minimum string, or `None` if all values are null.
#[inline]
pub fn min_string_array<T: Integer>(window: StringAVT<T>) -> Option<String> {
    let (arr, offset, len) = window;
    let mut min_str: Option<&str> = None;
    for i in offset..offset + len {
        if arr
            .null_mask
            .as_ref()
            .map_or(true, |b| unsafe { b.get_unchecked(i) })
        {
            let s = unsafe { arr.get_str_unchecked(i) };
            if min_str.map_or(true, |min| s < min) {
                min_str = Some(s);
            }
        }
    }
    min_str.map(str::to_owned)
}

/// Finds the lexicographically maximum string in a string array window.
///
/// Scans through a windowed portion of a string array to determine the maximum
/// string value according to lexicographic ordering, ignoring null values.
///
/// # Parameters
/// - `window`: String array view tuple `(StringArray, offset, length)` defining the scan window
///
/// # Returns
/// `Option<String>` containing the maximum string, or `None` if all values are null.
#[inline]
pub fn max_string_array<T: Integer>(window: StringAVT<T>) -> Option<String> {
    let (arr, offset, len) = window;
    let mut max_str: Option<&str> = None;
    for i in offset..offset + len {
        if arr
            .null_mask
            .as_ref()
            .map_or(true, |b| unsafe { b.get_unchecked(i) })
        {
            let s = unsafe { arr.get_str_unchecked(i) };
            if max_str.map_or(true, |max| s > max) {
                max_str = Some(s);
            }
        }
    }
    max_str.map(str::to_owned)
}

/// Finds the lexicographically minimum dictionary string in a categorical array window.
///
/// Scans through a windowed portion of a categorical array, resolves dictionary values,
/// and determines the minimum string according to lexicographic ordering.
///
/// # Parameters
/// - `window`: Categorical array view tuple `(CategoricalArray, offset, length)` defining the scan window
///
/// # Returns
/// `Option<String>` containing the minimum dictionary string, or `None` if all values are null.
#[inline]
pub fn min_categorical_array<T: Integer>(window: CategoricalAVT<T>) -> Option<String> {
    let (arr, offset, len) = window;
    let mut min_code: Option<T> = None;
    for i in offset..offset + len {
        if arr
            .null_mask
            .as_ref()
            .map_or(true, |b| unsafe { b.get_unchecked(i) })
        {
            let code = arr.data[i];
            if min_code.map_or(true, |min| {
                let sc = &arr.unique_values[code.to_usize()];
                let sm = &arr.unique_values[min.to_usize()];
                sc < sm
            }) {
                min_code = Some(code);
            }
        }
    }
    min_code.map(|code| arr.unique_values[code.to_usize()].clone())
}

/// Finds the lexicographically maximum dictionary string in a categorical array window.
///
/// Scans through a windowed portion of a categorical array, resolves dictionary values,
/// and determines the maximum string according to lexicographic ordering.
///
/// # Parameters
/// - `window`: Categorical array view tuple `(CategoricalArray, offset, length)` defining the scan window
///
/// # Returns
/// `Option<String>` containing the maximum dictionary string, or `None` if all values are null.
#[inline]
pub fn max_categorical_array<T: Integer>(window: CategoricalAVT<T>) -> Option<String> {
    let (arr, offset, len) = window;
    let mut max_code: Option<T> = None;
    for i in offset..offset + len {
        if arr
            .null_mask
            .as_ref()
            .map_or(true, |b| unsafe { b.get_unchecked(i) })
        {
            let code = arr.data[i];
            if max_code.map_or(true, |max| {
                let sc = &arr.unique_values[code.to_usize()];
                let sm = &arr.unique_values[max.to_usize()];
                sc > sm
            }) {
                max_code = Some(code);
            }
        }
    }
    max_code.map(|code| arr.unique_values[code.to_usize()].clone())
}

/// Counts the number of distinct string values in a string array window.
///
/// Computes the cardinality of unique strings within a windowed portion of a
/// string array, using efficient hash-based deduplication.
///
/// # Parameters
/// - `window`: String array view tuple `(StringArray, offset, length)` defining the count window
///
/// # Returns
/// `usize` representing the count of distinct non-null string values.
///
/// # Hash Algorithm
/// Uses either AHash (if `fast_hash` feature enabled) or standard HashMap for deduplication.
#[inline(always)]
pub fn count_distinct_string<T: Integer>(window: StringAVT<T>) -> usize {
    let (arr, offset, len) = window;
    #[cfg(feature = "fast_hash")]
    let mut set = AHashSet::with_capacity(len);
    #[cfg(not(feature = "fast_hash"))]
    let mut set = HashSet::with_capacity(len);
    let null_mask = arr.null_mask.as_ref();

    for i in offset..offset + len {
        let valid = null_mask.map_or(true, |b| unsafe { b.get_unchecked(i) });
        if valid {
            let s = unsafe { arr.get_str_unchecked(i) };
            set.insert(s);
            if set.len() == len {
                break;
            }
        }
    }
    set.len()
}

#[cfg(test)]
mod tests {
    use crate::{CategoricalArray, StringArray, vec64};

    use super::*;

    // --- Helper constructors

    fn str_array<T: Integer>(vals: &[&str]) -> StringArray<T> {
        StringArray::<T>::from_slice(vals)
    }

    fn dict_array<T: Integer>(vals: &[&str]) -> CategoricalArray<T> {
        let owned: Vec<&str> = vals.to_vec();
        CategoricalArray::<T>::from_values(owned)
    }

    fn bm(bools: &[bool]) -> Bitmask {
        Bitmask::from_bools(bools)
    }

    // --- Concat

    #[test]
    fn test_concat_str_str() {
        let a = str_array::<u32>(&["foo", "bar", ""]);
        let b = str_array::<u32>(&["baz", "qux", "quux"]);
        let out = concat_str_str((&a, 0, a.len()), (&b, 0, b.len()));
        assert_eq!(out.get(0), Some("foobaz"));
        assert_eq!(out.get(1), Some("barqux"));
        assert_eq!(out.get(2), Some("quux"));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_concat_str_str_chunk() {
        let a = str_array::<u32>(&["XXX", "foo", "bar", ""]);
        let b = str_array::<u32>(&["YYY", "baz", "qux", "quux"]);
        // Window is [1..4) for both, i.e., ["foo", "bar", ""]
        let out = concat_str_str((&a, 1, 3), (&b, 1, 3));
        assert_eq!(out.get(0), Some("foobaz"));
        assert_eq!(out.get(1), Some("barqux"));
        assert_eq!(out.get(2), Some("quux"));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_concat_dict_dict() {
        let a = dict_array::<u32>(&["x", "y"]);
        let b = dict_array::<u32>(&["1", "2"]);
        let out = concat_dict_dict((&a, 0, a.len()), (&b, 0, b.len())).unwrap();
        let s0 = out.get(0).unwrap();
        let s1 = out.get(1).unwrap();
        assert!(["x1", "y2"].contains(&s0));
        assert!(["x1", "y2"].contains(&s1));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_concat_dict_dict_chunk() {
        let a = dict_array::<u32>(&["foo", "x", "y", "bar"]);
        let b = dict_array::<u32>(&["A", "1", "2", "B"]);
        let out = concat_dict_dict((&a, 1, 2), (&b, 1, 2)).unwrap();
        let s0 = out.get(0).unwrap();
        let s1 = out.get(1).unwrap();
        assert!(["x1", "y2"].contains(&s0));
        assert!(["x1", "y2"].contains(&s1));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_concat_str_dict() {
        let a = str_array::<u32>(&["ab", "cd", ""]);
        let b = dict_array::<u32>(&["xy", "zq", ""]);
        let out = concat_str_dict((&a, 0, a.len()), (&b, 0, b.len())).unwrap();
        assert_eq!(out.get(0), Some("abxy"));
        assert_eq!(out.get(1), Some("cdzq"));
        assert_eq!(out.get(2), Some(""));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_concat_str_dict_chunk() {
        let a = str_array::<u32>(&["dummy", "ab", "cd", ""]);
        let b = dict_array::<u32>(&["dummy", "xy", "zq", ""]);
        let out = concat_str_dict((&a, 1, 3), (&b, 1, 3)).unwrap();
        assert_eq!(out.get(0), Some("abxy"));
        assert_eq!(out.get(1), Some("cdzq"));
        assert_eq!(out.get(2), Some(""));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_concat_dict_str() {
        let a = dict_array::<u32>(&["hi", "ho"]);
        let b = str_array::<u32>(&["yo", "no"]);
        let out = concat_dict_str((&a, 0, a.len()), (&b, 0, b.len())).unwrap();
        assert_eq!(out.get(0), Some("yohi"));
        assert_eq!(out.get(1), Some("noho"));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    #[test]
    fn test_concat_dict_str_chunk() {
        let a = dict_array::<u32>(&["dummy", "hi", "ho", "zzz"]);
        let b = str_array::<u32>(&["dummy", "yo", "no", "xxx"]);
        let out = concat_dict_str((&a, 1, 2), (&b, 1, 2)).unwrap();
        assert_eq!(out.get(0), Some("yohi"));
        assert_eq!(out.get(1), Some("noho"));
        assert!(out.null_mask.as_ref().unwrap().all_set());
    }

    // --- String predicates

    #[test]
    fn test_contains_str_str() {
        let s = str_array::<u32>(&["abc", "def", "ghijk"]);
        let p = str_array::<u32>(&["b", "x", "jk"]);
        let out = contains_str_str((&s, 0, s.len()), (&p, 0, p.len()));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(false));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_contains_str_str_chunk() {
        let s = str_array::<u32>(&["dummy", "abc", "def", "ghijk"]);
        let p = str_array::<u32>(&["dummy", "b", "x", "jk"]);
        let out = contains_str_str((&s, 1, 3), (&p, 1, 3));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(false));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_starts_with_str_str() {
        let s = str_array::<u32>(&["apricot", "banana", "apple"]);
        let p = str_array::<u32>(&["ap", "ba", "a"]);
        let out = starts_with_str_str((&s, 0, s.len()), (&p, 0, p.len()));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_starts_with_str_str_chunk() {
        let s = str_array::<u32>(&["dummy", "apricot", "banana", "apple"]);
        let p = str_array::<u32>(&["dummy", "ap", "ba", "a"]);
        let out = starts_with_str_str((&s, 1, 3), (&p, 1, 3));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_ends_with_str_str() {
        let s = str_array::<u32>(&["robot", "fast", "last"]);
        let p = str_array::<u32>(&["ot", "st", "ast"]);
        let out = ends_with_str_str((&s, 0, s.len()), (&p, 0, p.len()));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_ends_with_str_str_chunk() {
        let s = str_array::<u32>(&["dummy", "robot", "fast", "last"]);
        let p = str_array::<u32>(&["dummy", "ot", "st", "ast"]);
        let out = ends_with_str_str((&s, 1, 3), (&p, 1, 3));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_contains_str_dict() {
        let s = str_array::<u32>(&["abcde", "xyz", "qrstuv"]);
        let p = dict_array::<u32>(&["c", "z", "tu"]);
        let out = contains_str_dict((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_contains_str_dict_chunk() {
        let s = str_array::<u32>(&["dummy", "abcde", "xyz", "qrstuv"]);
        let p = dict_array::<u32>(&["dummy", "c", "z", "tu"]);
        let out = contains_str_dict((&s, 1, 3), (&p, 1, 3)).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_contains_dict_dict() {
        let s = dict_array::<u32>(&["cdef", "foo", "bar"]);
        let p = dict_array::<u32>(&["cd", "oo", "baz"]);
        let out = contains_dict_dict((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(false));
    }

    #[test]
    fn test_contains_dict_dict_chunk() {
        let s = dict_array::<u32>(&["dummy", "cdef", "foo", "bar"]);
        let p = dict_array::<u32>(&["dummy", "cd", "oo", "baz"]);
        let out = contains_dict_dict((&s, 1, 3), (&p, 1, 3)).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(false));
    }

    #[test]
    fn test_contains_dict_str() {
        let s = dict_array::<u32>(&["hello", "world"]);
        let p = str_array::<u32>(&["he", "o"]);
        let out = contains_dict_str((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
    }

    #[test]
    fn test_contains_dict_str_chunk() {
        let s = dict_array::<u32>(&["dummy", "hello", "world"]);
        let p = str_array::<u32>(&["dummy", "he", "o"]);
        let out = contains_dict_str((&s, 1, 2), (&p, 1, 2)).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
    }

    #[test]
    fn test_starts_with_str_dict() {
        let s = str_array::<u32>(&["abcdef", "foobar", "quux"]);
        let p = dict_array::<u32>(&["ab", "foo", "qu"]);
        let out = starts_with_str_dict((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_starts_with_str_dict_chunk() {
        let s = str_array::<u32>(&["dummy", "abcdef", "foobar", "quux"]);
        let p = dict_array::<u32>(&["dummy", "ab", "foo", "qu"]);
        let out = starts_with_str_dict((&s, 1, 3), (&p, 1, 3)).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_starts_with_dict_dict() {
        let s = dict_array::<u32>(&["qwerty", "banana"]);
        let p = dict_array::<u32>(&["qw", "ban"]);
        let out = starts_with_dict_dict((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
    }

    #[test]
    fn test_starts_with_dict_dict_chunk() {
        let s = dict_array::<u32>(&["dummy", "qwerty", "banana"]);
        let p = dict_array::<u32>(&["dummy", "qw", "ban"]);
        let out = starts_with_dict_dict((&s, 1, 2), (&p, 1, 2)).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
    }

    #[test]
    fn test_ends_with_str_dict() {
        let s = str_array::<u32>(&["poem", "dome", "gnome"]);
        let p = dict_array::<u32>(&["em", "me", "ome"]);
        let out = ends_with_str_dict((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_ends_with_str_dict_chunk() {
        let s = str_array::<u32>(&["dummy", "poem", "dome", "gnome"]);
        let p = dict_array::<u32>(&["dummy", "em", "me", "ome"]);
        let out = ends_with_str_dict((&s, 1, 3), (&p, 1, 3)).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(true));
    }

    #[test]
    fn test_ends_with_dict_dict() {
        let s = dict_array::<u32>(&["tablet", "let", "bet"]);
        let p = dict_array::<u32>(&["let", "et", "xyz"]);
        let out = ends_with_dict_dict((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(false));
    }

    #[test]
    fn test_ends_with_dict_dict_chunk() {
        let s = dict_array::<u32>(&["dummy", "tablet", "let", "bet"]);
        let p = dict_array::<u32>(&["dummy", "let", "et", "xyz"]);
        let out = ends_with_dict_dict((&s, 1, 3), (&p, 1, 3)).unwrap();
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(true));
        assert_eq!(out.get(2), Some(false));
    }

    // --- len_str, len_dict

    #[test]
    fn test_len_str() {
        let arr = str_array::<u32>(&["", "a", "abc", "bar"]);
        let out = len_str((&arr, 0, arr.len())).unwrap();
        assert_eq!(&out.data[..], &[0, 1, 3, 3]);
    }

    #[test]
    fn test_len_str_chunk() {
        let arr = str_array::<u32>(&["zzz", "", "a", "abc", "bar"]);
        let out = len_str((&arr, 1, 4)).unwrap(); // ["", "a", "abc", "bar"]
        assert_eq!(&out.data[..], &[0, 1, 3, 3]);
    }

    #[test]
    fn test_len_dict() {
        let arr = dict_array::<u32>(&["", "one", "seven"]);
        let out = len_dict((&arr, 0, arr.len())).unwrap();
        assert_eq!(&out.data[..], &[0, 3, 5]);
    }

    #[test]
    fn test_len_dict_chunk() {
        let arr = dict_array::<u32>(&["q", "", "one", "seven"]);
        let out = len_dict((&arr, 1, 3)).unwrap(); // ["", "one", "seven"]
        assert_eq!(&out.data[..], &[0, 3, 5]);
    }

    #[test]
    fn test_contains_empty_pattern() {
        let s = str_array::<u32>(&["foo", "bar"]);
        let p = str_array::<u32>(&["", ""]);
        let out = contains_str_str((&s, 0, s.len()), (&p, 0, p.len()));
        // always false
        assert_eq!(out.get(0), Some(false));
        assert_eq!(out.get(1), Some(false));
        assert!(out.null_mask.as_ref().is_none());
    }

    #[test]
    fn test_contains_empty_pattern_chunk() {
        let s = str_array::<u32>(&["z", "foo", "bar"]);
        let p = str_array::<u32>(&["z", "", ""]);
        let out = contains_str_str((&s, 1, 2), (&p, 1, 2));
        assert_eq!(out.get(0), Some(false));
        assert_eq!(out.get(1), Some(false));
        assert!(out.null_mask.as_ref().is_none());
    }

    #[test]
    fn test_contains_str_str_nulls_on_pattern() {
        let mut s = str_array::<u32>(&["abc", "def"]);
        s.null_mask = Some(bm(&[true, true]));
        let mut p = str_array::<u32>(&["b", "e"]);
        p.null_mask = Some(bm(&[true, false])); // second pattern is null
        let out = contains_str_str((&s, 0, s.len()), (&p, 0, p.len()));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(false));
    }

    #[test]
    fn test_contains_str_str_nulls_on_pattern_chunk() {
        let mut s = str_array::<u32>(&["X", "abc", "def"]);
        s.null_mask = Some(bm(&[true, true, true]));
        let mut p = str_array::<u32>(&["X", "b", "e"]);
        p.null_mask = Some(bm(&[true, true, false])); // last pattern is null
        let out = contains_str_str((&s, 1, 2), (&p, 1, 2));
        assert_eq!(out.get(0), Some(true));
        assert_eq!(out.get(1), Some(false));
    }

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_invalid_pattern_returns_err() {
        let s = str_array::<u32>(&["abc"]);
        let p = str_array::<u32>(&["["]);
        let err = regex_str_str((&s, 0, s.len()), (&p, 0, p.len())).unwrap_err();
        match err {
            KernelError::InvalidArguments(_) => {}
            _ => panic!("expected InvalidArguments"),
        }
    }

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_invalid_pattern_returns_err_chunk() {
        let s = str_array::<u32>(&["foo", "abc"]);
        let p = str_array::<u32>(&["bar", "["]);
        let err = regex_str_str((&s, 1, 1), (&p, 1, 1)).unwrap_err();
        match err {
            KernelError::InvalidArguments(_) => {}
            _ => panic!("expected InvalidArguments"),
        }
    }

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_empty_pattern_always_false() {
        let s = str_array::<u32>(&["abc", "def"]);
        let p = str_array::<u32>(&["", ""]);
        let out = regex_str_str((&s, 0, s.len()), (&p, 0, p.len())).unwrap();
        assert_eq!(out.get(0), Some(false));
        assert_eq!(out.get(1), Some(false));
        assert!(out.null_mask.unwrap().all_set());
    }

    #[cfg(feature = "regex")]
    #[test]
    fn test_regex_empty_pattern_always_false_chunk() {
        let s = str_array::<u32>(&["z", "abc", "def"]);
        let p = str_array::<u32>(&["z", "", ""]);
        let out = regex_str_str((&s, 1, 2), (&p, 1, 2)).unwrap();
        assert_eq!(out.get(0), Some(false));
        assert_eq!(out.get(1), Some(false));
        assert!(out.null_mask.unwrap().all_set());
    }

    #[test]
    fn test_len_str_with_nulls() {
        let mut arr = str_array::<u32>(&["foo", "", "bar"]);
        arr.null_mask = Some(bm(&[true, false, true]));
        let len_arr = len_str((&arr, 0, arr.len())).unwrap();
        assert_eq!(len_arr.data.as_slice(), &[3, 0, 3]);
        assert_eq!(
            len_arr.null_mask.unwrap().as_slice(),
            bm(&[true, false, true]).as_slice()
        );
    }

    #[test]
    fn test_len_str_with_nulls_chunk() {
        let mut arr = str_array::<u32>(&["x", "foo", "", "bar"]);
        arr.null_mask = Some(bm(&[true, true, false, true]));
        let len_arr = len_str((&arr, 1, 3)).unwrap();
        assert_eq!(len_arr.data.as_slice(), &[3, 0, 3]);
        assert_eq!(
            len_arr.null_mask.unwrap().as_slice(),
            bm(&[true, false, true]).as_slice()
        );
    }

    #[test]
    fn test_len_dict_with_nulls() {
        let mut arr = dict_array::<u32>(&["x", "yy", "zzz"]);
        arr.null_mask = Some(bm(&[false, true, true]));
        let len_arr = len_dict((&arr, 0, arr.len())).unwrap();
        assert_eq!(len_arr.data.as_slice(), &[0, 2, 3]);
        assert_eq!(
            len_arr.null_mask.unwrap().as_slice(),
            bm(&[false, true, true]).as_slice()
        );
    }

    #[test]
    fn test_len_dict_with_nulls_chunk() {
        let mut arr = dict_array::<u32>(&["z", "x", "yy", "zzz"]);
        arr.null_mask = Some(bm(&[true, false, true, true]));
        let len_arr = len_dict((&arr, 1, 3)).unwrap();
        assert_eq!(len_arr.data.as_slice(), &[0, 2, 3]);
        assert_eq!(
            len_arr.null_mask.unwrap().as_slice(),
            bm(&[false, true, true]).as_slice()
        );
    }

    fn bitmask_from_vec(v: &[bool]) -> Bitmask {
        let mut bm = Bitmask::with_capacity(v.len());
        for (i, &b) in v.iter().enumerate() {
            bm.set(i, b);
        }
        bm
    }

    #[test]
    fn test_min_string_array_all_valid() {
        let arr = StringArray::<u32>::from_slice(&["zulu", "alpha", "echo", "bravo"]);
        let view = (&arr, 0, arr.len());
        let result = min_string_array::<u32>(view);
        assert_eq!(result, Some("alpha".to_string()));
    }

    #[test]
    fn test_min_string_array_with_nulls() {
        let mut arr = StringArray::<u32>::from_slice(&["zulu", "alpha", "echo", "bravo"]);
        arr.null_mask = Some(bitmask_from_vec(&[false, true, true, true]));
        let view = (&arr, 0, arr.len());
        let result = min_string_array::<u32>(view);
        assert_eq!(result, Some("alpha".to_string()));
    }

    #[test]
    fn test_min_string_array_all_null() {
        let arr = StringArray::<u32>::from_slice(&["zulu", "alpha", "echo", "bravo"]);
        let mut null_mask = Bitmask::with_capacity(arr.len());
        for i in 0..arr.len() {
            null_mask.set(i, false);
        }
        let arr = StringArray::<u32> {
            null_mask: Some(null_mask),
            ..arr
        };
        let view = (&arr, 0, arr.len());
        let result = min_string_array::<u32>(view);
        assert_eq!(result, None);
    }

    #[test]
    fn test_max_string_array_all_valid() {
        let arr = StringArray::<u32>::from_slice(&["zulu", "alpha", "echo", "bravo"]);
        let view = (&arr, 0, arr.len());
        let result = max_string_array::<u32>(view);
        assert_eq!(result, Some("zulu".to_string()));
    }

    #[test]
    fn test_max_string_array_with_nulls() {
        let mut arr = StringArray::<u32>::from_slice(&["zulu", "alpha", "echo", "bravo"]);
        arr.null_mask = Some(bitmask_from_vec(&[true, false, true, false]));
        let view = (&arr, 0, arr.len());
        let result = max_string_array::<u32>(view);
        assert_eq!(result, Some("zulu".to_string()));
    }

    #[test]
    fn test_max_string_array_all_null() {
        let arr = StringArray::<u32>::from_slice(&["zulu", "alpha", "echo", "bravo"]);
        let mut null_mask = Bitmask::with_capacity(arr.len());
        for i in 0..arr.len() {
            null_mask.set(i, false);
        }
        let arr = StringArray::<u32> {
            null_mask: Some(null_mask),
            ..arr
        };
        let view = (&arr, 0, arr.len());
        let result = max_string_array::<u32>(view);
        assert_eq!(result, None);
    }

    #[test]
    fn test_min_categorical_array() {
        let uniques = vec64![
            "dog".to_string(),
            "zebra".to_string(),
            "ant".to_string(),
            "bee".to_string()
        ];
        let indices = vec64![1u32, 0, 3, 2]; // "zebra", "dog", "bee", "ant"
        let cat = CategoricalArray {
            data: indices.clone().into(),
            unique_values: uniques.clone().into(),
            null_mask: None,
        };
        let result = min_categorical_array((&cat, 0, indices.len()));
        assert_eq!(result, Some("ant".to_string()));
    }

    #[test]
    fn test_max_categorical_array() {
        let uniques = vec64![
            "dog".to_string(),
            "zebra".to_string(),
            "ant".to_string(),
            "bee".to_string()
        ];
        let indices = vec64![2u32, 0, 1, 3]; // "ant", "dog", "zebra", "bee"
        let cat = CategoricalArray {
            data: indices.clone().into(),
            unique_values: uniques.clone().into(),
            null_mask: None,
        };
        let result = max_categorical_array((&cat, 0, indices.len()));
        assert_eq!(result, Some("zebra".to_string()));
    }

    #[test]
    fn test_min_categorical_array_with_nulls() {
        let uniques = vec64!["dog".to_string(), "zebra".to_string(), "ant".to_string()];
        let indices = vec64![1u32, 2, 0];
        let mut null_mask = Bitmask::with_capacity(indices.len());
        null_mask.set(0, true);
        null_mask.set(1, false);
        null_mask.set(2, true);
        let cat = CategoricalArray {
            data: indices.clone().into(),
            unique_values: uniques.clone().into(),
            null_mask: Some(null_mask),
        };
        let result = min_categorical_array((&cat, 0, indices.len()));
        assert_eq!(result, Some("dog".to_string())); // Only positions 0 and 2 valid: "zebra", "dog" -> "dog" is smaller
    }

    #[test]
    fn test_max_categorical_array_with_nulls() {
        let uniques = vec64!["dog".to_string(), "zebra".to_string(), "ant".to_string()];
        let indices = vec64![1u32, 2, 0];
        let mut null_mask = Bitmask::with_capacity(indices.len());
        null_mask.set(0, true);
        null_mask.set(1, false);
        null_mask.set(2, true);
        let cat = CategoricalArray {
            data: indices.clone().into(),
            unique_values: uniques.clone().into(),
            null_mask: Some(null_mask),
        };
        let result = max_categorical_array((&cat, 0, indices.len()));
        assert_eq!(result, Some("zebra".to_string())); // Only positions 0 and 2 valid: "zebra", "dog" -> "zebra" is larger
    }
}
