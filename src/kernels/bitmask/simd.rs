// Copyright 2025 Peter Garfield Bower
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # **Bitmask SIMD Kernels** - *Vectorised High-Performance Bitmask Operations*
//!
//! SIMD-accelerated implementations of bitmask operations using portable vectorisation with `std::simd`.
//! These kernels provide optimal performance for large bitmask operations through
//! SIMD-parallel processing of multiple 64-bit words simultaneously.
//!
//! ## Overview
//!
//! This module contains vectorised implementations of all bitmask operations.
//! it uses configurable SIMD lane counts to adapt to different CPU architectures whilst maintaining code portability.
//!
//! We do not check for SIMD alignment here because it is guaranteed by the `Bitmask` as it is backed by *Minarrow*'s `Vec64`.
//!
//! ## Architecture Principles
//!
//! - **Portable SIMD**: Uses `std::simd` for cross-platform vectorisation without target-specific code
//! - **Configurable lanes**: Lane counts determined at build time for optimal performance per architecture
//! - **Hybrid processing**: SIMD inner loops with scalar tail handling for non-aligned lengths
//! - **Low-cost abstraction**: `Bitmask` is a light-weight structure over a `Vec64`. See Minarrow for details and benchmarks
//! demonstrating very low abstraction cost.
//!
//!
//! ### **Memory Access Patterns**
//! - Vectorised loads process multiple words per memory operation
//! - Sequential access patterns optimise cache utilisation
//! - Aligned access where possible for maximum performance
//! - Streaming patterns for large bitmask operations
//!
//! ## Specialised Algorithms
//!
//! ### **Population Count (Popcount)**
//! Uses SIMD reduction for optimal performance:
//! ```rust,ignore
//! let counts = simd_vector.count_ones();
//! total += counts.reduce_sum() as usize;
//! ```
//!
//! ### **Equality Testing**
//! Leverages SIMD comparison operations:
//! ```rust,ignore
//! let eq_mask = vector_a.simd_eq(vector_b);
//! if !eq_mask.all() { return false; }
//! ```

use core::simd::Simd;

use crate::kernels::arithmetic::simd::{W8, W16, W32, W64};
use crate::{Bitmask, BitmaskVT};

use crate::enums::operators::{LogicalOperator, UnaryOperator};
use crate::kernels::bitmask::{
    bitmask_window_bytes, bitmask_window_bytes_mut, clear_trailing_bits, mask_bits_as_words,
    mask_bits_as_words_mut,
};

/// Primitive bit ops

/// Performs vectorised bitwise binary operations (AND/OR/XOR) with configurable lane counts.
///
/// Core SIMD implementation for logical operations between bitmask windows. Processes data using
/// vectorised instructions with automatic scalar tail handling for optimal performance across
/// different data sizes and architectures.
///
/// # Type Parameters
/// - `LANES`: Number of u64 lanes to process simultaneously (typically 8, 16, 32, or 64)
///
/// # Parameters
/// - `lhs`: Left-hand side bitmask window as `(mask, offset, length)` tuple
/// - `rhs`: Right-hand side bitmask window as `(mask, offset, length)` tuple
/// - `op`: Logical operation to perform (AND, OR, XOR)
///
/// # Returns
/// A new `Bitmask` containing the vectorised operation results with proper trailing bit handling.
///
/// # Performance Characteristics
/// - Vectorised inner loop processes `LANES` words per iteration
/// - Scalar tail handling ensures correctness for non-aligned lengths
/// - Memory access patterns optimised for cache efficiency
/// - Lane count scaling provides architecture-specific optimisation
#[inline(always)]
pub fn bitmask_binop_simd<const LANES: usize>(
    lhs: BitmaskVT<'_>,
    rhs: BitmaskVT<'_>,
    op: LogicalOperator,
) -> Bitmask
where
{
    let (lhs_mask, lhs_off, len) = lhs;
    let (rhs_mask, rhs_off, _) = rhs;
    if len == 0 {
        return Bitmask::new_set_all(0, false);
    }
    let mut out = Bitmask::new_set_all(len, false);
    let nw = (len + 63) / 64;
    unsafe {
        let lp = mask_bits_as_words(bitmask_window_bytes(lhs_mask, lhs_off, len));
        let rp = mask_bits_as_words(bitmask_window_bytes(rhs_mask, rhs_off, len));
        let dp = mask_bits_as_words_mut(bitmask_window_bytes_mut(&mut out, 0, len));
        let mut i = 0;
        while i + LANES <= nw {
            let a = Simd::<u64, LANES>::from_slice(std::slice::from_raw_parts(lp.add(i), LANES));
            let b = Simd::<u64, LANES>::from_slice(std::slice::from_raw_parts(rp.add(i), LANES));
            let r = match op {
                LogicalOperator::And => a & b,
                LogicalOperator::Or => a | b,
                LogicalOperator::Xor => a ^ b,
            };
            std::ptr::copy_nonoverlapping(r.as_array().as_ptr(), dp.add(i), LANES);
            i += LANES;
        }
        // Tail often caused by `n % LANES != 0`; uses scalar fallback.
        for k in i..nw {
            let a = *lp.add(k);
            let b = *rp.add(k);
            *dp.add(k) = match op {
                LogicalOperator::And => a & b,
                LogicalOperator::Or => a | b,
                LogicalOperator::Xor => a ^ b,
            };
        }
    }
    out.len = len;
    clear_trailing_bits(&mut out);
    out
}

/// Performs vectorised bitwise unary operations (NOT) with configurable lane counts.
///
/// Core SIMD implementation for unary logical operations on bitmask windows. Processes data using
/// vectorised instructions with automatic scalar tail handling for optimal performance across
/// different data sizes and CPU architectures.
///
/// # Type Parameters
/// - `LANES`: Number of u64 lanes to process simultaneously (typically 8, 16, 32, or 64)
///
/// # Parameters
/// - `src`: Source bitmask window as `(mask, offset, length)` tuple
/// - `op`: Unary operation to perform (currently only NOT supported)
///
/// # Returns
/// A new `Bitmask` containing the vectorised operation results with proper trailing bit handling.
///
/// # Implementation Details
/// - Vectorised inner loop processes `LANES` words per iteration using SIMD NOT operations
/// - Scalar tail handling ensures correctness for non-aligned lengths
/// - Memory access patterns optimised for cache efficiency and sequential processing
/// - Lane count scaling provides architecture-specific optimisation for different CPU capabilities
///
/// # Performance Characteristics
/// - Memory bandwidth: Vectorised loads/stores improve memory subsystem utilisation
/// - Instruction throughput: Reduced total instruction count for large operations
/// - Pipeline efficiency: Better utilisation of modern CPU execution units
/// - Cache locality: Sequential access patterns optimise cache utilisation
#[inline(always)]
pub fn bitmask_unop_simd<const LANES: usize>(src: BitmaskVT<'_>, op: UnaryOperator) -> Bitmask
where
{
    let (mask, offset, len) = src;
    if len == 0 {
        return Bitmask::new_set_all(0, false);
    }
    let mut out = Bitmask::new_set_all(len, false);
    let nw = (len + 63) / 64;
    unsafe {
        let sp = mask_bits_as_words(bitmask_window_bytes(mask, offset, len));
        let dp = mask_bits_as_words_mut(bitmask_window_bytes_mut(&mut out, 0, len));
        let mut i = 0;
        while i + LANES <= nw {
            let a = Simd::<u64, LANES>::from_slice(std::slice::from_raw_parts(sp.add(i), LANES));
            let r = match op {
                UnaryOperator::Not => !a,
                _ => unreachable!(),
            };
            std::ptr::copy_nonoverlapping(r.as_array().as_ptr(), dp.add(i), LANES);
            i += LANES;
        }
        // Tail often caused by `n % LANES != 0`; uses scalar fallback.
        for k in i..nw {
            let a = *sp.add(k);
            *dp.add(k) = match op {
                UnaryOperator::Not => !a,
                _ => unreachable!(),
            };
        }
    }
    out.len = len;
    clear_trailing_bits(&mut out);
    out
}

// ---- Entry points ----
/// Performs vectorised bitwise AND operation between two bitmask windows.
///
/// High-performance SIMD implementation of logical AND using configurable lane counts for optimal
/// CPU architecture utilisation. Delegates to the core `bitmask_binop_simd` implementation with
/// the AND operator.
///
/// # Type Parameters
/// - `LANES`: Number of u64 lanes to process simultaneously for vectorised operations
///
/// # Parameters
/// - `lhs`: Left-hand side bitmask window as `(mask, offset, length)` tuple
/// - `rhs`: Right-hand side bitmask window as `(mask, offset, length)` tuple
///
/// # Returns
/// A new `Bitmask` containing bitwise AND results with proper trailing bit masking.
///
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::kernels::bitmask::simd::and_masks_simd;
///
/// // Process 8 lanes simultaneously (512 bits per instruction)
/// let result = and_masks_simd::<8>(lhs_window, rhs_window);
/// ```
#[inline(always)]
pub fn and_masks_simd<const LANES: usize>(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask
where
{
    bitmask_binop_simd::<LANES>(lhs, rhs, LogicalOperator::And)
}

/// Performs vectorised bitwise OR operation between two bitmask windows.
///
/// High-performance SIMD implementation of logical OR using configurable lane counts for optimal
/// CPU architecture utilisation. Delegates to the core `bitmask_binop_simd` implementation with
/// the OR operator.
///
/// # Type Parameters
/// - `LANES`: Number of u64 lanes to process simultaneously for vectorised operations
///
/// # Parameters
/// - `lhs`: Left-hand side bitmask window as `(mask, offset, length)` tuple
/// - `rhs`: Right-hand side bitmask window as `(mask, offset, length)` tuple
///
/// # Returns
/// A new `Bitmask` containing bitwise OR results with proper trailing bit masking.
///
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::kernels::bitmask::simd::or_masks_simd;
///
/// // Process 16 lanes simultaneously (1024 bits per instruction)
/// let result = or_masks_simd::<16>(lhs_window, rhs_window);
/// ```
#[inline(always)]
pub fn or_masks_simd<const LANES: usize>(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask
where
{
    bitmask_binop_simd::<LANES>(lhs, rhs, LogicalOperator::Or)
}

/// Performs vectorised bitwise XOR operation between two bitmask windows.
///
/// High-performance SIMD implementation of logical exclusive-OR using configurable lane counts
/// for optimal CPU architecture utilisation. Delegates to the core `bitmask_binop_simd`
/// implementation with the XOR operator.
///
/// # Type Parameters
/// - `LANES`: Number of u64 lanes to process simultaneously for vectorised operations
///
/// # Parameters
/// - `lhs`: Left-hand side bitmask window as `(mask, offset, length)` tuple
/// - `rhs`: Right-hand side bitmask window as `(mask, offset, length)` tuple
///
/// # Returns
/// A new `Bitmask` containing bitwise XOR results with proper trailing bit masking.
///
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::kernels::bitmask::simd::xor_masks_simd;
///
/// // Process 32 lanes simultaneously (2048 bits per instruction)
/// let result = xor_masks_simd::<32>(lhs_window, rhs_window);
/// ```
#[inline(always)]
pub fn xor_masks_simd<const LANES: usize>(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask
where
{
    bitmask_binop_simd::<LANES>(lhs, rhs, LogicalOperator::Xor)
}

/// Performs vectorised bitwise NOT operation on a bitmask window.
///
/// High-performance SIMD implementation of logical NOT using configurable lane counts for optimal
/// CPU architecture utilisation. Delegates to the core `bitmask_unop_simd` implementation with
/// the NOT operator.
///
/// # Type Parameters
/// - `LANES`: Number of u64 lanes to process simultaneously for vectorised operations
///
/// # Parameters
/// - `src`: Source bitmask window as `(mask, offset, length)` tuple
///
/// # Returns
/// A new `Bitmask` containing bitwise NOT results with proper trailing bit masking.
///
/// # Usage Example
/// ```rust,ignore
/// use simd_kernels::kernels::bitmask::simd::not_mask_simd;
///
/// // Process 8 lanes simultaneously (512 bits per instruction)
/// let inverted = not_mask_simd::<8>(source_window);
/// ```
#[inline(always)]
pub fn not_mask_simd<const LANES: usize>(src: BitmaskVT<'_>) -> Bitmask
where
{
    bitmask_unop_simd::<LANES>(src, UnaryOperator::Not)
}

/// Bitwise "in" for boolean bitmasks: each output bit is true if lhs bit is in the set of bits in rhs.
#[inline]
pub fn in_mask_simd<const LANES: usize>(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask
where
{
    let (lhs_mask, lhs_off, len) = lhs;
    let (rhs_mask, rhs_off, rlen) = rhs;
    debug_assert_eq!(len, rlen, "in_mask: window length mismatch");

    if len == 0 {
        return Bitmask::new_set_all(0, false);
    }

    // Check which boolean values are present in rhs using word-level ops.
    // Trailing bits in the last word must be masked off to avoid false positives.
    let n_words = (len + 63) / 64;
    let trailing = len & 63;
    let mut any_set = 0u64;
    let mut any_unset = 0u64;
    unsafe {
        let rp = rhs_mask.bits.as_ptr().cast::<u64>().add(rhs_off / 64);
        for k in 0..n_words {
            let mut w = *rp.add(k);
            if k == n_words - 1 && trailing != 0 {
                let valid_mask = (1u64 << trailing) - 1;
                w &= valid_mask;
                any_set |= w;
                any_unset |= (!w) & valid_mask;
            } else {
                any_set |= w;
                any_unset |= !w;
            }
            if any_set != 0 && any_unset != 0 {
                break;
            }
        }
    }
    let has_true = any_set != 0;
    let has_false = any_unset != 0;

    match (has_true, has_false) {
        // Set contains both values: every bit is a member
        (true, true) => Bitmask::new_set_all(len, true),
        // Only true in rhs: output bit is set iff lhs bit is true
        (true, false) => lhs_mask.slice_clone(lhs_off, len),
        // Only false in rhs: output bit is set iff lhs bit is false
        (false, true) => not_mask_simd::<LANES>((lhs_mask, lhs_off, len)),
        // Empty set: no bits are members
        (false, false) => Bitmask::new_set_all(len, false),
    }
}

/// Performs vectorised bitwise "not in" membership test for boolean bitmasks.
///
/// Computes the logical complement of the "in" operation where each output bit is true if the
/// corresponding lhs bit is NOT in the set of bits defined by rhs. This function delegates to
/// `in_mask_simd` followed by `not_mask_simd` for optimal performance.
///
/// # Type Parameters
/// - `LANES`: Number of u64 lanes to process simultaneously for vectorised operations
///
/// # Parameters
/// - `lhs`: Left-hand side bitmask window as `(mask, offset, length)` tuple (test values)
/// - `rhs`: Right-hand side bitmask window as `(mask, offset, length)` tuple (set definition)
///
/// # Returns
/// A new `Bitmask` where each bit is true if the corresponding lhs bit is not in the rhs set.
#[inline]
pub fn not_in_mask_simd<const LANES: usize>(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask
where
{
    let mask = in_mask_simd::<LANES>(lhs, rhs);
    not_mask_simd::<LANES>((&mask, 0, mask.len))
}

/// Produces a bitmask where each output bit is 1 iff the corresponding bits of `a` and `b` are equal.
#[inline]
pub fn eq_mask_simd<const LANES: usize>(a: BitmaskVT<'_>, b: BitmaskVT<'_>) -> Bitmask
where
{
    let (am, ao, len) = a;
    let (bm, bo, blen) = b;
    debug_assert_eq!(len, blen, "BitWindow length mismatch in eq_bits_mask");
    if len == 0 {
        return Bitmask::new_set_all(0, true);
    }
    if ao % 64 != 0 || bo % 64 != 0 {
        panic!(
            "eq_bits_mask: offsets must be 64-bit aligned (got a: {}, b: {})",
            ao, bo
        );
    }
    let n_words = (len + 63) / 64;
    let mut out = Bitmask::new_set_all(len, false);

    unsafe {
        let ap = am.bits.as_ptr().cast::<u64>().add(ao / 64);
        let bp = bm.bits.as_ptr().cast::<u64>().add(bo / 64);
        let dp = out.bits.as_mut_ptr().cast::<u64>();
        let aw = std::slice::from_raw_parts(ap, n_words);
        let bw = std::slice::from_raw_parts(bp, n_words);

        #[cfg(feature = "simd")]
        {
            let mut i = 0;
            while i + LANES <= n_words {
                let sa = Simd::<u64, LANES>::from_slice(&aw[i..i + LANES]);
                let sb = Simd::<u64, LANES>::from_slice(&bw[i..i + LANES]);
                let eq = !(sa ^ sb);
                std::ptr::copy_nonoverlapping(eq.as_array().as_ptr(), dp.add(i), LANES);
                i += LANES;
            }
            for k in i..n_words {
                *dp.add(k) = !(aw[k] ^ bw[k]);
            }
        }
        #[cfg(not(feature = "simd"))]
        {
            for k in 0..n_words {
                *dp.add(k) = !(aw[k] ^ bw[k]);
            }
        }
    }
    out.mask_trailing_bits();
    out
}

/// Performs vectorised bitwise inequality comparison between two bitmask windows.
///
/// Computes the logical complement of equality where each output bit is true if the corresponding
/// bits from the two input windows are different. This function delegates to `eq_mask_simd`
/// followed by bitwise NOT for optimal performance.
///
/// # Type Parameters
/// - `LANES`: Number of u64 lanes to process simultaneously for vectorised operations
///
/// # Parameters
/// - `a`: First bitmask window as `(mask, offset, length)` tuple
/// - `b`: Second bitmask window as `(mask, offset, length)` tuple
///
/// # Returns
/// A new `Bitmask` where each bit is true if the corresponding input bits are different.
#[inline]
pub fn ne_mask_simd<const LANES: usize>(a: BitmaskVT<'_>, b: BitmaskVT<'_>) -> Bitmask
where
{
    !eq_mask_simd::<LANES>(a, b)
}

/// Tests if all corresponding bits between two bitmask windows are different.
///
/// Performs bulk inequality comparison across entire bitmask windows by computing the logical
/// complement of `all_eq_mask_simd`. Returns true only if every corresponding bit pair differs
/// between the two input windows.
///
/// # Type Parameters
/// - `LANES`: Number of u64 lanes to process simultaneously for vectorised comparison
///
/// # Parameters
/// - `a`: First bitmask window as `(mask, offset, length)` tuple
/// - `b`: Second bitmask window as `(mask, offset, length)` tuple
///
/// # Returns
/// `true` if all corresponding bits are different, `false` if any bits are equal.
#[inline]
pub fn all_ne_mask_simd<const LANES: usize>(a: BitmaskVT<'_>, b: BitmaskVT<'_>) -> bool
where
{
    !all_eq_mask_simd::<LANES>(a, b)
}

/// Vectorised equality test across entire bitmask windows with early termination.
///
/// Performs bulk equality comparison between two bitmask windows using SIMD comparison operations.
/// Processes multiple words simultaneously and terminates early when differences are detected.
///
/// # Type Parameters
/// - `LANES`: Number of u64 lanes to process simultaneously for vectorised comparison
///
/// # Parameters
/// - `a`: First bitmask window as `(mask, offset, length)` tuple
/// - `b`: Second bitmask window as `(mask, offset, length)` tuple
///
/// # Returns
/// `true` if all corresponding bits are equal (ignoring slack bits), `false` otherwise.
#[inline]
pub fn all_eq_mask_simd<const LANES: usize>(a: BitmaskVT<'_>, b: BitmaskVT<'_>) -> bool
where
{
    let (am, ao, len) = a;
    let (bm, bo, blen) = b;
    debug_assert_eq!(len, blen, "BitWindow length mismatch in all_eq_mask");

    if len == 0 {
        return true;
    }

    // Short masks: single word comparison with trailing bit mask
    if len < 64 {
        let wa = unsafe { am.word_unchecked(ao / 64) };
        let wb = unsafe { bm.word_unchecked(bo / 64) };
        let valid_mask = (1u64 << len) - 1;
        return (wa & valid_mask) == (wb & valid_mask);
    }

    if ao % 64 != 0 || bo % 64 != 0 {
        panic!(
            "all_eq_mask_simd: offsets must be 64-bit aligned (got a: {}, b: {})",
            ao, bo
        );
    }
    let n_words = (len + 63) / 64;
    let trailing = len & 63;

    unsafe {
        let aw = std::slice::from_raw_parts(am.bits.as_ptr().cast::<u64>().add(ao / 64), n_words);
        let bw = std::slice::from_raw_parts(bm.bits.as_ptr().cast::<u64>().add(bo / 64), n_words);

        #[cfg(feature = "simd")]
        {
            use std::simd::prelude::SimdPartialEq;
            let mut i = 0;
            while i + LANES <= n_words {
                let sa = Simd::<u64, LANES>::from_slice(&aw[i..i + LANES]);
                let sb = Simd::<u64, LANES>::from_slice(&bw[i..i + LANES]);
                if !sa.simd_eq(sb).all() {
                    return false;
                }
                i += LANES;
            }
            for k in i..n_words {
                if k == n_words - 1 && trailing != 0 {
                    let mask = (1u64 << trailing) - 1;
                    if (aw[k] & mask) != (bw[k] & mask) {
                        return false;
                    }
                } else if aw[k] != bw[k] {
                    return false;
                }
            }
        }
        #[cfg(not(feature = "simd"))]
        {
            for k in 0..n_words {
                if k == n_words - 1 && trailing != 0 {
                    let mask = (1u64 << trailing) - 1;
                    if (aw[k] & mask) != (bw[k] & mask) {
                        return false;
                    }
                } else if aw[k] != bw[k] {
                    return false;
                }
            }
        }
    }
    true
}

/// Vectorised population count with SIMD reduction.
///
/// Counts set bits in a bitmask window using SIMD popcount with horizontal reduction.
///
/// # Type Parameters
/// - `LANES`: Number of u64 lanes to process simultaneously
///
/// # Parameters
/// - `m`: Bitmask window as `(mask, offset, length)` tuple
///
/// # Returns
/// The total count of set bits in the specified window.
#[inline]
pub fn popcount_mask_simd<const LANES: usize>(m: BitmaskVT<'_>) -> usize
where
{
    let (mask, offset, len) = m;
    if len == 0 {
        return 0;
    }
    let n_words = (len + 63) / 64;
    let word_start = offset / 64;
    let mut acc = 0usize;

    unsafe {
        let words =
            std::slice::from_raw_parts(mask.bits.as_ptr().cast::<u64>().add(word_start), n_words);

        #[cfg(feature = "simd")]
        {
            use std::simd::prelude::SimdUint;
            let mut i = 0;
            while i + LANES <= n_words {
                let v = Simd::<u64, LANES>::from_slice(&words[i..i + LANES]);
                acc += v.count_ones().reduce_sum() as usize;
                i += LANES;
            }
            for k in i..n_words {
                if k == n_words - 1 && len % 64 != 0 {
                    let slack_mask = (1u64 << (len % 64)) - 1;
                    acc += (words[k] & slack_mask).count_ones() as usize;
                } else {
                    acc += words[k].count_ones() as usize;
                }
            }
        }
        #[cfg(not(feature = "simd"))]
        {
            for k in 0..n_words {
                if k == n_words - 1 && len % 64 != 0 {
                    let slack_mask = (1u64 << (len % 64)) - 1;
                    acc += (words[k] & slack_mask).count_ones() as usize;
                } else {
                    acc += words[k].count_ones() as usize;
                }
            }
        }
    }
    acc
}

/// Returns true if all bits in the mask are set (1).
#[inline]
pub fn all_true_mask_simd<const LANES: usize>(mask: &Bitmask) -> bool
where
{
    if mask.len == 0 {
        return true;
    }
    // Short masks: single word comparison
    if mask.len < 64 {
        let w = unsafe { mask.word_unchecked(0) };
        let valid_mask = (1u64 << mask.len) - 1;
        return (w & valid_mask) == valid_mask;
    }
    let n_bits = mask.len;
    let n_words = (n_bits + 63) / 64;
    let words: &[u64] =
        unsafe { std::slice::from_raw_parts(mask.bits.as_ptr() as *const u64, n_words) };

    let simd_chunks = n_words / LANES;

    let all_ones = Simd::<u64, LANES>::splat(!0u64);
    for chunk in 0..simd_chunks {
        let base = chunk * LANES;
        let arr = Simd::<u64, LANES>::from_slice(&words[base..base + LANES]);
        if arr != all_ones {
            return false;
        }
    }
    // Tail often caused by `n % LANES =! 0`; uses scalar fallback
    let tail_words = n_words % LANES;
    let base = simd_chunks * LANES;
    for k in 0..tail_words {
        if base + k == n_words - 1 && n_bits % 64 != 0 {
            let valid_bits = n_bits % 64;
            let slack_mask = (1u64 << valid_bits) - 1;
            if words[base + k] != slack_mask {
                return false;
            }
        } else {
            if words[base + k] != !0u64 {
                return false;
            }
        }
    }
    true
}

/// Returns true if all bits in the mask are set to (0).
pub fn all_false_mask_simd<const LANES: usize>(mask: &Bitmask) -> bool
where
{
    if mask.len == 0 {
        return true;
    }
    // Short masks: single word comparison
    if mask.len < 64 {
        let w = unsafe { mask.word_unchecked(0) };
        let valid_mask = (1u64 << mask.len) - 1;
        return (w & valid_mask) == 0;
    }
    let n_bits = mask.len;
    let n_words = (n_bits + 63) / 64;
    let words: &[u64] =
        unsafe { std::slice::from_raw_parts(mask.bits.as_ptr() as *const u64, n_words) };

    let simd_chunks = n_words / LANES;
    for chunk in 0..simd_chunks {
        let base = chunk * LANES;
        let arr = Simd::<u64, LANES>::from_slice(&words[base..base + LANES]);
        if arr != Simd::<u64, LANES>::splat(0u64) {
            return false;
        }
    }
    let tail_words = n_words % LANES;
    let base = simd_chunks * LANES;
    for k in 0..tail_words {
        if base + k == n_words - 1 && n_bits % 64 != 0 {
            let valid_bits = n_bits % 64;
            let slack_mask = (1u64 << valid_bits) - 1;
            if words[base + k] & slack_mask != 0 {
                return false;
            }
        } else {
            if words[base + k] != 0u64 {
                return false;
            }
        }
    }
    true
}

/// Generates a SIMD equality mask function for a given element type and lane count.
/// Processes LANES elements per iteration, with a scalar tail for the remainder.
macro_rules! impl_simd_eq_mask {
    ($fn_name:ident, $t:ty, $lanes:expr) => {
        pub fn $fn_name(data: &[$t], field_mask: $t, target: $t) -> Bitmask {
            use std::simd::cmp::SimdPartialEq;
            use vec64::Vec64;
            let n = data.len();
            let n_bytes = (n + 7) / 8;
            let mut bytes = Vec64::<u8>::with_capacity(n_bytes);
            bytes.resize(n_bytes, 0);

            let mask_vec = Simd::<$t, $lanes>::splat(field_mask);
            let target_vec = Simd::<$t, $lanes>::splat(target);

            let chunks = n / $lanes;
            for i in 0..chunks {
                let d = Simd::<$t, $lanes>::from_slice(&data[i * $lanes..]);
                let masked = d & mask_vec;
                let cmp = masked.simd_eq(target_vec);
                let bits = cmp.to_bitmask() as u64;
                let bit_pos = i * $lanes;
                let byte_idx = bit_pos / 8;
                let bit_shift = bit_pos % 8;
                // Write result bits. For LANES >= 8 bit_shift is always 0
                // since LANES is a power of 2. For LANES < 8, sub-byte
                // results OR into position within the byte.
                let shifted = bits << bit_shift;
                for b in 0..(($lanes + 7) / 8) {
                    bytes[byte_idx + b] |= (shifted >> (b * 8)) as u8;
                }
            }

            // Scalar tail
            let start = chunks * $lanes;
            for j in start..n {
                if (data[j] & field_mask) == target {
                    bytes[j / 8] |= 1 << (j % 8);
                }
            }

            Bitmask::new(bytes, n)
        }
    };
}

impl_simd_eq_mask!(simd_eq_mask_u8, u8, W8);
impl_simd_eq_mask!(simd_eq_mask_u16, u16, W16);
impl_simd_eq_mask!(simd_eq_mask_u32, u32, W32);
impl_simd_eq_mask!(simd_eq_mask_u64, u64, W64);

#[cfg(test)]
mod tests {
    use crate::{Bitmask, BitmaskVT};

    use super::*;

    macro_rules! simd_bitmask_suite {
        ($mod_name:ident, $lanes:expr) => {
            mod $mod_name {
                use super::*;
                const LANES: usize = $lanes;

                fn bm(bits: &[bool]) -> Bitmask {
                    let mut m = Bitmask::new_set_all(bits.len(), false);
                    for (i, b) in bits.iter().enumerate() {
                        if *b {
                            m.set(i, true);
                        }
                    }
                    m
                }
                fn slice(mask: &Bitmask) -> BitmaskVT<'_> {
                    (mask, 0, mask.len)
                }

                #[test]
                fn test_and_masks_simd() {
                    let a = bm(&[true, false, true, false, true, true, false, false]);
                    let b = bm(&[true, true, false, false, true, false, true, false]);
                    let c = and_masks_simd::<LANES>(slice(&a), slice(&b));
                    for i in 0..a.len {
                        assert_eq!(c.get(i), a.get(i) & b.get(i), "bit {i}");
                    }
                }

                #[test]
                fn test_or_masks_simd() {
                    let a = bm(&[true, false, true, false, true, true, false, false]);
                    let b = bm(&[true, true, false, false, true, false, true, false]);
                    let c = or_masks_simd::<LANES>(slice(&a), slice(&b));
                    for i in 0..a.len {
                        assert_eq!(c.get(i), a.get(i) | b.get(i), "bit {i}");
                    }
                }

                #[test]
                fn test_xor_masks_simd() {
                    let a = bm(&[true, false, true, false, true, true, false, false]);
                    let b = bm(&[true, true, false, false, true, false, true, false]);
                    let c = xor_masks_simd::<LANES>(slice(&a), slice(&b));
                    for i in 0..a.len {
                        assert_eq!(c.get(i), a.get(i) ^ b.get(i), "bit {i}");
                    }
                }

                #[test]
                fn test_not_mask_simd() {
                    let a = bm(&[true, false, true, false]);
                    let c = not_mask_simd::<LANES>(slice(&a));
                    for i in 0..a.len {
                        assert_eq!(c.get(i), !a.get(i));
                    }
                }

                #[test]
                fn test_in_mask_simd_variants() {
                    let lhs = bm(&[true, false, true, false]);
                    // RHS = [true]: only 'true' in rhs
                    let rhs_true = bm(&[true; 4]);
                    let out = in_mask_simd::<LANES>(slice(&lhs), slice(&rhs_true));
                    for i in 0..lhs.len {
                        assert_eq!(out.get(i), lhs.get(i), "in_mask, only true, bit {i}");
                    }
                    // RHS = [false]: only 'false' in rhs
                    let rhs_false = bm(&[false; 4]);
                    let out = in_mask_simd::<LANES>(slice(&lhs), slice(&rhs_false));
                    for i in 0..lhs.len {
                        assert_eq!(out.get(i), !lhs.get(i), "in_mask, only false, bit {i}");
                    }
                    // RHS = [true, false]: both present
                    let rhs_both = bm(&[true, false, true, false]);
                    let out = in_mask_simd::<LANES>(slice(&lhs), slice(&rhs_both));
                    for i in 0..lhs.len {
                        assert!(out.get(i), "in_mask, both true/false, bit {i}");
                    }
                    // RHS empty
                    let rhs_empty = bm(&[false; 0]);
                    let out = in_mask_simd::<LANES>((&lhs, 0, 0), (&rhs_empty, 0, 0));
                    assert_eq!(out.len, 0);
                }

                #[test]
                fn test_not_in_mask_simd() {
                    let lhs = bm(&[true, false, true, false]);
                    let rhs = bm(&[true, false, true, false]);
                    let in_mask = in_mask_simd::<LANES>(slice(&lhs), slice(&rhs));
                    let not_in = not_in_mask_simd::<LANES>(slice(&lhs), slice(&rhs));
                    for i in 0..lhs.len {
                        assert_eq!(not_in.get(i), !in_mask.get(i));
                    }
                }

                #[test]
                fn test_eq_mask_simd_and_ne_mask_simd() {
                    let a = bm(&[true, false, true, false]);
                    let b = bm(&[true, false, false, true]);
                    let eq = eq_mask_simd::<LANES>(slice(&a), slice(&b));
                    let ne = ne_mask_simd::<LANES>(slice(&a), slice(&b));
                    for i in 0..a.len {
                        assert_eq!(eq.get(i), a.get(i) == b.get(i), "eq_mask bit {i}");
                        assert_eq!(ne.get(i), a.get(i) != b.get(i), "ne_mask bit {i}");
                    }
                }

                #[test]
                fn test_all_eq_mask_simd() {
                    let a = bm(&[true, false, true, false, true, true, false, false]);
                    let b = bm(&[true, false, true, false, true, true, false, false]);
                    assert!(all_eq_mask_simd::<LANES>(slice(&a), slice(&b)));
                    let mut b2 = b.clone();
                    b2.set(0, false);
                    assert!(!all_eq_mask_simd::<LANES>(slice(&a), slice(&b2)));
                }

                #[test]
                fn test_all_ne_mask_simd() {
                    let a = bm(&[true, false, true]);
                    let b = bm(&[false, true, false]);
                    assert!(all_ne_mask_simd::<LANES>(slice(&a), slice(&b)));
                    assert!(!all_ne_mask_simd::<LANES>(slice(&a), slice(&a)));
                }

                #[test]
                fn test_popcount_mask_simd() {
                    let a = bm(&[true, false, true, false, true, false, false, true]);
                    let pop = popcount_mask_simd::<LANES>(slice(&a));
                    assert_eq!(pop, 4);
                }

                #[test]
                fn test_all_true_mask_simd_and_false() {
                    let all_true = Bitmask::new_set_all(64 * LANES, true);
                    assert!(all_true_mask_simd::<LANES>(&all_true));
                    let mut not_true = all_true.clone();
                    not_true.set(3, false);
                    assert!(!all_true_mask_simd::<LANES>(&not_true));
                }

                #[test]
                fn test_all_false_mask_simd() {
                    let all_true = Bitmask::new_set_all(64 * LANES, true);
                    assert!(!all_false_mask_simd::<LANES>(&all_true));
                    let all_false = Bitmask::new_set_all(64 * LANES, false);
                    assert!(all_false_mask_simd::<LANES>(&all_false));
                }
            }
        };
    }

    include!(concat!(env!("OUT_DIR"), "/simd_lanes.rs"));

    simd_bitmask_suite!(simd_bitmask_w8, W8);
    simd_bitmask_suite!(simd_bitmask_w16, W16);
    simd_bitmask_suite!(simd_bitmask_w32, W32);
    simd_bitmask_suite!(simd_bitmask_w64, W64);
}
