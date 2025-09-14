// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under MIT License.

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

use core::simd::{LaneCount, Simd, SupportedLaneCount};

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
    LaneCount<LANES>: SupportedLaneCount,
{
    let (lhs_mask, lhs_off, len) = lhs;
    let (rhs_mask, rhs_off, _) = rhs;
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
    LaneCount<LANES>: SupportedLaneCount,
{
    let (mask, offset, len) = src;
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
    LaneCount<LANES>: SupportedLaneCount,
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
    LaneCount<LANES>: SupportedLaneCount,
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
    LaneCount<LANES>: SupportedLaneCount,
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
    LaneCount<LANES>: SupportedLaneCount,
{
    bitmask_unop_simd::<LANES>(src, UnaryOperator::Not)
}

/// Bitwise "in" for boolean bitmasks: each output bit is true if lhs bit is in the set of bits in rhs.
#[inline]
pub fn in_mask_simd<const LANES: usize>(lhs: BitmaskVT<'_>, rhs: BitmaskVT<'_>) -> Bitmask
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let (lhs_mask, lhs_off, len) = lhs;
    let (rhs_mask, rhs_off, rlen) = rhs;
    debug_assert_eq!(len, rlen, "in_mask: window length mismatch");

    // Scan rhs to see which values are present (true, false)
    let mut has_true = false;
    let mut has_false = false;
    for i in 0..len {
        let v = unsafe { rhs_mask.get_unchecked(rhs_off + i) };
        if v {
            has_true = true;
        } else {
            has_false = true;
        }
        if has_true && has_false {
            break;
        }
    }

    match (has_true, has_false) {
        (true, true) => {
            // Set contains both: every bit is in the set
            Bitmask::new_set_all(len, true)
        }
        (true, false) => {
            // Only 'true' in rhs: output bit is set iff lhs bit is true
            lhs_mask.slice_clone(lhs_off, len)
        }
        (false, true) => {
            // Only 'false' in rhs: output bit is set iff lhs bit is false
            not_mask_simd::<LANES>((lhs_mask, lhs_off, len))
        }
        (false, false) => {
            // Set is empty: all bits false
            Bitmask::new_set_all(len, false)
        }
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
    LaneCount<LANES>: SupportedLaneCount,
{
    let mask = in_mask_simd::<LANES>(lhs, rhs);
    not_mask_simd::<LANES>((&mask, 0, mask.len))
}

/// Produces a bitmask where each output bit is 1 iff the corresponding bits of `a` and `b` are equal.
#[inline]
pub fn eq_mask_simd<const LANES: usize>(a: BitmaskVT<'_>, b: BitmaskVT<'_>) -> Bitmask
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let (am, ao, len) = a;
    let (bm, bo, blen) = b;
    debug_assert_eq!(len, blen, "BitWindow length mismatch in eq_bits_mask");
    if ao % 64 != 0 || bo % 64 != 0 {
        panic!(
            "eq_bits_mask: offsets must be 64-bit aligned (got a: {}, b: {})",
            ao, bo
        );
    }
    let a_words = ao / 64;
    let b_words = bo / 64;
    let n_words = (len + 63) / 64;
    let mut out = Bitmask::new_set_all(len, false);

    {
        use core::simd::Simd;
        let simd_chunks = n_words / LANES;
        let tail_words = n_words % LANES;
        for chunk in 0..simd_chunks {
            let mut arr_a = [0u64; LANES];
            let mut arr_b = [0u64; LANES];
            for lane in 0..LANES {
                arr_a[lane] = unsafe { am.word_unchecked(a_words + chunk * LANES + lane) };
                arr_b[lane] = unsafe { bm.word_unchecked(b_words + chunk * LANES + lane) };
            }
            let sa = Simd::<u64, LANES>::from_array(arr_a);
            let sb = Simd::<u64, LANES>::from_array(arr_b);
            let eq = !(sa ^ sb);
            for lane in 0..LANES {
                unsafe {
                    out.set_word_unchecked(chunk * LANES + lane, eq[lane]);
                }
            }
        }
        let base = simd_chunks * LANES;
        for k in 0..tail_words {
            let wa = unsafe { am.word_unchecked(a_words + base + k) };
            let wb = unsafe { bm.word_unchecked(b_words + base + k) };
            let eq = !(wa ^ wb);
            unsafe {
                out.set_word_unchecked(base + k, eq);
            }
        }
    }
    #[cfg(not(feature = "simd"))]
    {
        for k in 0..n_words {
            let wa = unsafe { am.word_unchecked(a_words + k) };
            let wb = unsafe { bm.word_unchecked(b_words + k) };
            let eq = !(wa ^ wb);
            unsafe {
                out.set_word_unchecked(k, eq);
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
    LaneCount<LANES>: SupportedLaneCount,
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
    LaneCount<LANES>: SupportedLaneCount,
{
    !all_eq_mask_simd::<LANES>(a, b)
}

/// Vectorised equality test across entire bitmask windows with early termination optimisation.
///
/// Performs bulk equality comparison between two bitmask windows using SIMD comparison operations.
/// The implementation processes multiple words simultaneously and uses early termination to avoid
/// unnecessary work when differences are detected.
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
    LaneCount<LANES>: SupportedLaneCount,
{
    let (am, ao, len) = a;

    // Mask < 64 bits early exit
    if len < 64 {
        for i in 0..len {
            if a.0.get(a.1 + i) != unsafe { b.0.get_unchecked(b.1 + i) } {
                return false;
            }
        }
        return true;
    }

    let (bm, bo, blen) = b;
    debug_assert_eq!(len, blen, "BitWindow length mismatch in all_eq_mask");
    if ao % 64 != 0 || bo % 64 != 0 {
        panic!(
            "all_eq_mask_simd: offsets must be 64-bit aligned (got a: {}, b: {})",
            ao, bo
        );
    }
    let a_words = ao / 64;
    let b_words = bo / 64;
    let n_words = (len + 63) / 64;
    let trailing = len & 63;

    use core::simd::Simd;
    use std::simd::prelude::SimdPartialEq;

    let simd_chunks = n_words / LANES;
    let tail_words = n_words % LANES;

    for chunk in 0..simd_chunks {
        let mut arr_a = [0u64; LANES];
        let mut arr_b = [0u64; LANES];
        for lane in 0..LANES {
            arr_a[lane] = unsafe { am.word_unchecked(a_words + chunk * LANES + lane) };
            arr_b[lane] = unsafe { bm.word_unchecked(b_words + chunk * LANES + lane) };
        }
        let sa = Simd::<u64, LANES>::from_array(arr_a);
        let sb = Simd::<u64, LANES>::from_array(arr_b);
        let eq_mask = sa.simd_eq(sb);
        if !eq_mask.all() {
            return false;
        }
    }

    let base = simd_chunks * LANES;
    for k in 0..tail_words {
        let idx = base + k;
        let wa = unsafe { am.word_unchecked(a_words + idx) };
        let wb = unsafe { bm.word_unchecked(b_words + idx) };
        // For the last (possibly partial) word, mask slack bits
        if idx == n_words - 1 && trailing != 0 {
            let mask = (1u64 << trailing) - 1;
            if (wa & mask) != (wb & mask) {
                return false;
            }
        } else if wa != wb {
            return false;
        }
    }
    true
}

/// Vectorised population count (number of set bits) with SIMD reduction for optimal performance.
///
/// Computes the total number of set bits in a bitmask window using SIMD population count instructions
/// followed by horizontal reduction. This implementation provides significant performance improvements
/// for large bitmasks through parallel processing of multiple words.
///
/// # Type Parameters
/// - `LANES`: Number of u64 lanes to process simultaneously for vectorised popcount operations
///
/// # Parameters
/// - `m`: Bitmask window as `(mask, offset, length)` tuple
///
/// # Returns
/// The total count of set bits in the specified window.
#[inline]
pub fn popcount_mask_simd<const LANES: usize>(m: BitmaskVT<'_>) -> usize
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let (mask, offset, len) = m;
    let n_words = (len + 63) / 64;
    let word_start = offset / 64;
    let mut acc = 0usize;

    {
        use core::simd::Simd;
        use std::simd::prelude::SimdUint;

        let simd_chunks = n_words / LANES;
        let tail_words = n_words % LANES;

        for chunk in 0..simd_chunks {
            let mut arr = [0u64; LANES];
            for lane in 0..LANES {
                arr[lane] = unsafe { mask.word_unchecked(word_start + chunk * LANES + lane) };
            }
            let v = Simd::<u64, LANES>::from_array(arr);
            let counts = v.count_ones();
            acc += counts.reduce_sum() as usize;
        }

        // Tail scalar loop for any remaining words
        let base = simd_chunks * LANES;
        for k in 0..tail_words {
            let word = unsafe { mask.word_unchecked(word_start + base + k) };
            // Mask off slack bits in final word if needed
            if base + k == n_words - 1 && len % 64 != 0 {
                let valid = len % 64;
                let slack_mask = (1u64 << valid) - 1;
                acc += (word & slack_mask).count_ones() as usize;
            } else {
                acc += word.count_ones() as usize;
            }
        }
    }
    acc
}

/// Returns true if all bits in the mask are set (1).
#[inline]
pub fn all_true_mask_simd<const LANES: usize>(mask: &Bitmask) -> bool
where
    LaneCount<LANES>: SupportedLaneCount,
{
    if mask.len < 64 {
        for i in 0..mask.len {
            if !unsafe { mask.get_unchecked(i) } {
                return false;
            }
        }
        return true;
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
    LaneCount<LANES>: SupportedLaneCount,
{
    if mask.len < 64 {
        for i in 0..mask.len {
            if unsafe { mask.get_unchecked(i) } {
                return false;
            }
        }
        return true;
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
