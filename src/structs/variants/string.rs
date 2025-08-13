//! # StringArray Module- *Mid-Level, Inner Typed String Array* 
//!
//! Arrow-compatible UTF-8, variable-length string array backed by a compact
//! `offsets + data (+ optional null_mask)` layout.
//!
//! ## Overview
//! - Supports Arrow’s `String` (`u32` offsets) and `LargeString` (`u64` offsets).
//! - Storage:
//!   - **offsets**: length = `len + 1`; i-th string = `data[offsets[i]..offsets[i+1]]`
//!   - **data**: concatenated UTF-8 bytes
//!   - **null_mask** *(optional)*: `Bitmask` where `1 = valid`, `0 = null`
//! - Zero-copy friendly and interops with the Arrow C Data Interface.
//! - Append-oriented API with reserve/resize helpers and fast conversions
//!   (e.g., to `CategoricalArray`).
//!
//! ## Features
//! - Builders: `from_slice`, `from_vec`, `from_vec64`, `from_parts`.
//! - Mutation: `push_str`, `set_str`, `push_null`, `push_nulls`, `reserve`, `resize`.
//! - Iteration: `iter_str*` (by value), optional parallel iterators behind `parallel_proc`.
//! - Conversions: `to_categorical_array()`.
//!
//! ## When to use
//! Use for variable-length UTF-8 text with Arrow interop, compact memory layout,
//! and high-throughput append/scan workloads.
//!
//! ## Safety note
//! Trait methods from `MaskedArray` that return `&'static str` are for trait
//! compatibility only—the data actually borrows from `self`. Prefer the `*_str`
//! methods in this module for correct lifetime management.
use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::mem::transmute;
use std::ops::{Deref, DerefMut, Index, Range};

use num_traits::{NumCast, Zero};
#[cfg(feature = "parallel_proc")]
use rayon::iter::ParallelIterator;

use crate::structs::vec64::Vec64;
use crate::traits::masked_array::MaskedArray;
use crate::traits::print::MAX_PREVIEW;
use crate::traits::type_unions::Integer;
use crate::utils::validate_null_mask_len;
use crate::{
    Bitmask, Buffer, CategoricalArray, Length, Offset, StringAVT, impl_arc_masked_array, vec64,
};

/// # StringArray
///
/// UTF-8 encoded, variable-length Arrow-compatible string array
///
/// ## Role
/// - Many will prefer the higher level `Array` type, which dispatches to this when
/// necessary.
/// - Can be used as a standalone array or as the text arm of `TextArray` / `Array`.
///
/// ## Fields
/// - **Offsets**: indices into the `data` buffer. The i-th string is at `data[offsets[i]..offsets[i+1]]`.
/// - **Data**: concatenated UTF-8 encoded bytes for all strings.
/// - **Null mask**: optional bit-packed validity bitmap (1=valid, 0=null).
///
/// ## Arrow compatibility
/// The `Apache Arrow` framework defines two string types:
/// - `String`: uses 32-bit offsets (`u32`)
/// - `LargeString`: uses 64-bit offsets (`u64`)
///
/// Specify either `u32` or `u64` as the generic parameter depending on the target
/// Arrow type. Doing so maintains a memory layout compatible with Arrow, enabling
/// zero-copy data transfer between this structure and Arrow arrays.
///
/// ## Example
/// ```rust
/// use minarrow::{StringArray, MaskedArray, vec64};
///
/// let arr = StringArray::<u32>::from_vec64(vec64![
///     "alpha",
///     "beta",
///     "gamma"
/// ], None);
///
/// assert_eq!(arr.len(), 3);
/// assert_eq!(arr.get_str(0), Some("alpha"));
/// assert_eq!(arr.get_str(1), Some("beta"));
/// assert_eq!(arr.get_str(2), Some("gamma"));
/// ```
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug)]
pub struct StringArray<T> {
    /// Offsets into the values buffer. The i-th string is at values[offsets[i]..offsets[i+1]].
    pub offsets: Buffer<T>,

    /// Concatenated UTF-8 byte values for all strings.
    pub data: Buffer<u8>,

    /// Optional null mask (bit-packed; 1=valid, 0=null).
    pub null_mask: Option<Bitmask>,
}

impl<T: Integer> StringArray<T> {
    /// Constructs a new, empty array.
    #[inline]
    pub fn new(
        data: impl Into<Buffer<u8>>,
        null_mask: Option<Bitmask>,
        offsets: impl Into<Buffer<T>>,
    ) -> Self {
        let data: Buffer<u8> = data.into();
        let offsets: Buffer<T> = offsets.into();
        validate_null_mask_len(offsets.len() - 1, &null_mask);
        Self {
            data,
            null_mask,
            offsets,
        }
    }

    /// Constructs a dense StringArray from a slice of string slices (no nulls).
    #[inline]
    pub fn from_slice(slice: &[&str]) -> Self {
        let n = slice.len();
        let mut offsets = Vec64::with_capacity(n + 1);
        let mut data = Vec64::new();
        offsets.push(T::zero());
        for s in slice {
            data.extend_from_slice(s.as_bytes());
            offsets.push(NumCast::from(data.len()).expect("Offset conversion failed"));
        }
        Self {
            offsets: offsets.into(),
            data: data.into(),
            null_mask: None,
        }
    }

    /// Constructs a StringArray with reserved capacity.
    #[inline]
    pub fn with_capacity(n_strings: usize, values_cap: usize, null_mask: bool) -> Self {
        let mut offsets = Vec64::with_capacity(n_strings + 1);
        offsets.push(T::zero());
        Self {
            offsets: offsets.into(),
            data: Vec64::with_capacity(values_cap).into(),
            null_mask: if null_mask {
                Some(Bitmask::with_capacity(n_strings))
            } else {
                None
            },
        }
    }

    /// Constructs a StringArray from an already aligned set of strings
    /// represented as `&str`, packed consecutively into a `Vec64<u8>`, and
    /// paired with offsets defining the start of each string.
    #[inline]
    pub fn from_vec64(strings: Vec64<&str>, null_mask: Option<Bitmask>) -> Self {
        let mut offsets = Vec64::with_capacity(strings.len() + 1);
        let mut data = Vec64::new();
        let mut current_offset = T::zero();

        offsets.push(current_offset);
        for s in strings.iter() {
            let bytes = s.as_bytes();
            data.extend_from_slice(bytes);
            current_offset =
                current_offset + T::from(bytes.len()).expect("offset conversion failed");
            offsets.push(current_offset);
        }

        Self {
            offsets: offsets.into(),
            data: data.into(),
            null_mask,
        }
    }

    /// Constructs a StringArray from Vec64<String>
    #[inline]
    pub fn from_vec64_owned(strings: Vec64<String>, null_mask: Option<Bitmask>) -> Self {
        let mut offsets = Vec64::with_capacity(strings.len() + 1);
        let mut data = Vec64::new();
        let mut current_offset = T::zero();

        offsets.push(current_offset);
        for s in strings.iter() {
            let bytes = s.as_bytes();
            data.extend_from_slice(bytes);
            current_offset =
                current_offset + T::from(bytes.len()).expect("offset conversion failed");
            offsets.push(current_offset);
        }

        Self {
            offsets: offsets.into(),
            data: data.into(),
            null_mask,
        }
    }

    /// Converts a standard `Vec<&str>` into a 64-byte aligned StringArray.
    #[inline]
    pub fn from_vec(strings: Vec<&str>, null_mask: Option<Bitmask>) -> Self {
        Self::from_vec64(strings.into(), null_mask)
    }

    /// Take ownership of **offsets**, **values**, and an optional null bitmap.
    /// The usual Arrow invariaLnts must hold (`offsets[0]==0`, last offset ==
    /// `data.len()`, monotonically non-decreasing).
    #[inline]
    pub fn from_parts(offsets: Vec64<T>, data: Vec64<u8>, null_mask: Option<Bitmask>) -> Self {
        debug_assert!(!offsets.is_empty() && offsets[0].to_usize() == 0);
        debug_assert_eq!(offsets.last().unwrap().to_usize(), Some(data.len()));
        Self {
            offsets: offsets.into(),
            data: data.into(),
            null_mask,
        }
    }

    /// Returns the string value at the given index with the correct lifetime.
    ///
    /// # Panics
    /// Panics if the index is out-of-bounds or offsets are invalid.
    #[inline]
    pub fn get_str(&self, idx: usize) -> Option<&str> {
        if self.is_null(idx) {
            return None;
        }
        let start = self.offsets[idx].to_usize();
        let end = self.offsets[idx + 1].to_usize();
        Some(unsafe { std::str::from_utf8_unchecked(&self.data[start..end]) })
    }

    /// Sets the string at the given index, updating offsets, data buffer, and null mask.
    ///
    /// Panics if `idx >= self.len()`.
    #[inline]
    pub fn set_str(&mut self, idx: usize, value: &str) {
        assert!(idx < self.len(), "index out of bounds");

        let bytes = value.as_bytes();
        let old_end = self.offsets[idx + 1].to_usize();
        let old_start = self.offsets[idx].to_usize();
        let old_len = old_end - old_start;

        // Replace in-place if lengths match
        if old_len == bytes.len() {
            self.data[old_start..old_end].copy_from_slice(bytes);
        } else {
            // Remove old slice and insert new string
            drop(self.data.splice(old_start..old_end, bytes.iter().copied()));

            let delta = bytes.len() as isize - old_len as isize;
            for i in idx + 1..=self.len() {
                let off = self.offsets[i].to_usize() as isize + delta;
                self.offsets[i] = T::from_usize(off as usize);
            }
        }

        // Update null mask
        if let Some(mask) = &mut self.null_mask {
            mask.set(idx, true);
        } else {
            let mut m = Bitmask::new_set_all(self.len(), false);
            m.set(idx, true);
            self.null_mask = Some(m);
        }
    }

    /// Like `set`, but skips bounds checks on `idx`.
    #[inline(always)]
    pub unsafe fn set_str_unchecked(&mut self, idx: usize, value: &str) {
        // append new bytes
        let bytes = value.as_bytes();
        let old_len = self.data.len();
        self.data.extend_from_slice(bytes);
        let new_len = self.data.len();

        // update offsets[idx]..offsets[idx+1]
        let off = &mut self.offsets;
        let t_old = T::from_usize(old_len);
        let t_new = T::from_usize(new_len);
        // assume offsets has length ≥ idx+2
        off.as_mut_slice()[idx] = t_old;
        off.as_mut_slice()[idx + 1] = t_new;

        // mark present
        if let Some(mask) = &mut self.null_mask {
            mask.set(idx, true);
        } else {
            let mut m = Bitmask::new_set_all(self.len(), false);
            m.set(idx, true);
            self.null_mask = Some(m);
        }
    }

    /// Appends a string value to the array without any bounds or safety checks.
    ///
    /// # Safety
    /// Caller must ensure that:
    /// - `self.offsets` has at least one value already (offset[0])
    /// - `self.offsets` and `self.null_mask` (if present) have sufficient capacity
    /// - `value` is valid UTF-8 (guaranteed by &str)
    #[inline(always)]
    pub unsafe fn push_str_unchecked(&mut self, value: &str) {
        let bytes = value.as_bytes();
        let current_offset = *self.offsets.last().unwrap(); // trusted by contract
        let next_offset = current_offset + unsafe { T::from(bytes.len()).unwrap_unchecked() };

        self.data.extend_from_slice(bytes);
        self.offsets.push(next_offset);

        let idx = self.len() - 1; // safe because we just pushed an offset
        if let Some(mask) = self.null_mask.as_mut() {
            unsafe { mask.set_unchecked(idx, true) };
        }
    }

    /// Returns the string value at the given index without bounds checks, with correct lifetime.
    ///
    /// # Safety
    /// This method does not perform bounds checking on either the `offsets` or the `data`.
    /// Caller must guarantee that `idx + 1 < self.offsets.len()` and that the byte range
    /// `data[offsets[idx]..offsets[idx + 1]]` is valid and represents valid UTF-8.
    ///
    /// # Returns
    /// A `&str` slice with the actual lifetime tied to `self`, or `None` if the value is null.
    ///
    /// # Undefined Behaviour
    /// Invoking this with an out-of-bounds index or invalid UTF-8 data results in UB.
    #[inline(always)]
    pub unsafe fn get_str_unchecked(&self, idx: usize) -> &str {
        if let Some(mask) = &self.null_mask {
            if !unsafe { mask.get_unchecked(idx) } {
                return "";
            }
        }

        let start = unsafe { self.offsets.get_unchecked(idx).to_usize().unwrap() };
        let end = unsafe { self.offsets.get_unchecked(idx + 1).to_usize().unwrap() };
        unsafe { std::str::from_utf8_unchecked(&self.data[start..end]) }
    }

    /// Returns an iterator over all values in the array, skipping null checks.
    #[inline]
    pub fn iter_str(&self) -> impl Iterator<Item = &str> + '_ {
        (0..self.len()).map(move |i| {
            let start = self.offsets[i].to_usize();
            let end = self.offsets[i + 1].to_usize();
            unsafe { std::str::from_utf8_unchecked(&self.data[start..end]) }
        })
    }

    /// Returns an iterator over `Option<&str>` values.
    #[inline]
    pub fn iter_str_opt(&self) -> impl Iterator<Item = Option<&str>> + '_ {
        (0..self.len()).map(move |i| {
            if self.is_null(i) {
                None
            } else {
                let start = self.offsets[i].to_usize();
                let end = self.offsets[i + 1].to_usize();
                Some(unsafe { std::str::from_utf8_unchecked(&self.data[start..end]) })
            }
        })
    }

    /// Returns an iterator over a range of `&str` values, skipping null checks.
    #[inline]
    pub fn iter_str_range(&self, offset: usize, len: usize) -> impl Iterator<Item = &str> + '_ {
        (offset..offset + len).map(move |i| {
            let start = self.offsets[i].to_usize();
            let end = self.offsets[i + 1].to_usize();
            unsafe { std::str::from_utf8_unchecked(&self.data[start..end]) }
        })
    }

    /// Returns an iterator over a range of `Option<&str>` values.
    #[inline]
    pub fn iter_str_opt_range(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = Option<&str>> + '_ {
        (offset..offset + len).map(move |i| {
            if self.is_null(i) {
                None
            } else {
                let start = self.offsets[i].to_usize();
                let end = self.offsets[i + 1].to_usize();
                Some(unsafe { std::str::from_utf8_unchecked(&self.data[start..end]) })
            }
        })
    }

    /// Pushes a string onto the array
    #[inline]
    pub fn push_str(&mut self, value: &str) {
        let len_before = <T as NumCast>::from(self.data.len()).unwrap();
        self.data.extend_from_slice(value.as_bytes());
        let new_offset = len_before + <T as NumCast>::from(value.len()).unwrap();
        self.offsets.push(new_offset);
        let idx = self.len() - 1;
        if let Some(m) = &mut self.null_mask {
            m.set(idx, true);
        }
    }

    /// Bulk append - reserves for `count` more offsets and `byte_cap` more values.
    #[inline]
    pub fn reserve(&mut self, count: usize, byte_cap: usize) {
        self.offsets.reserve(count);
        let len = self.len();
        self.data.reserve(byte_cap);
        if let Some(m) = &mut self.null_mask {
            m.ensure_capacity(len + count);
        }
    }

    /// Converts the `StringArray<T>` into a `CategoricalArray<T>`,
    /// preserving nulls and interning unique strings.
    pub fn to_categorical_array(&self) -> CategoricalArray<T> {
        let len = self.len();
        let mut uniques = Vec64::<String>::new();
        let mut dict = HashMap::<&str, usize>::new();
        let mut indices = Vec64::<T>::with_capacity(len);

        for i in 0..len {
            if self.is_null(i) {
                indices.push(T::from_usize(0));
                continue;
            }

            let start = self.offsets[i].to_usize();
            let end = self.offsets[i + 1].to_usize();
            let bytes = &self.data[start..end];
            let s = std::str::from_utf8(bytes).unwrap();

            let code = *dict.entry(s).or_insert_with(|| {
                let idx = uniques.len();
                uniques.push(s.to_string());
                idx
            });

            indices.push(T::from_usize(code));
        }

        CategoricalArray {
            data: indices.into(),
            unique_values: uniques.into(),
            null_mask: self.null_mask.clone(),
        }
    }

    /// Raw‐bytes accessor
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.data.as_ref()
    }

    /// Slices the data values from offset to offset + length,
    /// as a &[u8] slice, whilst retaining those parameters for any
    /// downstream reconstruction.
    ///
    /// As this slices raw bytes one may prefer `to_window` which carries
    /// a reference to the whole object, but is not pre-sliced.
    pub fn slice_tuple(&self, offset: usize, len: usize) -> (&[u8], Offset, Length) {
        (&self.data.as_ref()[offset..offset + len], offset, len)
    }
}

/// ⚠️ The string implementation of `MaskedArray` is primarily to support
/// the type contract, and null handling. Many of the methods have
/// `_str` variants e.g., `get_str` vs. `get`, etc., and are the preferred
/// methods for standard usage. These are inlined accordingly.
/// However, when building on top of the trait
/// contract, one should carefully review the core methods below as there
/// are lifetime workarounds implemented to ensure ergonomic usage within
/// all the *other* types, due to `String` behaving differently by default.
///
/// In any case, the recommended pattern is to enum match via `Array` to
/// the `_str` variants, as the below equivalents have not been inlined with
/// respect to binary size.
///
/// See below for further details.
impl<T: Integer> MaskedArray for StringArray<T> {
    type T = T;

    type Container = Buffer<u8>;

    type LogicalType = String;

    type CopyType = &'static str;

    fn data(&self) -> &Self::Container {
        &self.data
    }

    fn data_mut(&mut self) -> &mut Self::Container {
        &mut self.data
    }

    /// Returns the string value at the given index, or `None` if the value is null.
    ///
    /// # ⚠️ WARNING - prefer `get_str`
    /// This method returns a `&static str` for trait compatibility. However, the returned
    /// reference **borrows from the backing buffer of the array** and must not outlive
    /// the lifetime of `self`. It is **not truly static**.
    ///
    /// *Instead, prefer `get_str` for practical use*, or, if you
    /// are using this to build on top of the trait, ensure that you *do not store* the values.
    ///
    /// This is an intentional (but unfortunate) design trade-off to maintain a uniform trait
    /// interface across array types without introducing lifetime parameters everywhere.
    /// For example, numeric types do not require lifetimes, and the alternatives would either be:
    /// 1)  Forcing all non-string types to use `<'a>` across these main trait methods
    /// 2)  Returning owned string copies rather than borrows.
    ///  
    /// Hence, prefer `get_str` when you have the option, and use this within its actual lifetime
    /// context when you don't have a choice (i.e., when building off the trait contract).
    ///
    /// ## ⚠️ Incorrect Usage (do **not** do this):
    ///
    /// ```
    /// use minarrow::{MaskedArray, StringArray};
    /// let s: &str;
    /// {
    ///     let arr = StringArray::<u32>::from_slice(&["a", "b"]);
    ///     s = unsafe { arr.get(0).unwrap() }; // Not truly static
    /// }
    /// // ⚠️ Use-after-free — Undefined Behaviour
    /// println!("{}", s);
    /// ```
    ///
    /// # Safety
    ///
    /// - Caller must ensure `idx` is within bounds.
    /// - The backing array must not be mutated or dropped for the duration of the borrow.
    /// - The offsets and data must be valid.
    ///
    /// # Panics
    /// Panics if offset values are inconsistent or indexing violates memory bounds.
    #[inline]
    fn get(&self, idx: usize) -> Option<&'static str> {
        if self.is_null(idx) {
            return None;
        }
        let start = self.offsets[idx].to_usize();
        let end = self.offsets[idx + 1].to_usize();

        Some(unsafe {
            std::mem::transmute::<&str, &'static str>(std::str::from_utf8_unchecked(
                &self.data[start..end],
            ))
        })
    }

    /// Sets the string at the given index, updating offsets, data buffer, and null mask.
    ///
    /// # ⚠️ Prefer `set_str` as it avoids a reallocation.
    ///
    /// Panics if `idx >= self.len()`.
    #[inline]
    fn set(&mut self, idx: usize, value: String) {
        self.set_str(idx, &value)
    }

    /// Returns the string value at the given index, or `None` if the value is null,
    /// without bounds checking. Prefer "unchecked" for performance critical code.
    ///
    /// # ⚠️ WARNING - prefer `get_str_unchecked`
    /// This method returns a `&static str` for trait compatibility. However, the returned
    /// reference **borrows from the backing buffer of the array** and must not outlive
    /// the lifetime of `self`. It is **not truly static**.
    ///
    /// *Instead, prefer `get_str` for practical use*, or, if you
    /// are using this to build on top of the trait, ensure that you *do not store* the values.
    ///
    /// This is an intentional (but unfortunate) design trade-off to maintain a uniform trait
    /// interface across array types without introducing lifetime parameters everywhere.
    /// For example, numeric types do not require lifetimes, and the alternatives would either be:
    /// 1)  Forcing all non-string types to use `<'a>` across these main trait methods
    /// 2)  Returning owned string copies rather than borrows.
    ///  
    /// Hence, prefer `get_str` when you have the option, and use this within its actual lifetime
    /// context when you don't have a choice (i.e., when building off the trait contract).
    ///
    /// ## ⚠️ Incorrect Usage (do **not** do this):
    ///
    /// ```
    /// use minarrow::{MaskedArray, StringArray};
    /// let s: &str;
    /// {
    ///     let arr = StringArray::<u32>::from_slice(&["a", "b"]);
    ///     s = unsafe { arr.get_unchecked(0).unwrap() }; // Not truly static
    /// }
    /// // ⚠️ Use-after-free — Undefined Behaviour
    /// println!("{}", s);
    /// ```
    ///
    /// # Safety
    ///
    /// - Caller must ensure `idx` is within bounds.
    /// - The backing array must not be mutated or dropped for the duration of the borrow.
    /// - The offsets and data must be valid.
    ///
    /// # Panics
    /// Panics if offset values are inconsistent or indexing violates memory bounds.
    #[inline]
    unsafe fn get_unchecked(&self, idx: usize) -> Option<&'static str> {
        // null‐check
        if let Some(mask) = &self.null_mask {
            if !mask.get(idx) {
                return None;
            }
        }
        // slice out the bytes without checking offsets bounds
        let start = unsafe { self.offsets.get_unchecked(idx).to_usize().unwrap() };
        let end = unsafe { self.offsets.get_unchecked(idx + 1).to_usize().unwrap() };
        Some(unsafe {
            std::mem::transmute::<&str, &'static str>(std::str::from_utf8_unchecked(
                &self.data[start..end],
            ))
        })
    }

    /// Like `set`, but skips bounds checks on `idx`.
    ///
    /// # ⚠️ Prefer `set_str_unchecked` as it avoids a reallocation.
    /// ```
    #[inline]
    unsafe fn set_unchecked(&mut self, idx: usize, value: String) {
        unsafe { self.set_str_unchecked(idx, &value) };
    }

    /// Returns an iterator over all values in the array, skipping null checks.
    ///
    /// # Safety Note
    ///
    /// # ⚠️ WARNING
    /// This method returns a `&static str` for trait compatibility. However, the returned
    /// reference **borrows from the backing buffer of the array** and must not outlive
    /// the lifetime of `self`. It is **not truly static**.
    ///
    /// *Instead, prefer `iter_str` for practical use*, or, if you
    /// are using this to build on top of the trait, ensure that you *do not store* the values.
    ///
    /// This is an intentional (but unfortunate) design trade-off to maintain a uniform trait
    /// interface across array types without introducing lifetime parameters everywhere.
    /// For example, numeric types do not require lifetimes, and the alternatives would either be
    /// 1)  Forcing all non-string types to use `<'a>` across these main trait methods
    /// 2)  Returning owned string copies rather than borrows.
    ///  
    /// Hence, prefer `iter_str` when you have the option, and use this within its actual lifetime
    /// context when you don't have a choice (i.e., when building off the trait contract).
    ///
    /// ## ⚠️ Incorrect Usage (do **not** do this):
    /// ```
    /// use minarrow::{MaskedArray, StringArray};
    /// let data: &str;
    /// {
    ///     let arr = StringArray::<u32>::from_slice(&["alpha", "beta"]);
    ///     data = arr.iter().next().unwrap(); // undefined behaviour
    /// }
    /// println!("{}", data); // ⚠️ Bad: Use-after-free — compiler will not complain
    /// ```
    #[inline]
    fn iter(&self) -> impl Iterator<Item = &'static str> + '_ {
        (0..self.len()).map(move |i| {
            let start = self.offsets[i].to_usize();
            let end = self.offsets[i + 1].to_usize();
            unsafe {
                transmute::<&str, &'static str>(std::str::from_utf8_unchecked(
                    &self.data[start..end],
                ))
            }
        })
    }

    /// Returns an iterator over `Option<&str>` where null entries return `None`.
    ///
    /// # ⚠️ Safety Note
    ///
    /// The same caveats apply as in [`iter`]: the returned `String` *must not*
    /// be assumed to live beyond the array. See `iter` for more explanation.
    /// ```
    #[inline]
    fn iter_opt(&self) -> impl Iterator<Item = Option<&'static str>> + '_ {
        (0..self.len()).map(move |i| {
            if self.is_null(i) {
                None
            } else {
                let start = self.offsets[i].to_usize();
                let end = self.offsets[i + 1].to_usize();
                Some(unsafe {
                    transmute::<&str, &'static str>(std::str::from_utf8_unchecked(
                        &self.data[start..end],
                    ))
                })
            }
        })
    }

    /// Returns an iterator over a range of `&'static str` values for trait compatibility.
    ///
    /// ⚠️ WARNING: The returned references are not truly `'static`.
    #[inline]
    fn iter_range(&self, offset: usize, len: usize) -> impl Iterator<Item = &'static str> + '_ {
        (offset..offset + len).map(move |i| {
            let start = self.offsets[i].to_usize();
            let end = self.offsets[i + 1].to_usize();
            unsafe {
                std::mem::transmute::<&str, &'static str>(std::str::from_utf8_unchecked(
                    &self.data[start..end],
                ))
            }
        })
    }

    /// Returns an iterator over a range of `Option<&'static str>` values.
    ///
    /// ⚠️ WARNING: The returned references are not truly `'static`.
    #[inline]
    fn iter_opt_range(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = Option<&'static str>> + '_ {
        (offset..offset + len).map(move |i| {
            if self.is_null(i) {
                None
            } else {
                let start = self.offsets[i].to_usize();
                let end = self.offsets[i + 1].to_usize();
                Some(unsafe {
                    std::mem::transmute::<&str, &'static str>(std::str::from_utf8_unchecked(
                        &self.data[start..end],
                    ))
                })
            }
        })
    }

    /// Appends a string value to the array, updating offsets and null mask as required.
    ///
    /// ⚠️ Prefer `push_str` as it avoids an additional String allocation.
    #[inline]
    fn push(&mut self, s: String) {
        self.push_str(&s)
    }

    /// Appends a `String` to the array without any bounds or safety checks.
    ///
    /// ⚠️ Prefer `push_str` as it avoids an additional String allocation.
    ///
    /// # Safety
    /// This simply forwards to `push_str_unchecked`, and has the same requirements.
    #[inline(always)]
    unsafe fn push_unchecked(&mut self, value: String) {
        unsafe { self.push_str_unchecked(&value) };
    }

    /// Appends a null value to the array, updating offsets and null mask as required.
    #[inline]
    fn push_null(&mut self) {
        let last = *self.offsets.last().unwrap();
        self.offsets.push(last);
        let idx = self.len() - 1;
        match self.null_mask.as_mut() {
            Some(m) => m.set(idx, false),
            None => {
                let mut nm = Bitmask::new_set_all(self.len(), true);
                nm.set(idx, false);
                self.null_mask = Some(nm);
            }
        }
    }

    /// Returns the total number of nulls.
    fn null_count(&self) -> usize {
        self.null_mask
            .as_ref()
            .map(|mask| mask.null_count())
            .unwrap_or(0)
    }

    /// Returns a logical slice of the string array, as a new `StringArray` object,
    /// Copies the selected strings.
    ///
    /// For a non-copy slice view, see `slice` from the parent Array object
    /// or the `AsRef` trait implementation.
    fn slice_clone(&self, offset: usize, len: usize) -> Self {
        assert!(offset + len <= self.len(), "slice out of bounds");

        let start_byte = self.offsets[offset].to_usize();
        let end_byte = self.offsets[offset + len].to_usize();

        let sliced_data = Vec64::from_slice(&self.data[start_byte..end_byte]);
        let mut sliced_offsets = Vec64::<T>::with_capacity(len + 1);
        let base = self.offsets[offset].to_usize();
        for i in 0..=len {
            let relative = self.offsets[offset + i].to_usize() - base;
            sliced_offsets.push(T::from(relative).unwrap());
        }
        let sliced_mask = self
            .null_mask
            .as_ref()
            .map(|mask| mask.slice_clone(offset, len));
        StringArray {
            offsets: sliced_offsets.into(),
            data: sliced_data.into(),
            null_mask: sliced_mask,
        }
    }

    /// Converts a `StringArray` with its window parameters
    /// to a `StringArrayView'a>` alias. Like a slice, but
    /// retains access to the `&StringArray`.
    ///
    /// `Offset` and `Length` are `usize` aliases.
    #[inline(always)]
    fn tuple_ref<'a>(&'a self, offset: Offset, len: Length) -> StringAVT<'a, T> {
        (&self, offset, len)
    }

    /// Resizes the array to contain `n` elements, each set to the provided logical string value.
    ///
    /// If `n` is greater than the current length, the `value` is appended repeatedly.
    /// If `n` is smaller, the array is truncated at the correct byte and offset boundary.
    ///
    /// The `null_mask` is left untouched. Callers are responsible for managing validity if needed.
    ///
    /// # Panics
    /// Panics if `offsets` are invalid or not of length `self.len() + 1`.
    fn resize(&mut self, n: usize, value: String) {
        let current_len = self.len();
        let value_bytes = value.as_bytes();
        let value_len = value_bytes.len();

        let mut current_offset = if let Some(last) = self.offsets.last() {
            last.to_usize().unwrap()
        } else {
            0
        };

        if n > current_len {
            self.offsets.reserve(n - current_len);
            self.data.reserve((n - current_len) * value_len);

            for _ in current_len..n {
                self.data.extend_from_slice(value_bytes);
                current_offset += value_len;
                self.offsets.push(T::from_usize(current_offset));
            }
        } else if n < current_len {
            let byte_end = self.offsets[n].to_usize();
            self.data.truncate(byte_end);
            self.offsets.truncate(n + 1);
        }
    }

    /// Returns a reference to the null bitmask
    fn null_mask(&self) -> Option<&Bitmask> {
        self.null_mask.as_ref()
    }

    /// Returns a mutable reference to the null bitmask
    fn null_mask_mut(&mut self) -> Option<&mut Bitmask> {
        self.null_mask.as_mut()
    }

    /// Sets the null bitmask
    fn set_null_mask(&mut self, mask: Option<Bitmask>) {
        self.null_mask = mask;
    }

    #[inline]
    unsafe fn push_null_unchecked(&mut self) {
        // first, append a default element
        let idx = self.len();
        unsafe { self.set_unchecked(idx, Self::LogicalType::default()) };

        if let Some(mask) = self.null_mask_mut() {
            // mark null
            unsafe { mask.set_unchecked(idx, false) };
        } else {
            // initialise a new mask and mark this slot null
            let mut m = Bitmask::new_set_all(idx, true);
            unsafe { m.set_unchecked(idx, false) };
            self.set_null_mask(Some(m));
        }
    }

    /// Sets the null mask at a specific index to null
    #[inline]
    fn set_null(&mut self, idx: usize) {
        if let Some(nmask) = &mut self.null_mask_mut() {
            nmask.set(idx, false);
        } else {
            let mut m = Bitmask::new_set_all(self.len(), true);
            m.set(idx, false);
            self.set_null_mask(Some(m));
        }
    }

    /// Sets the null mask at a specific index to null without bounds checks
    unsafe fn set_null_unchecked(&mut self, idx: usize) {
        if let Some(mask) = self.null_mask_mut() {
            mask.set(idx, false);
        } else {
            let mut m = Bitmask::new_set_all(self.len(), true);
            m.set(idx, false);
            self.set_null_mask(Some(m));
        }
    }

    /// Appends `n` null string values to the array.
    ///
    /// This extends both the `offsets` and `null_mask`. The `data` buffer remains unchanged
    /// because null entries have no payload.
    #[inline]
    fn push_nulls(&mut self, n: usize) {
        let start = self.len();
        let end = start + n;

        // Extend offsets with duplicates of the last offset value.
        let last = *self.offsets.last().unwrap_or(&T::from_usize(0));
        self.offsets.resize(end + 1, last);

        if let Some(mask) = self.null_mask_mut() {
            mask.resize(end, false);
        } else {
            let mut m = Bitmask::new_set_all(end, true);
            for i in start..end {
                m.set(i, false);
            }
            self.set_null_mask(Some(m));
        }
    }

    /// Appends `n` null values to the array without performing bounds checks on the mask.
    ///
    /// # Safety
    /// The caller must ensure that the mask and offset buffers have or will have sufficient capacity.
    #[inline]
    unsafe fn push_nulls_unchecked(&mut self, n: usize) {
        let start = self.len();
        let end = start + n;

        let last = *self.offsets.last().unwrap_or(&T::from_usize(0));
        self.offsets.resize(end + 1, last);

        if let Some(mask) = self.null_mask_mut() {
            mask.resize(end, true);
            for i in 0..n {
                unsafe { mask.set_unchecked(start + i, false) };
            }
        } else {
            let mut m = Bitmask::new_set_all(end, true);
            for i in start..end {
                unsafe { m.set_unchecked(i, false) };
            }
            self.set_null_mask(Some(m));
        }
    }

    /// Returns the length of the string buffer offsets
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Appends all values (and null mask if present) from `other` to `self`.
    fn append_array(&mut self, other: &Self) {
        let orig_len = self.len();
        let other_len = other.len();

        if other_len == 0 {
            return;
        }

        // 1. Append data
        self.data.extend_from_slice(&other.data);

        let prev_last_offset = *self
            .offsets
            .last()
            .expect("StringArray must have at least one offset");
        for off in other.offsets.iter().skip(1) {
            let new_offset = prev_last_offset + (*off - other.offsets[0]);
            self.offsets.push(new_offset);
        }

        // 3. Null mask
        match (self.null_mask_mut(), other.null_mask()) {
            (Some(self_mask), Some(other_mask)) => {
                self_mask.extend_from_bitmask(other_mask);
            }
            (Some(self_mask), None) => {
                self_mask.resize(orig_len + other_len, true);
            }
            (None, Some(other_mask)) => {
                let mut mask = Bitmask::new_set_all(orig_len + other_len, true);
                for i in 0..other_len {
                    mask.set(orig_len + i, other_mask.get(i));
                }
                self.set_null_mask(Some(mask));
            }
            (None, None) => {
                // No mask in either: nothing to do.
            }
        }
    }
}

#[cfg(feature = "parallel_proc")]
impl<T: Integer + Send + Sync> StringArray<T> {
    /// Parallel iterator over all string elements. Nulls are yielded as `""` (empty).
    #[inline]
    pub fn par_iter(&self) -> impl ParallelIterator<Item = &str> + '_ {
        use rayon::prelude::*;
        let data = &self.data;
        let offsets = &self.offsets;
        let null_mask = self.null_mask.as_ref();
        (0..self.len()).into_par_iter().map(move |i| {
            if null_mask.map(|m| !m.get(i)).unwrap_or(false) {
                ""
            } else {
                let s = offsets[i].to_usize();
                let e = offsets[i + 1].to_usize();
                unsafe { std::str::from_utf8_unchecked(&data[s..e]) }
            }
        })
    }

    /// Parallel iterator over Option<&str>, yields None if value is null.
    #[inline]
    pub fn par_iter_opt(&self) -> impl ParallelIterator<Item = Option<&str>> + '_ {
        self.par_iter_range_opt(0, self.len())
    }

    /// Zero-copy parallel iterator over window [start, end)
    #[inline]
    pub fn par_iter_range(
        &self,
        start: usize,
        end: usize,
    ) -> impl ParallelIterator<Item = &str> + '_ {
        use rayon::prelude::*;
        let data = &self.data;
        let offsets = &self.offsets;
        let null_mask = self.null_mask.as_ref();
        debug_assert!(start <= end && end <= self.len());
        (start..end).into_par_iter().map(move |i| {
            if null_mask.map(|m| !m.get(i)).unwrap_or(false) {
                ""
            } else {
                let s = offsets[i].to_usize();
                let e = offsets[i + 1].to_usize();
                unsafe { std::str::from_utf8_unchecked(&data[s..e]) }
            }
        })
    }

    /// Parallel iterator over window [start, end), None if null
    #[inline]
    pub fn par_iter_range_opt(
        &self,
        start: usize,
        end: usize,
    ) -> impl ParallelIterator<Item = Option<&str>> + '_ {
        use rayon::prelude::*;
        let data = &self.data;
        let offsets = &self.offsets;
        let null_mask = self.null_mask.as_ref();
        debug_assert!(start <= end && end <= self.len());
        (start..end).into_par_iter().map(move |i| {
            if null_mask.map(|m| !m.get(i)).unwrap_or(false) {
                None
            } else {
                let s = offsets[i].to_usize();
                let e = offsets[i + 1].to_usize();
                Some(unsafe { std::str::from_utf8_unchecked(&data[s..e]) })
            }
        })
    }

    /// Zero-copy parallel iterator over window [start, end) without bounds checks
    #[inline]
    pub unsafe fn par_iter_range_unchecked(
        &self,
        start: usize,
        end: usize,
    ) -> impl rayon::prelude::ParallelIterator<Item = &str> + '_ {
        use rayon::prelude::*;
        let data = &self.data;
        let offsets = &self.offsets;
        let null_mask = self.null_mask.as_ref();
        (start..end).into_par_iter().map(move |i| {
            if null_mask.map(|m| !m.get(i)).unwrap_or(false) {
                ""
            } else {
                let s = unsafe { *offsets.get_unchecked(i) }.to_usize();
                let e = unsafe { *offsets.get_unchecked(i + 1) }.to_usize();
                unsafe { std::str::from_utf8_unchecked(&data[s..e]) }
            }
        })
    }

    /// Parallel iterator over window [start, end) without bounds checks. None if null.
    #[inline]
    pub unsafe fn par_iter_range_opt_unchecked(
        &self,
        start: usize,
        end: usize,
    ) -> impl rayon::prelude::ParallelIterator<Item = Option<&str>> + '_ {
        use rayon::prelude::*;
        let data = &self.data;
        let offsets = &self.offsets;
        let null_mask = self.null_mask.as_ref();
        (start..end).into_par_iter().map(move |i| {
            if null_mask.map(|m| !m.get(i)).unwrap_or(false) {
                None
            } else {
                let s = unsafe { *offsets.get_unchecked(i) }.to_usize();
                let e = unsafe { *offsets.get_unchecked(i + 1) }.to_usize();
                Some(unsafe { std::str::from_utf8_unchecked(&data[s..e]) })
            }
        })
    }
}

impl_arc_masked_array!(
    Inner = StringArray<T>,
    T = T,
    Container = Buffer<u8>,
    LogicalType = String,
    CopyType = &'static str,
    BufferT = u8,
    Variant = TextArray,
    Bound = Integer,
);

impl<T: Zero> Default for StringArray<T> {
    fn default() -> Self {
        Self {
            offsets: vec64![T::zero()].into(),
            data: Vec64::new().into(),
            null_mask: None,
        }
    }
}

// Raw byte, not string slices
impl<T> Deref for StringArray<T> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.data.as_ref()
    }
}

impl<T> AsRef<[u8]> for StringArray<T> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.data.as_ref()
    }
}

impl<T> DerefMut for StringArray<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        self.data.as_mut()
    }
}

impl<T> AsMut<[u8]> for StringArray<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.data.as_mut()
    }
}

impl<T: Integer> Index<usize> for StringArray<T> {
    type Output = str;

    /// Returns the string at the specified index.
    ///
    /// # Panics
    /// Panics if the index is out-of-bounds or the data is not valid UTF-8.
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let start = self.offsets[index].to_usize();
        let end = self.offsets[index + 1].to_usize();
        std::str::from_utf8(&self.data[start..end]).expect("Invalid UTF-8")
    }
}

impl<T: crate::traits::type_unions::Integer> Index<Range<usize>> for StringArray<T> {
    type Output = str;

    #[inline]
    fn index(&self, range: Range<usize>) -> &Self::Output {
        let start = self.offsets[range.start].to_usize();
        let end = self.offsets[range.end].to_usize();
        unsafe { std::str::from_utf8_unchecked(&self.data[start..end]) }
    }
}

impl<T> Display for StringArray<T>
where
    T: Integer,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let len = self.len();
        let nulls = self.null_count();

        writeln!(
            f,
            "StringArray [{} values]s] (dtype: string, nulls: {})",
            len, nulls
        )?;

        write!(f, "[")?;

        for i in 0..usize::min(len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, ", ")?;
            }

            match self.get_str(i) {
                Some(s) => write!(f, "\"{}\"", s)?,
                None => write!(f, "null")?,
            }
        }

        if len > MAX_PREVIEW {
            write!(f, ", … ({} total)", len)?;
        }

        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offsets<T: Integer>(slice: &[u64]) -> Vec64<T> {
        slice.iter().map(|&x| T::from(x).unwrap()).collect()
    }

    #[test]
    fn test_new_and_with_capacity_u32() {
        let arr: StringArray<u32> = StringArray::default();
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.offsets, offsets(&[0]));
        assert!(arr.data.is_empty());
        assert!(arr.null_mask.is_none());

        let arr: StringArray<u32> = StringArray::with_capacity(10, 64, true);
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.offsets, offsets(&[0]));
        assert!(arr.data.capacity() >= 64);
        assert!(arr.offsets.capacity() >= 11);
        assert!(arr.null_mask.is_some());
        assert!(arr.null_mask.as_ref().unwrap().capacity() >= 10);
    }

    #[test]
    fn test_push_and_get_u32() {
        let mut arr: StringArray<u32> = StringArray::with_capacity(3, 16, false);
        arr.push_str("foo");
        arr.push_str("bar");
        arr.push_str("baz");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some("foo"));
        assert_eq!(arr.get(1), Some("bar"));
        assert_eq!(arr.get(2), Some("baz"));
        assert_eq!(arr.data, Vec64::from(b"foobarbaz" as &[u8]));
        assert!(!arr.is_null(0));
        assert!(!arr.is_null(2));
    }

    #[test]
    fn test_push_and_get_with_null_mask_u32() {
        let mut arr: StringArray<u32> = StringArray::with_capacity(2, 8, true);
        arr.push_str("abc");
        arr.push_null();
        arr.push_str("def");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some("abc"));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(2), Some("def"));
        assert!(!arr.is_null(0));
        assert!(arr.is_null(1));
        assert!(!arr.is_null(2));
        assert_eq!(arr.offsets, offsets(&[0, 3, 3, 6]));
    }

    #[test]
    fn test_push_null_auto_mask_u32() {
        let mut arr: StringArray<u32> = StringArray::default();
        arr.push_str("cat");
        arr.push_null();
        arr.push_str("dog");
        assert_eq!(arr.get(1), None);
        assert!(arr.null_mask.is_some());
        assert_eq!(arr.get(2), Some("dog"));
    }

    #[test]
    fn test_offsets_and_values_alignment_u32() {
        let mut arr: StringArray<u32> = StringArray::default();
        arr.push_str("a");
        arr.push_str("bc");
        arr.push_str("d");
        assert_eq!(arr.offsets, offsets(&[0, 1, 3, 4]));
        assert_eq!(arr.data, Vec64::from(b"abcd" as &[u8]));
    }

    #[test]
    fn test_is_empty_u32() {
        let arr: StringArray<u32> = StringArray::default();
        assert!(arr.is_empty());
        let mut arr: StringArray<u32> = StringArray::default();
        arr.push_str("foo");
        assert!(!arr.is_empty());
    }

    #[test]
    fn test_reserve_u32() {
        let mut arr: StringArray<u32> = StringArray::with_capacity(1, 1, true);
        let old_cap = arr.offsets.capacity();
        arr.reserve(20, 100);
        assert!(arr.offsets.capacity() >= old_cap);
        assert!(arr.data.capacity() >= 100);
        assert!(arr.null_mask.as_ref().unwrap().capacity() >= ((20 + 7) / 8));
    }

    #[test]
    fn test_bulk_push_and_masking_u32() {
        let mut arr: StringArray<u32> = StringArray::with_capacity(4, 16, true);
        arr.push_str("foo");
        arr.push_str("bar");
        arr.push_null();
        arr.push_str("baz");
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some("foo"));
        assert_eq!(arr.get(2), None);
        assert_eq!(arr.get(3), Some("baz"));
        assert!(arr.null_mask.as_ref().is_some());
    }

    #[test]
    fn test_offsets_do_not_grow_too_fast_u32() {
        let mut arr: StringArray<u32> = StringArray::default();
        for _ in 0..100 {
            arr.push_str("x");
        }
        assert_eq!(arr.offsets.len(), 101);
        assert_eq!(arr.data.len(), 100);
        for i in 0..100 {
            assert_eq!(arr.get(i), Some("x"));
        }
    }

    #[test]
    fn test_null_mask_not_present_u32() {
        let mut arr: StringArray<u32> = StringArray::with_capacity(2, 10, false);
        arr.push_str("a");
        arr.push_str("b");
        assert!(arr.null_mask.is_none());
        assert_eq!(arr.get(1), Some("b"));
        assert!(!arr.is_null(1));
    }

    #[test]
    fn test_new_and_with_capacity_u64() {
        let arr: StringArray<u64> = StringArray::default();
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.offsets, offsets(&[0]));
        assert!(arr.data.is_empty());
        assert!(arr.null_mask.is_none());

        let arr: StringArray<u64> = StringArray::with_capacity(10, 64, true);
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.offsets, offsets(&[0]));
        assert!(arr.data.capacity() >= 64);
        assert!(arr.offsets.capacity() >= 11);
        assert!(arr.null_mask.is_some());
    }

    #[test]
    fn test_push_and_get_u64() {
        let mut arr: StringArray<u64> = StringArray::with_capacity(3, 16, false);
        arr.push_str("foo");
        arr.push_str("bar");
        arr.push_str("baz");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some("foo"));
        assert_eq!(arr.get(1), Some("bar"));
        assert_eq!(arr.get(2), Some("baz"));
        assert_eq!(arr.data, Vec64::from(b"foobarbaz" as &[u8]));
        assert!(!arr.is_null(0));
        assert!(!arr.is_null(2));
    }

    #[test]
    fn test_offsets_and_values_alignment_u64() {
        let mut arr: StringArray<u64> = StringArray::default();
        arr.push_str("a");
        arr.push_str("bc");
        arr.push_str("d");
        assert_eq!(arr.offsets, offsets(&[0, 1, 3, 4]));
        assert_eq!(arr.data, Vec64::from(b"abcd" as &[u8]));
    }

    #[test]
    fn test_string_array_slice() {
        let mut arr = StringArray::<u32>::default();
        arr.push_str("apple");
        arr.push_str("banana");
        arr.push_str("cherry");
        arr.push_null();
        arr.push_str("date");

        let sliced = arr.slice_clone(1, 3);
        assert_eq!(sliced.len(), 3);
        assert_eq!(sliced.get(0), Some("banana"));
        assert_eq!(sliced.get(1), Some("cherry"));
        assert_eq!(sliced.get(2), None); // was null
        assert_eq!(sliced.null_count(), 1);
    }

    #[test]
    fn test_to_categorical_array_roundtrip() {
        let strings = vec!["foo", "bar", "foo", "", "bar"];
        let mask = Bitmask::from_bools(&[true, true, true, false, true]);

        let input =
            StringArray::<u32>::from_vec(strings.iter().map(|s| *s).collect(), Some(mask.clone()));

        let cat: CategoricalArray<u32> = input.to_categorical_array();
        let restored = cat.to_string_array();

        for i in 0..input.len() {
            assert_eq!(input.get(i), restored.get(i), "Mismatch at index {}", i);
        }

        assert_eq!(restored.null_mask.unwrap().as_slice(), mask.as_slice());
    }

    #[test]
    fn test_resize_truncate_and_extend() {
        let mut arr = StringArray::<u32>::from_slice(&["a", "bb", "ccc"]);
        arr.resize(2, "ignored".to_string());
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.get(0), Some("a"));
        assert_eq!(arr.get(1), Some("bb"));

        arr.resize(5, "x".to_string());
        assert_eq!(arr.len(), 5);
        assert_eq!(arr.get(2), Some("x"));
        assert_eq!(arr.get(3), Some("x"));
        assert_eq!(arr.get(4), Some("x"));
    }

    #[test]
    fn test_push_nulls_and_mask_updates() {
        let mut arr: StringArray<u32> = StringArray::with_capacity(0, 0, true);
        arr.push_nulls(3);
        assert_eq!(arr.len(), 3);
        assert!(arr.is_null(0));
        assert!(arr.is_null(1));
        assert!(arr.is_null(2));
        assert_eq!(arr.get(0), None);
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(2), None);
        assert_eq!(arr.offsets, offsets(&[0, 0, 0, 0]));
    }

    #[test]
    fn test_resize_edge_cases() {
        let mut arr: StringArray<u32> = StringArray::default();
        arr.resize(0, "abc".to_string());
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.offsets, offsets(&[0]));

        arr.resize(2, "hi".to_string());
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.get(0), Some("hi"));
        assert_eq!(arr.get(1), Some("hi"));
        assert_eq!(arr.offsets, offsets(&[0, 2, 4]));
        assert_eq!(arr.data, Vec64::from(b"hihi" as &[u8]));
    }
}

#[cfg(test)]
#[cfg(feature = "parallel_proc")]
mod parallel_tests {
    use super::*;

    #[test]
    fn test_stringarray_par_iter_no_nulls() {
        let arr = StringArray::<u32>::from_slice(&["foo", "bar", "baz"]);
        let mut got: Vec<&str> = arr.par_iter().collect();
        got.sort();
        assert_eq!(got, vec!["bar", "baz", "foo"]);
    }

    #[test]
    fn test_stringarray_par_iter_opt_with_nulls() {
        let mut arr = StringArray::<u32>::with_capacity(3, 10, true);
        arr.push_str("a");
        arr.push_null();
        arr.push_str("b");
        let mut got: Vec<Option<&str>> = arr.par_iter_opt().collect();
        // Parallel order is not guaranteed.
        got.sort_by_key(|x| x.map(|s| s.to_owned()));
        assert_eq!(got, vec![None, Some("a"), Some("b")]);
    }

    #[test]
    fn test_stringarray_par_iter_with_nulls_yields_empty() {
        let mut arr = StringArray::<u32>::with_capacity(3, 10, true);
        arr.push_str("xx");
        arr.push_null();
        arr.push_str("yy");
        let got: Vec<&str> = arr.par_iter().collect();
        // Should yield ["xx", "", "yy"] in some order
        assert_eq!(got.iter().filter(|&&s| s == "").count(), 1);
        assert!(got.contains(&"xx"));
        assert!(got.contains(&"yy"));
    }

    #[test]
    fn test_stringarray_par_iter_range_unchecked() {
        let arr = StringArray::<u32>::from_slice(&["foo", "bar", "baz", "qux"]);
        let out: Vec<&str> = unsafe { arr.par_iter_range_unchecked(1, 4).collect() };
        assert_eq!(out, vec!["bar", "baz", "qux"]);
    }

    #[test]
    fn test_stringarray_par_iter_range_opt_unchecked() {
        let mut arr = StringArray::<u32>::from_slice(&["a", "b", "c", "d", "e"]);
        // Null mask: only positions 1,3 valid
        arr.null_mask = Some(Bitmask::from_bools(&[false, true, false, true, false]));
        let out: Vec<Option<&str>> = unsafe { arr.par_iter_range_opt_unchecked(1, 5).collect() };
        assert_eq!(
            out,
            vec![
                Some("b"), // 1 (valid)
                None,      // 2 (null)
                Some("d"), // 3 (valid)
                None       // 4 (null)
            ]
        );
    }

    #[test]
    fn test_append_array_stringarray() {
        use crate::traits::masked_array::MaskedArray;
        // Build first array: ["ab", "c"]
        let mut arr1 = StringArray::<u32>::from_slice(&["ab", "c"]);
        // Build second array: ["de", "", "fgh"]
        let mut arr2 = StringArray::<u32>::from_slice(&["de", "", "fgh"]);
        arr2.set_null(1);

        // Validate before append
        assert_eq!(arr1.len(), 2);
        assert_eq!(arr2.len(), 3);
        assert_eq!(arr2.get_str(1), None);
        assert_eq!(arr2.get_str(2), Some("fgh"));

        // Append
        arr1.append_array(&arr2);

        // After append: ["ab", "c", "de", None, "fgh"]
        assert_eq!(arr1.len(), 5);
        let values: Vec<Option<&str>> = (0..5).map(|i| arr1.get_str(i)).collect();
        assert_eq!(
            values,
            vec![Some("ab"), Some("c"), Some("de"), None, Some("fgh"),]
        );

        // Validate offset rebasing (each offset should be increasing, last == data.len())
        let last_offset = arr1.offsets.last().cloned().unwrap();
        assert_eq!(last_offset, arr1.data.len() as u32);

        // Validate null mask
        assert_eq!(arr1.null_count(), 1);
        assert!(!arr1.null_mask.as_ref().unwrap().get(3)); // 3rd appended value is null
        assert!(arr1.null_mask.as_ref().unwrap().get(2));
        assert!(arr1.null_mask.as_ref().unwrap().get(4));
    }
}
