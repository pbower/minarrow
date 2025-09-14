//! # **CategoricalArray Module** - *Mid-Level, Inner Typed Categorical Array*
//!
//! CategoricalArray uses dictionary-encoded strings where each row stores a
//! small integer “code” that references a per-column dictionary of unique strings.
//! This saves memory and accelerates comparisons/joins when many values repeat.
//!
//! ## Interop
//! - Arrow-compatible dictionary layout (`indices` + string `dictionary`), and
//!   round-trips over the Arrow C Data Interface to/from the `Dictionary` array type.
//! - Index width is the generic `T` (e.g., `u8/u16/u32/u64`) and corresponds to
//!   Arrow’s `CategoricalIndexType`.
//!
//! ## Features
//! - Optional `null_mask`: bit-packed, where `1 = valid`, `0 = null`
//! - Builders from raw values (`from_values`, `from_vec64`) and from raw parts.
//! - Iterators over indices and over resolved strings (nullable and non-nullable).
//! - Convert to a dense `StringArray` via `to_string_array()` when needed.
//! - Parallel helpers behind `parallel_proc` feature.
//!
//! ## When to use
//! Use for arrays with repeated strings to reduce memory and speed up operations.

use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::slice::{Iter, IterMut};

#[cfg(feature = "parallel_proc")]
use rayon::iter::ParallelIterator;

use crate::aliases::CategoricalAVT;
use crate::structs::allocator::Alloc64;
use crate::structs::vec64::Vec64;
use crate::traits::shape::Shape;
use crate::enums::shape_dim::ShapeDim;
use crate::traits::type_unions::Integer;
use crate::utils::validate_null_mask_len;
use crate::{
    Bitmask, Buffer, Length, MaskedArray, Offset, StringArray, impl_arc_masked_array,
    impl_array_ref_deref,
};

/// # CategoricalArray
///
/// Categorical array with unique string instances mapped to indices.
///
/// ## Role
/// - Many will prefer the higher level `Array` type, which dispatches to this when
/// necessary.
/// - Can be used as a standalone text array or as the text arm of `TextArray` / `Array`.
///
/// ## Description
/// Compatible with the `Arrow Dictionary` memory layout, where each value is
/// represented as an index into a dictionary of unique strings, and materialises
/// into the format over FFI.
///
/// ### Fields:
/// - `data`: indices buffer referencing entries in `unique_values`.
/// - `unique_values`: dictionary of unique string values.
/// - `null_mask`: optional bit-packed validity bitmap (1=valid, 0=null).
///
/// ## Purpose
/// Consider this when you have a common set of unique string values, and want to
/// save space and increase speed by storing the string values only once
/// *(in the `unique_values` Vec)*, and then only the integers that map to them
/// in the `data` field.
///
/// ## Example
/// ```rust
/// use minarrow::{CategoricalArray, MaskedArray};
///
/// let arr = CategoricalArray::<u8>::from_values(vec!["apple", "banana", "apple", "cherry"]);
/// assert_eq!(arr.len(), 4);
///
/// // Indices into the unique_values dictionary
/// assert_eq!(arr.indices(), &[0u8, 1, 0, 2]);
///
/// // Dictionary of unique values
/// assert_eq!(arr.values(), &["apple".to_string(), "banana".to_string(), "cherry".to_string()]);
///
/// // Resolved value lookups
/// assert_eq!(arr.get_str(0), Some("apple"));
/// assert_eq!(arr.get_str(1), Some("banana"));
/// assert_eq!(arr.get_str(2), Some("apple"));
/// assert_eq!(arr.get_str(3), Some("cherry"));
/// ```
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub struct CategoricalArray<T: Integer> {
    /// Indices buffer (references into the dictionary).
    pub data: Buffer<T>,
    /// Dictionary values (unique values, i.e., Vec64<String>).
    pub unique_values: Vec64<String>,
    /// Optional null mask (bit-packed; 1=valid, 0=null).
    pub null_mask: Option<Bitmask>,
}

impl<T: Integer> CategoricalArray<T> {
    /// Constructs a new CategoricalArray
    #[inline]
    pub fn new(
        data: impl Into<Buffer<T>>,
        unique_values: Vec64<String>,
        null_mask: Option<Bitmask>,
    ) -> Self {
        let data: Buffer<T> = data.into();

        validate_null_mask_len(data.len(), &null_mask);
        for (i, code) in data.iter().enumerate() {
            let idx = code
                .to_usize()
                .unwrap_or_else(|| panic!("Failed to convert code to usize at position {}", i));
            assert!(
                idx < unique_values.len(),
                "Index {} out of bounds for unique_values (len = {}) at position {}",
                idx,
                unique_values.len(),
                i
            );
        }

        Self {
            data: data,
            unique_values,
            null_mask,
        }
    }

    /// Build a categorical column from raw string values, auto-deriving the dictionary.
    #[inline]
    pub fn from_vec64(values: Vec64<&str>, null_mask: Option<Bitmask>) -> Self {
        validate_null_mask_len(values.len(), &null_mask);

        let len = values.len();
        let mut codes = Vec64::with_capacity(len);
        let mut unique_values: Vec64<String> = Vec64::new();
        let mut dict = HashMap::new();

        for (i, s) in values.into_iter().enumerate() {
            // nulls get the default code, but do not participate in the dictionary
            let is_valid = null_mask.as_ref().map_or(true, |m| m.get(i));
            if !is_valid {
                codes.push(T::default());
                continue;
            }

            if let Some(&code) = dict.get(&s) {
                codes.push(code);
            } else {
                let idx = unique_values.len();
                let code = T::try_from(idx).ok().unwrap_or_else(|| {
                    panic!(
                        "Unique category count ({}) exceeds capacity of index type {}",
                        idx + 1,
                        std::any::type_name::<T>()
                    )
                });
                unique_values.push(s.to_string());
                dict.insert(s, code);
                codes.push(code);
            }
        }

        Self {
            data: codes.into(),
            unique_values,
            null_mask,
        }
    }

    /// Vec wrapper
    #[inline]
    pub fn from_vec(values: Vec<&str>, null_mask: Option<Bitmask>) -> Self {
        Self::from_vec64(values.into(), null_mask)
    }

    /// Constructs a new CategoricalArray without validation. The caller must ensure consistency.
    #[inline]
    pub fn new_unchecked(
        data: Vec64<T>,
        unique_values: Vec64<String>,
        null_mask: Option<Bitmask>,
    ) -> Self {
        Self {
            data: data.into(),
            unique_values,
            null_mask,
        }
    }

    /// Constructs a dense DictionaryArray from index and value slices (no nulls).
    #[inline]
    pub fn from_slices(indices: &[T], unique_values: &[String]) -> Self {
        assert!(
            indices.iter().all(|&idx| {
                let i = idx.to_usize();
                i < unique_values.len()
            }),
            "All indices must be valid for unique_values"
        );
        Self {
            data: Vec64(indices.to_vec_in(Alloc64)).into(),
            unique_values: Vec64(unique_values.to_vec_in(Alloc64)).into(),
            null_mask: None,
        }
    }

    /// Returns the current dictionary values as a slice.
    #[inline]
    pub fn values(&self) -> &[String] {
        &self.unique_values
    }

    /// Returns the dictionary indices as a slice.
    ///
    /// Remember, the indices are the data,
    /// because the values are the unique Strings,
    /// in contrast to what a dictionary usually refers to.
    #[inline]
    pub fn indices(&self) -> &[T] {
        &self.data
    }

    /// Returns an iterator of dictionary indices (backing buffer).
    pub fn indices_iter(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    /// Returns an iterator of dictionary values (unique strings).
    pub fn values_iter(&self) -> Iter<'_, String> {
        self.unique_values.iter()
    }

    /// Returns a mutable iterator over indices buffer.
    pub fn indices_iter_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Returns a mutable iterator over dictionary values.
    pub fn values_iter_mut(&mut self) -> IterMut<'_, String> {
        self.unique_values.iter_mut()
    }

    /// Extend with an iterator of &str.
    pub fn extend<'a, I: Iterator<Item = &'a str>>(&mut self, iter: I) {
        for s in iter {
            self.push(s.to_owned());
        }
    }

    /// Append string, adding to dictionary if new. Returns dictionary index used.
    #[inline]
    pub fn push_str(&mut self, value: &str) -> T {
        let dict_idx = match self.unique_values.iter().position(|s| s == value) {
            Some(i) => <T>::from_usize(i),
            None => {
                let i = self.unique_values.len();
                self.unique_values.push(value.to_owned());
                <T>::from_usize(i)
            }
        };
        self.data.push(dict_idx);
        let row = self.len() - 1;
        if let Some(mask) = &mut self.null_mask {
            mask.set(row, true);
        }
        dict_idx
    }

    /// Appends a string without bounds checks, adding to the dictionary if new.
    ///
    /// # Safety
    /// - The caller must ensure `self.data` has sufficient capacity (i.e., already resized).
    /// - `self.null_mask`, if present, must also have space for this index.
    /// - This method assumes exclusive mutable access and no concurrent modification.
    #[inline(always)]
    pub unsafe fn push_str_unchecked(&mut self, value: &str) {
        let idx = self.data.len();
        unsafe { self.set_str_unchecked(idx, value) };
    }

    /// Retrieves the value at the given index, or None if null.
    #[inline]
    pub fn get_str(&self, idx: usize) -> Option<&str> {
        if self.is_null(idx) {
            return None;
        }
        let dict_idx = self.data[idx].to_usize();
        Some(&self.unique_values[dict_idx])
    }

    /// Like `get`, but skips bounds checks.
    #[inline(always)]
    pub unsafe fn get_str_unchecked(&self, idx: usize) -> &str {
        if let Some(mask) = &self.null_mask {
            if !unsafe { mask.get_unchecked(idx) } {
                return "";
            }
        }
        let dict_idx = unsafe { self.data.get_unchecked(idx).to_usize().unwrap() };
        unsafe { &self.unique_values.get_unchecked(dict_idx) }
    }

    /// Sets the value at `idx`. Marks as valid.
    #[inline]
    pub fn set_str(&mut self, idx: usize, value: &str) {
        assert!(idx < self.data.len(), "index out of bounds");

        let dict_idx = if let Some(pos) = self.unique_values.iter().position(|s| s == value) {
            T::from_usize(pos)
        } else {
            let i = self.unique_values.len();
            self.unique_values.push(value.to_owned());
            T::from_usize(i)
        };

        self.data[idx] = dict_idx;

        if let Some(mask) = &mut self.null_mask {
            mask.set(idx, true);
        } else {
            let mut m = Bitmask::new_set_all(self.data.len(), false);
            m.set(idx, true);
            self.null_mask = Some(m);
        }
    }

    /// Like `set`, but skips all bounds checks.
    #[inline(always)]
    pub unsafe fn set_str_unchecked(&mut self, idx: usize, value: &str) {
        // find or insert
        let pos = self.unique_values.iter().position(|s| s == value);
        let code = if let Some(p) = pos {
            T::from_usize(p)
        } else {
            let new_i = self.unique_values.len();
            self.unique_values.push(value.to_owned());
            T::from_usize(new_i)
        };
        // write code and mark non-null
        let data = self.data.as_mut_slice();
        data[idx] = code;
        if let Some(mask) = &mut self.null_mask {
            mask.set(idx, true);
        } else {
            let mut m = Bitmask::new_set_all(self.len(), false);
            m.set(idx, true);
            self.null_mask = Some(m);
        }
    }

    /// Returns an iterator of &str (nulls yielded as empty string).
    #[inline]
    pub fn iter_str(&self) -> impl Iterator<Item = &str> + '_ {
        self.data.iter().enumerate().map(move |(idx, &dict_idx)| {
            if self.is_null(idx) {
                ""
            } else {
                &self.unique_values[dict_idx.to_usize()]
            }
        })
    }

    /// Returns an iterator of Option<&str>, None if value is null.
    #[inline]
    pub fn iter_str_opt(&self) -> impl Iterator<Item = Option<&str>> + '_ {
        self.data.iter().enumerate().map(move |(idx, &dict_idx)| {
            if self.is_null(idx) {
                None
            } else {
                Some(unsafe {
                    std::mem::transmute::<&str, &'static str>(
                        &self.unique_values[dict_idx.to_usize()],
                    )
                })
            }
        })
    }

    /// Returns an iterator of `&str` values (nulls yield `""`) for a specified range.
    #[inline]
    pub fn iter_str_range(&self, offset: usize, len: usize) -> impl Iterator<Item = &str> + '_ {
        self.data[offset..offset + len]
            .iter()
            .enumerate()
            .map(move |(i, &dict_idx)| {
                let idx = offset + i;
                if self.is_null(idx) {
                    ""
                } else {
                    &self.unique_values[dict_idx.to_usize()]
                }
            })
    }

    /// Returns an iterator of `Option<&str>` values for a specified range.
    #[inline]
    pub fn iter_str_opt_range(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = Option<&str>> + '_ {
        self.data[offset..offset + len]
            .iter()
            .enumerate()
            .map(move |(i, &dict_idx)| {
                let idx = offset + i;
                if self.is_null(idx) {
                    None
                } else {
                    Some(unsafe {
                        std::mem::transmute::<&str, &'static str>(
                            &self.unique_values[dict_idx.to_usize()],
                        )
                    })
                }
            })
    }

    /// Build from an iterator of &str in one pass.
    pub fn from_values<'a, I: IntoIterator<Item = &'a str>>(iter: I) -> Self {
        use std::collections::HashMap;
        let mut dict = Vec64::<String>::new();
        let mut map = HashMap::<&str, usize>::new();
        let mut idx_buf = Vec64::<T>::new();

        for s in iter {
            let pos = *map.entry(s).or_insert_with(|| {
                let i = dict.len();
                dict.push(s.to_owned());
                i
            });
            idx_buf.push(<T>::from_usize(pos));
        }

        Self {
            data: idx_buf.into(),
            unique_values: dict.into(),
            null_mask: None,
        }
    }

    /// Create from raw buffers (indices & dictionary) without copying.
    #[inline]
    pub fn from_parts(
        indices: Vec64<T>,
        unique_values: Vec64<String>,
        null_mask: Option<Bitmask>,
    ) -> Self {
        Self {
            data: indices.into(),
            unique_values: unique_values.into(),
            null_mask,
        }
    }

    /// Materialise the categorical as a dense StringArray<T>.
    #[inline]
    pub fn to_string_array(&self) -> StringArray<T> {
        let len = self.data.len();
        let mut offsets = Vec64::with_capacity(len + 1);
        let mut data = Vec64::<u8>::new();
        offsets.push(T::zero());

        for i in 0..len {
            if self.is_null(i) {
                offsets.push(T::from(data.len()).unwrap());
            } else {
                let dict_idx = self.data[i].to_usize();
                let s = &self.unique_values[dict_idx];
                data.extend_from_slice(s.as_bytes());
                offsets.push(T::from(data.len()).unwrap());
            }
        }

        StringArray {
            offsets: offsets.into(),
            data: data.into(),
            null_mask: self.null_mask.clone(),
        }
    }
}

impl<T: Integer> MaskedArray for CategoricalArray<T> {
    type T = T;

    type Container = Buffer<T>;

    type LogicalType = String;

    type CopyType = &'static str;

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    fn data(&self) -> &Self::Container {
        &self.data
    }

    fn data_mut(&mut self) -> &mut Self::Container {
        &mut self.data
    }

    /// Retrieves the value at the given index, or `None` if null.
    ///
    /// # ⚠️ WARNING - prefer `get_str`
    /// This method returns a `&static str` for trait compatibility. However, the returned
    /// reference **borrows from the backing buffer of the array** and must not outlive
    /// the lifetime of `self`. It is **not truly static**.
    ///
    /// *Instead, prefer `get_str` for practical use*, or, if you
    /// are using this to build on top of the trait, ensure that you *do not store* the values.
    ///
    /// # Panics
    /// Panics if `idx >= self.len()` or if `data[idx]` is an invalid index into `unique_values`.
    #[inline]
    fn get(&self, idx: usize) -> Option<Self::CopyType> {
        if self.is_null(idx) {
            return None;
        }

        let dict_idx = self.data[idx].to_usize();

        // SAFETY: Must not escape beyond the borrow lifetime of self.
        Some(unsafe { std::mem::transmute::<&str, &'static str>(&self.unique_values[dict_idx]) })
    }

    /// Sets the value at `idx`. Marks as valid.
    ///
    /// # ⚠️ Prefer `set_str`, which avoids a reallocation.
    #[inline]
    fn set(&mut self, idx: usize, value: Self::LogicalType) {
        self.set_str(idx, &value)
    }

    /// Like `get`, but skips bounds checks on both the data and dictionary index.
    ///
    /// # ⚠️ WARNING - prefer `get_str_unchecked`
    /// This method returns a `&static str` for trait compatibility. However, the returned
    /// reference **borrows from the backing buffer of the array** and must not outlive
    /// the lifetime of `self`. It is **not truly static**.
    ///
    /// *Instead, prefer `get_str_unchecked` for practical use*, or, if you
    /// are using this to build on top of the trait, ensure that you *do not store* the values.
    ///
    /// # Safety
    /// Caller must ensure:
    /// - `idx` is within bounds of `self.data`
    /// - `self.data[idx]` yields a valid index into `self.unique_values`
    /// - The result is not held beyond the lifetime of `self`
    ///
    /// The transmute casts the internal `&str` to `'static`, but this is only valid
    /// as long as `self` is alive. Do **not** persist the reference.
    #[inline]
    unsafe fn get_unchecked(&self, idx: usize) -> Option<Self::CopyType> {
        if let Some(mask) = &self.null_mask {
            if !mask.get(idx) {
                return None;
            }
        }

        let dict_idx = unsafe { self.data.get_unchecked(idx).to_usize().unwrap() };
        Some(unsafe {
            std::mem::transmute::<&str, &'static str>(self.unique_values.get_unchecked(dict_idx))
        })
    }

    /// Like `set`, but skips all bounds checks.
    ///
    /// ⚠️ Prefer `set_str_unchecked` as it avoids a reallocation.
    #[inline]
    unsafe fn set_unchecked(&mut self, idx: usize, value: Self::LogicalType) {
        // find or insert
        let pos = self.unique_values.iter().position(|s| *s == value);
        let code = if let Some(p) = pos {
            T::from_usize(p)
        } else {
            let new_i = self.unique_values.len();
            self.unique_values.push(value.to_owned());
            T::from_usize(new_i)
        };
        // write code and mark non-null
        let data = self.data.as_mut_slice();
        data[idx] = code;
        if let Some(mask) = &mut self.null_mask {
            mask.set(idx, true);
        } else {
            let mut m = Bitmask::new_set_all(self.len(), false);
            m.set(idx, true);
            self.null_mask = Some(m);
        }
    }

    /// Returns an iterator of `&'static str` values.
    ///
    /// # ⚠️ WARNING - prefer `iter_str`
    /// This method returns a `&static str` for trait compatibility. However, the returned
    /// reference **borrows from the backing buffer of the array** and must not outlive
    /// the lifetime of `self`. It is **not truly static**.
    ///
    /// *Instead, prefer `iter_str` for practical use*, or, if you
    /// are using this to build on top of the trait, ensure that you *do not store* the values.
    ///
    /// Nulls are represented as an empty string `""`.
    #[inline]
    fn iter(&self) -> impl Iterator<Item = Self::CopyType> + '_ {
        self.data.iter().enumerate().map(move |(idx, &dict_idx)| {
            if self.is_null(idx) {
                ""
            } else {
                unsafe {
                    std::mem::transmute::<&str, &'static str>(
                        &self.unique_values[dict_idx.to_usize()],
                    )
                }
            }
        })
    }

    /// Returns an iterator over `Option<&'static str>`, yielding `None` for nulls.
    ///
    /// # ⚠️ WARNING - prefer `iter_str_opt`
    /// This method returns a `&static str` for trait compatibility. However, the returned
    /// reference **borrows from the backing buffer of the array** and must not outlive
    /// the lifetime of `self`. It is **not truly static**.
    ///
    /// *Instead, prefer `iter_str_opt` for practical use*, or, if you
    /// are using this to build on top of the trait, ensure that you *do not store* the values.
    #[inline]
    fn iter_opt(&self) -> impl Iterator<Item = Option<Self::CopyType>> + '_ {
        self.data.iter().enumerate().map(move |(idx, &dict_idx)| {
            if self.is_null(idx) {
                None
            } else {
                Some(unsafe {
                    std::mem::transmute::<&str, &'static str>(
                        &self.unique_values[dict_idx.to_usize()],
                    )
                })
            }
        })
    }

    /// Returns an iterator of `&'static str` values for a specified range.
    ///
    /// ⚠️ WARNING - prefer `iter_str_range`
    /// The returned references borrow from the backing buffer and are not truly static.
    #[inline]
    fn iter_range(&self, offset: usize, len: usize) -> impl Iterator<Item = &'static str> + '_ {
        self.data[offset..offset + len]
            .iter()
            .enumerate()
            .map(move |(i, &dict_idx)| {
                let idx = offset + i;
                if self.is_null(idx) {
                    ""
                } else {
                    unsafe {
                        std::mem::transmute::<&str, &'static str>(
                            &self.unique_values[dict_idx.to_usize()],
                        )
                    }
                }
            })
    }

    /// Returns an iterator over `Option<&'static str>` values for a specified range.
    ///
    /// ⚠️ WARNING - prefer `iter_str_opt_range`
    /// The returned references borrow from the backing buffer and are not truly static.
    #[inline]
    fn iter_opt_range(
        &self,
        offset: usize,
        len: usize,
    ) -> impl Iterator<Item = Option<&'static str>> + '_ {
        self.data[offset..offset + len]
            .iter()
            .enumerate()
            .map(move |(i, &dict_idx)| {
                let idx = offset + i;
                if self.is_null(idx) {
                    None
                } else {
                    Some(unsafe {
                        std::mem::transmute::<&str, &'static str>(
                            &self.unique_values[dict_idx.to_usize()],
                        )
                    })
                }
            })
    }

    /// Append string, adding to dictionary if new.
    ///
    /// ⚠️ Prefer `push_str` as it avoids a reallocation.
    #[inline]
    fn push(&mut self, value: Self::LogicalType) {
        self.push_str(&value);
    }

    /// Append string, adding to dictionary if new, without bounds checking
    ///
    /// ⚠️ Prefer `push_str_unchecked` as it avoids a reallocation.
    ///
    /// # Safety
    /// - The caller must ensure `self.data` has sufficient capacity (i.e., already resized).
    /// - `self.null_mask`, if present, must also have space for this index.
    /// - This method assumes exclusive mutable access and no concurrent modification.
    #[inline]
    unsafe fn push_unchecked(&mut self, value: Self::LogicalType) {
        self.push_str(&value);
    }

    /// Returns a logical slice of the categorical array [offset, offset+len)
    /// as a new `CategoricalArray` object.
    ///
    /// For a non-copy slice view, use `slice` from the parent Array object
    fn slice_clone(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= self.data.len(),
            "slice window out of bounds"
        );

        let data = self.data[offset..offset + len].to_vec_in(Alloc64);
        let null_mask = self
            .null_mask
            .as_ref()
            .map(|nm| nm.slice_clone(offset, len));
        Self {
            data: Vec64(data).into(),
            unique_values: self.unique_values.clone(),
            null_mask,
        }
    }

    /// Borrows a `CategoricalArray` with its window parameters
    /// to a `CategoricalArrayView<'a>` alias. Like a slice, but
    /// retains access to the `&CategoricalArray`.
    ///
    /// `Offset` and `Length` are `usize` aliases.
    #[inline(always)]
    fn tuple_ref<'a>(&'a self, offset: Offset, len: Length) -> CategoricalAVT<'a, T> {
        (&self, offset, len)
    }

    /// Returns the total number of nulls.
    fn null_count(&self) -> usize {
        self.null_mask
            .as_ref()
            .map(|m| m.count_zeros())
            .unwrap_or(0)
    }

    /// Resizes the data in-place so that `len` is equal to `new_len`.
    fn resize(&mut self, n: usize, value: Self::LogicalType) {
        let current_len = self.len();

        // Temporary index map to avoid duplicate dictionary search
        let mut index_map: HashMap<String, u32> = self
            .unique_values
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        let code = intern(&value, &mut index_map, &mut self.unique_values);
        let encoded = T::from_usize(code as usize);

        if n > current_len {
            self.data.reserve(n - current_len);
            for _ in current_len..n {
                self.data.push(encoded);
            }
        } else if n < current_len {
            self.data.truncate(n);
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

    /// Sets the bitmask from a supplied one or `None`
    fn set_null_mask(&mut self, mask: Option<Bitmask>) {
        self.null_mask = mask
    }

    /// Appends all values (and null mask if present) from `other` to `self`.
    fn append_array(&mut self, other: &Self) {
        let orig_len = self.len();
        let other_len = other.len();

        if other_len == 0 {
            return;
        }

        // Append data
        self.data_mut().extend_from_slice(other.data());

        // Handle null masks
        match (self.null_mask_mut(), other.null_mask()) {
            (Some(self_mask), Some(other_mask)) => {
                self_mask.extend_from_bitmask(other_mask);
            }
            (Some(self_mask), None) => {
                // Mark all appended as valid.
                self_mask.resize(orig_len + other_len, true);
            }
            (None, Some(other_mask)) => {
                // Materialise new null mask for self, all existing valid.
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

    /// Extends the categorical array from an iterator with pre-allocated capacity.
    /// Reserves capacity in the underlying index buffer to avoid reallocations
    /// during bulk insertion. Dictionary is expanded as new unique values are encountered.
    fn extend_from_iter_with_capacity<I>(&mut self, iter: I, additional_capacity: usize)
    where
        I: Iterator<Item = Self::LogicalType>,
    {
        self.data.reserve(additional_capacity);
        let values: Vec<Self::LogicalType> = iter.collect();
        let start_len = self.data.len();
        // Extend the length to accommodate new elements
        self.data.resize(start_len + values.len(), T::from_usize(0));
        // Extend null mask if it exists
        if let Some(mask) = &mut self.null_mask {
            mask.resize(start_len + values.len(), true);
        }
        // Now use unchecked operations since we have proper length
        for (i, value) in values.iter().enumerate() {
            let dict_idx = match self.unique_values.iter().position(|s| s == &value.to_string()) {
                Some(idx) => T::from_usize(idx),
                None => {
                    let idx = self.unique_values.len();
                    self.unique_values.push(value.to_string());
                    T::from_usize(idx)
                }
            };
            {
                let data = self.data.as_mut_slice();
                data[start_len + i] = dict_idx;
            }
            if let Some(mask) = &mut self.null_mask {
                unsafe { mask.set_unchecked(start_len + i, true) };
            }
        }
    }

    /// Extends the categorical array from a slice of string values.
    /// Pre-allocates capacity for the index buffer and efficiently processes
    /// each string through the internal dictionary for optimal categorical encoding.
    fn extend_from_slice(&mut self, slice: &[Self::LogicalType]) {
        let start_len = self.data.len();
        self.data.reserve(slice.len());
        // Extend the length to accommodate new elements
        self.data.resize(start_len + slice.len(), T::from_usize(0));
        // Extend null mask if it exists
        if let Some(mask) = &mut self.null_mask {
            mask.resize(start_len + slice.len(), true);
        }
        // Now use unchecked operations since we have proper length
        for (i, value) in slice.iter().enumerate() {
            let dict_idx = match self.unique_values.iter().position(|s| s == &value.to_string()) {
                Some(idx) => T::from_usize(idx),
                None => {
                    let idx = self.unique_values.len();
                    self.unique_values.push(value.to_string());
                    T::from_usize(idx)
                }
            };
            {
                let data = self.data.as_mut_slice();
                data[start_len + i] = dict_idx;
            }
            if let Some(mask) = &mut self.null_mask {
                unsafe { mask.set_unchecked(start_len + i, true) };
            }
        }
    }

    /// Creates a new categorical array filled with the specified string repeated `count` times.
    /// The dictionary will contain only one unique value, making this highly memory-efficient 
    /// for repeated categorical values.
    fn fill(value: Self::LogicalType, count: usize) -> Self {
        let mut array = CategoricalArray::<T>::from_vec64(crate::Vec64::with_capacity(count), None);
        // Extend the length to accommodate new elements
        array.data.resize(count, T::from_usize(0));
        // Get or add the dictionary entry once
        array.unique_values.push(value.to_string());
        let dict_index = T::from_usize(0);
        // Now use unchecked operations since we have proper length
        for i in 0..count {
            {
                let data = array.data.as_mut_slice();
                data[i] = dict_index;
            }
        }
        array
    }
}

#[cfg(feature = "parallel_proc")]
impl<T: Integer + Send + Sync> CategoricalArray<T> {
    /// Parallel iterator over &str (null yields "").
    #[inline]
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, T> {
        self.data.par_iter()
    }

    /// Parallel mut iterator over &str (null yields "").
    #[inline]
    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, T> {
        self.data.par_iter_mut()
    }

    /// Parallel iterator over Option<&str> (None if null).
    #[inline]
    pub fn par_iter_opt(&self) -> impl ParallelIterator<Item = Option<&str>> + '_ {
        self.par_iter_range_opt(0, self.len())
    }

    /// `[start,end)` → `&str` (null ⇒ `""`)
    #[inline]
    pub fn par_iter_range(
        &self,
        start: usize,
        end: usize,
    ) -> impl ParallelIterator<Item = &str> + '_ {
        use rayon::prelude::*;
        let null_mask = self.null_mask.as_ref();
        let dict = &self.unique_values;
        let idx_buf = &self.data;
        debug_assert!(start <= end && end <= idx_buf.len());
        (start..end).into_par_iter().map(move |i| {
            if null_mask.map(|m| !m.get(i)).unwrap_or(false) {
                ""
            } else {
                &dict[idx_buf[i].to_usize()]
            }
        })
    }

    // `[start,end)` → `Option<&str>`
    #[inline]
    pub fn par_iter_range_opt(
        &self,
        start: usize,
        end: usize,
    ) -> impl ParallelIterator<Item = Option<&str>> + '_ {
        use rayon::prelude::*;
        let null_mask = self.null_mask.as_ref();
        let dict = &self.unique_values;
        let idx_buf = &self.data;
        debug_assert!(start <= end && end <= idx_buf.len());
        (start..end).into_par_iter().map(move |i| {
            if null_mask.map(|m| !m.get(i)).unwrap_or(false) {
                None
            } else {
                Some(dict[idx_buf[i].to_usize()].as_str())
            }
        })
    }

    /// `[start,end)` → `&str` (null ⇒ `""`) — no bounds checks
    #[inline]
    pub fn par_iter_range_unchecked(
        &self,
        start: usize,
        end: usize,
    ) -> impl rayon::prelude::ParallelIterator<Item = &str> + '_ {
        use rayon::prelude::*;
        let null_mask = self.null_mask.as_ref();
        let dict = &self.unique_values;
        let idx_buf = &self.data;
        (start..end).into_par_iter().map(move |i| {
            if let Some(mask) = null_mask {
                if !unsafe { mask.get_unchecked(i) } {
                    return "";
                }
            }
            let idx = unsafe { *idx_buf.get_unchecked(i) }.to_usize();
            unsafe { dict.get_unchecked(idx).as_str() }
        })
    }

    /// `[start,end)` → `Option<&str>` —  no bounds checks
    #[inline]
    pub fn par_iter_range_opt_unchecked(
        &self,
        start: usize,
        end: usize,
    ) -> impl rayon::prelude::ParallelIterator<Item = Option<&str>> + '_ {
        use rayon::prelude::*;
        let null_mask = self.null_mask.as_ref();
        let dict = &self.unique_values;
        let idx_buf = &self.data;
        (start..end).into_par_iter().map(move |i| {
            if let Some(mask) = null_mask {
                if !unsafe { mask.get_unchecked(i) } {
                    return None;
                }
            }
            let idx = unsafe { *idx_buf.get_unchecked(i) }.to_usize();
            Some(unsafe { dict.get_unchecked(idx).as_str() })
        })
    }
}

impl<T: Integer> Shape for CategoricalArray<T> {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

// Intern for building the dictionary
#[inline(always)]
fn intern(s: &str, dict: &mut HashMap<String, u32>, uniq: &mut Vec64<String>) -> u32 {
    if let Some(&code) = dict.get(s) {
        code
    } else {
        let idx = uniq.len() as u32;
        uniq.push(s.to_owned());
        dict.insert(s.to_owned(), idx);
        idx
    }
}

impl_arc_masked_array!(
    Inner = CategoricalArray<T>,
    T = T,
    Container = Buffer<T>,
    LogicalType = String,
    CopyType = &'static str,
    BufferT = T,
    Variant = TextArray,
    Bound = Integer,
);

impl_array_ref_deref!(CategoricalArray<T>: Integer);

impl<T> Display for CategoricalArray<T>
where
    T: Integer + std::fmt::Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let len = self.len();
        let null_count = self.null_count();
        let dict_size = self.unique_values.len();

        writeln!(
            f,
            "CategoricalArray [{} values]s] (dtype: categorical[str], nulls: {}, dictionary size: {})",
            len, null_count, dict_size
        )?;

        const MAX_PREVIEW: usize = 25;
        write!(f, "[")?;
        for i in 0..usize::min(len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, ", ")?;
            }
            match self.get(i) {
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
    use crate::traits::masked_array::MaskedArray;
    use crate::vec64;

    fn bm(bits: &[bool]) -> Bitmask {
        let mut m = Bitmask::new_set_all(bits.len(), false);
        for (i, &b) in bits.iter().enumerate() {
            m.set(i, b);
        }
        m
    }

    #[test]
    fn empty_new() {
        let arr = CategoricalArray::<u8>::default();
        assert!(arr.is_empty());
        assert!(arr.values().is_empty());
    }
    #[test]
    fn push_and_get() {
        let mut arr = CategoricalArray::<u8>::default();
        let i1 = arr.push_str("hello");
        let i2 = arr.push_str("world");
        let i3 = arr.push_str("hello");
        assert_eq!(i1, 0);
        assert_eq!(i2, 1);
        assert_eq!(i3, 0);
        assert_eq!(arr.indices(), &[0u8, 1, 0]);
        assert_eq!(arr.values(), &["hello", "world".into()]);
        assert_eq!(arr.get(1), Some("world"));
    }

    #[test]
    fn null_handling() {
        let mut arr = CategoricalArray::<u16>::default();
        arr.push_str("a");
        arr.push_null();
        arr.push_str("b");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some("a"));
        assert_eq!(arr.get(1), None);
        assert!(arr.is_null(1));
        assert_eq!(arr.get(2), Some("b"));
    }

    #[test]
    fn set_overwrite_and_new() {
        let mut arr = CategoricalArray::<u32>::default();
        arr.push_str("x");
        arr.push_str("y");
        arr.set_str(1, "x");
        assert_eq!(arr.get(1), Some("x"));
        arr.set_str(0, "zebra");
        assert!(arr.values().contains(&"zebra".to_string()));
        assert_eq!(arr.get(0), Some("zebra"));
    }

    #[test]
    fn extend_and_builder() {
        let mut arr = CategoricalArray::<u8>::default();
        arr.extend(["a", "b", "a", "c"].iter().copied());
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(2), Some("a"));

        let built = CategoricalArray::<u8>::from_values(vec!["k", "l", "k"]);
        assert_eq!(built.indices(), &[0u8, 1, 0]);
        assert_eq!(built.get(1), Some("l"));
    }

    #[test]
    fn set_null_after_push() {
        let mut arr = CategoricalArray::<u8>::default();
        arr.push_str("one");
        arr.push_str("two");
        arr.set_null(1);
        assert!(arr.is_null(1));
        assert_eq!(arr.get(1), None);
    }

    #[test]
    fn test_categorical_iter() {
        let arr =
            CategoricalArray::from_slices(&[0u32, 1, 2], &["a".into(), "b".into(), "c".into()]);
        let vals: Vec<_> = arr.iter().collect();
        assert_eq!(vals, vec!["a", "b", "c"]);
        let opt: Vec<_> = arr.iter_str_opt().collect();
        assert_eq!(opt, vec![Some("a"), Some("b"), Some("c")]);
    }

    #[test]
    fn test_categorical_array_slice() {
        let mut arr = CategoricalArray::<u8>::default();
        arr.data.extend_from_slice(&[2, 1, 0]);
        arr.unique_values.extend_from_slice(&[
            "green".to_string(),
            "blue".to_string(),
            "red".to_string(),
        ]);
        arr.null_mask = Some(Bitmask::from_bools(&[false, true, true]));
        let sliced = arr.slice_clone(0, 3);
        assert_eq!(
            sliced.iter_str_opt().collect::<Vec<_>>(),
            vec![None, Some("blue"), Some("green")]
        );
    }

    #[test]
    fn test_categorical_set_and_get() {
        let mut arr = CategoricalArray::<u32>::from_values(["a", "b", "c"].iter().cloned());
        // initial null mask none => all valid
        assert!(arr.null_mask.is_none());

        // set index 1 to "d" (new entry)
        arr.set_str(1, "d");
        assert_eq!(arr.get(1), Some("d"));
        // dictionary should have "d" appended
        assert_eq!(arr.unique_values.len(), 4);
        assert!(arr.unique_values.contains(&"d".to_string()));

        // set index 2 to existing "a"
        arr.set_str(2, "a");
        assert_eq!(arr.get(2), Some("a"));
        // dictionary length unchanged
        assert_eq!(arr.unique_values.len(), 4);
    }

    #[test]
    fn test_categorical_set_unchecked_and_null_mask() {
        let mut arr = CategoricalArray::<u32>::from_values(["x", "y", "z"].iter().cloned());
        arr.null_mask = Some(bm(&[true, false, true]));

        // unsafe unchecked set index 1 to "w"
        unsafe { arr.set_str_unchecked(1, "w") };
        // now index 1 should be "w"
        assert_eq!(arr.get(1), Some("w"));
        // null mask at 1 now true
        let mask = arr.null_mask.as_ref().unwrap();
        assert!(mask.get(1));
        // dictionary should contain "w"
        assert!(arr.unique_values.contains(&"w".to_string()));
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_categorical_set_oob() {
        let mut arr = CategoricalArray::<u32>::from_values(["foo"].iter().cloned());
        // this should panic
        arr.set_str(5, "bar");
    }

    #[test]
    fn test_to_string_array() {
        let unique = vec64!["foo".to_string(), "bar".to_string()];
        let data = vec64![0u32, 0u32, 1u32];
        let mut mask = Bitmask::new_set_all(3, true);
        mask.set(1, false); // second entry is null

        let cat = CategoricalArray {
            data: data.into(),
            unique_values: unique,
            null_mask: Some(mask),
        };

        let str_arr = cat.to_string_array();

        assert_eq!(str_arr.get(0), Some("foo"));
        assert_eq!(str_arr.get(1), None);
        assert_eq!(str_arr.get(2), Some("bar"));

        assert_eq!(str_arr.offsets, vec64![0u32, 3, 3, 6]);
        assert_eq!(str_arr.data, Vec64::from_slice(b"foobar"));
        assert_eq!(str_arr.null_mask.unwrap().count_zeros(), 1);
    }

    #[test]
    fn test_iterators_yield_correct_values() {
        let mut arr = CategoricalArray::<u8>::default();
        arr.push_str("cat");
        arr.push_str("dog");
        arr.push_str("bird");

        let mut it = arr.indices_iter();
        assert_eq!(it.next(), Some(&0u8));
        assert_eq!(it.next(), Some(&1u8));

        let mut it = arr.values_iter();
        assert!(it.any(|s| s == "cat"));
        assert!(it.any(|s| s == "dog"));

        let mut it_mut = arr.indices_iter_mut();
        if let Some(v) = it_mut.next() {
            *v = 2;
        }
        assert_eq!(arr.get(0), Some("bird"));
    }

    #[test]
    fn test_resize_expands_and_truncates() {
        let mut arr = CategoricalArray::<u8>::default();
        arr.push_str("one");
        arr.push_str("two");

        arr.resize(5, "two".to_string());
        assert_eq!(arr.len(), 5);
        assert_eq!(arr.get(4), Some("two"));

        arr.resize(2, "ignored".to_string());
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_from_parts_exact_match() {
        let data = vec64![0u8, 1u8];
        let dict = vec64!["alpha".to_string(), "beta".to_string()];
        let mask = Some(Bitmask::from_bools(&[true, false]));
        let arr = CategoricalArray::from_parts(data, dict, mask.clone());

        assert_eq!(arr.get(0), Some("alpha"));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.null_mask(), mask.as_ref());
    }

    #[test]
    fn test_batch_extend_from_iter_with_capacity() {
        let mut arr = CategoricalArray::<u32>::default();
        let data = vec!["cat".to_string(), "dog".to_string(), "cat".to_string(), "bird".to_string()];
        
        arr.extend_from_iter_with_capacity(data.into_iter(), 4);
        
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some("cat"));
        assert_eq!(arr.get(1), Some("dog"));
        assert_eq!(arr.get(2), Some("cat"));
        assert_eq!(arr.get(3), Some("bird"));
        
        // Dictionary should have 3 unique values
        assert_eq!(arr.unique_values.len(), 3);
    }

    #[test]
    fn test_batch_extend_from_slice_dictionary_growth() {
        let mut arr = CategoricalArray::<u32>::default();
        arr.push("initial".to_string());
        
        let data = &["apple".to_string(), "banana".to_string(), "apple".to_string()];
        arr.extend_from_slice(data);
        
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some("initial"));
        assert_eq!(arr.get(1), Some("apple"));
        assert_eq!(arr.get(2), Some("banana"));
        assert_eq!(arr.get(3), Some("apple"));
        
        // Dictionary: initial, apple, banana
        assert_eq!(arr.unique_values.len(), 3);
    }

    #[test]
    fn test_batch_fill_single_category() {
        let arr = CategoricalArray::<u32>::fill("repeated".to_string(), 100);
        
        assert_eq!(arr.len(), 100);
        assert_eq!(arr.null_count(), 0);
        
        // All values should be the same category
        for i in 0..100 {
            assert_eq!(arr.get(i), Some("repeated"));
        }
        
        // Dictionary should contain only one unique value
        assert_eq!(arr.unique_values.len(), 1);
        assert_eq!(arr.unique_values[0], "repeated");
        
        // All indices should point to the same dictionary entry (0)
        for i in 0..100 {
            assert_eq!(arr.data[i], 0u32);
        }
    }

    #[test]
    fn test_batch_operations_with_nulls() {
        let mut arr = CategoricalArray::<u32>::default();
        arr.push("first".to_string());
        arr.push_null();
        
        let data = &["second".to_string(), "first".to_string()];
        arr.extend_from_slice(data);
        
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some("first"));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(2), Some("second"));
        assert_eq!(arr.get(3), Some("first"));
        assert!(arr.null_count() >= 1); // At least the initial null
        
        // Dictionary: first, second  
        assert!(arr.unique_values.len() >= 2); // At least first and second
    }

    #[test]
    fn test_batch_operations_preserve_categorical_efficiency() {
        let mut arr = CategoricalArray::<u32>::default();
        
        // Create data with many repeated categories
        let categories = ["A", "B", "C"];
        let mut data = Vec::new();
        for _ in 0..100 {
            for cat in &categories {
                data.push(cat.to_string());
            }
        }
        
        arr.extend_from_slice(&data);
        
        assert_eq!(arr.len(), 300);
        assert_eq!(arr.unique_values.len(), 3); // Only 3 unique despite 300 entries
        
        // Verify all categories are represented correctly
        for i in 0..300 {
            let expected = categories[i % 3];
            assert_eq!(arr.get(i), Some(expected));
        }
    }
}

#[cfg(test)]
#[cfg(feature = "parallel_proc")]
mod parallel_tests {
    use super::*;
    use crate::vec64;
    #[test]
    fn test_categorical_par_iter() {
        let arr =
            CategoricalArray::from_slices(&[0u32, 1, 2], &["a".into(), "b".into(), "c".into()]);
        let vals: Vec<_> = arr.par_iter().collect();
        assert_eq!(vals.len(), 3);
        let opt: Vec<_> = arr.par_iter_opt().collect();
        assert!(opt.iter().all(|v| v.is_some()));
    }

    #[test]
    fn test_categoricalarray_par_iter_opt() {
        let mut arr = CategoricalArray::<u32>::default();
        arr.push_str("alpha");
        arr.push_str("beta");
        arr.push_null();
        arr.push_str("gamma");

        let par: Vec<_> = arr.par_iter_opt().collect();
        let expected = vec![Some("alpha"), Some("beta"), None, Some("gamma")];
        assert_eq!(par, expected);
    }

    #[test]
    fn test_categoricalarray_par_iter_range_unchecked() {
        let dict = vec64!["one".to_string(), "two".to_string(), "three".to_string()];
        let arr = CategoricalArray::<u32>::from_parts(vec64![0, 2, 1, 0, 2], dict, None);
        let out: Vec<&str> = arr.par_iter_range_unchecked(1, 4).collect();
        assert_eq!(out, vec!["three", "two", "one"]);
    }

    #[test]
    fn test_categoricalarray_par_iter_range_opt_unchecked() {
        let dict = vec64!["x".to_string(), "y".to_string(), "z".to_string()];
        let mut arr = CategoricalArray::<u32>::from_parts(vec64![1, 0, 2, 1, 0], dict, None);
        arr.null_mask = Some(Bitmask::from_bools(&[true, false, true, false, true]));
        let out: Vec<Option<&str>> = arr.par_iter_range_opt_unchecked(0, 5).collect();
        assert_eq!(
            out,
            vec![
                Some("y"), // 0 (valid)
                None,      // 1 (null)
                Some("z"), // 2 (valid)
                None,      // 3 (null)
                Some("x")  // 4 (valid)
            ]
        );
    }
}
