//! # TemporalArrayView Module - *Windowed View over a TemporalArray*
//!
//! `TemporalArrayV` is a **read-only, windowed view** over a [`TemporalArray`].
//! It exposes a zero-copy slice `[offset .. offset + len)` for fast, indexable
//! access to datetime/timestamp data.
//!
//! ## Role
//! - Let APIs accept either a full `TemporalArray` or a pre-sliced view.
//! - Avoid deep copies while enabling per-window operations and previews.
//! - Optionally cache per-window null counts to speed up repeated scans.
//!
//! ## Behaviour
//! - Works with 32-bit and 64-bit datetime variants behind `TemporalArray`.
//! - Accessors - [`get_i64`](TemporalArrayV::get_i64) and
//!   [`get_i32`](TemporalArrayV::get_i32) (the latter returns `None` for 64-bit).
//! - Slicing yields another borrowed view; buffers are not cloned.
//!
//! ## Threading
//! - Not thread-safe: uses `Cell` to cache the window’s null count.
//! - For parallel use, create per-thread views via [`slice`](TemporalArrayV::slice).
//!
//! ## Interop
//! - Convert to an owned `TemporalArray` of the window with
//!   [`to_temporal_array`](TemporalArrayV::to_temporal_array).
//! - Lift to `Array` with [`as_array`](TemporalArrayV::as_array) for enum-level APIs.
//!
//! ## Invariants
//! - `offset + len <= array.len()`
//! - `len` is the logical element count of this view.

use std::cell::Cell;
use std::fmt::{self, Debug, Display, Formatter};

use crate::traits::print::MAX_PREVIEW;
use crate::{Array, ArrayV, BitmaskV, MaskedArray, TemporalArray};

/// # TemporalArrayView
///
/// Read-only, zero-copy view over a `[offset .. offset + len)` window of a
/// [`TemporalArray`].
///
/// ## Purpose
/// - Return an indexable subrange without cloning buffers.
/// - Optionally cache per-window null counts for faster repeated passes.
///
/// ## Behaviour
/// - Supports 32-bit and 64-bit datetime storage behind `TemporalArray`.
/// - Provides upcast helpers - [`get_i64`](Self::get_i64) and
///   [`get_i32`](Self::get_i32).
/// - Further slicing yields another borrowed view.
///
/// ## Fields
/// - `array`: backing [`TemporalArray`] (enum over temporal types).
/// - `offset`: starting index into the backing array.
/// - `len`: logical number of elements in the view.
/// - `null_count`: cached `Option<usize>` for this window (internal).
///
/// ## Notes
/// - Not thread-safe due to `Cell`. Create per-thread views with [`slice`](Self::slice).
/// - Use [`to_temporal_array`](Self::to_temporal_array) to materialise the window.
#[derive(Clone, PartialEq)]
pub struct TemporalArrayV {
    pub array: TemporalArray,
    pub offset: usize,
    len: usize,
    null_count: Cell<Option<usize>>
}

impl TemporalArrayV {
    /// Creates a new `TemporalArrayView` with the given offset and length.
    pub fn new(array: TemporalArray, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= array.len(),
            "TemporalArrayView: window out of bounds (offset + len = {}, array.len = {})",
            offset + len,
            array.len()
        );
        Self {
            array,
            offset,
            len,
            null_count: Cell::new(None)
        }
    }

    /// Creates a new `TemporalArrayView` with a precomputed null count.
    pub fn with_null_count(
        array: TemporalArray,
        offset: usize,
        len: usize,
        null_count: usize
    ) -> Self {
        assert!(
            offset + len <= array.len(),
            "TemporalArrayView: window out of bounds (offset + len = {}, array.len = {})",
            offset + len,
            array.len()
        );
        Self {
            array,
            offset,
            len,
            null_count: Cell::new(Some(null_count))
        }
    }

    /// Returns `true` if the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the value at logical index `i` within the window as i64 (upcasts i32 variant).
    #[inline]
    pub fn get_i64(&self, i: usize) -> Option<i64> {
        if i >= self.len {
            return None;
        }
        let phys_idx = self.offset + i;
        match &self.array {
            TemporalArray::Datetime32(arr) => arr.get(phys_idx).map(|v| v as i64),
            TemporalArray::Datetime64(arr) => arr.get(phys_idx),
            TemporalArray::Null => None
        }
    }

    /// Returns the value at logical index `i` as i32 (None for Datetime64).
    #[inline]
    pub fn get_i32(&self, i: usize) -> Option<i32> {
        if i >= self.len {
            return None;
        }
        let phys_idx = self.offset + i;
        match &self.array {
            TemporalArray::Datetime32(arr) => arr.get(phys_idx),
            TemporalArray::Datetime64(_) => None,
            TemporalArray::Null => None
        }
    }

    /// Returns a sliced `TemporalArrayView` from the current view.
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(offset + len <= self.len, "TemporalArrayView::slice: out of bounds");
        Self {
            array: self.array.clone(),
            offset: self.offset + offset,
            len,
            null_count: Cell::new(None)
        }
    }

    /// Returns the length of the window
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the underlying array as an `Array` enum value.
    ///
    /// Useful to access its inner methods
    #[inline]
    pub fn as_array(&self) -> Array {
        Array::TemporalArray(self.array.clone()) // Arc clone for data buffer
    }

    /// Converts the view into a sliced `TemporalArray`.
    pub fn to_temporal_array(&self) -> TemporalArray {
        self.as_array().slice_clone(self.offset, self.len).dt()
    }

    /// Returns the end index of the view.
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Returns the view as a tuple `(array, offset, len)`
    #[inline]
    pub fn as_tuple(&self) -> (TemporalArray, usize, usize) {
        (self.array.clone(), self.offset, self.len)
    }

    /// Returns the number of nulls in the view.
    #[inline]
    pub fn null_count(&self) -> usize {
        if let Some(count) = self.null_count.get() {
            return count;
        }
        let count = match self.array.null_mask() {
            Some(mask) => mask.to_window(self.offset, self.len).count_zeros(),
            None => 0
        };
        self.null_count.set(Some(count));
        count
    }

    /// Returns the null mask as a windowed `BitmaskView`.
    #[inline]
    pub fn null_mask_view(&self) -> Option<BitmaskV> {
        self.array.null_mask().map(|mask| mask.to_window(self.offset, self.len))
    }

    /// Sets the cached null count for the view.
    #[inline]
    pub fn set_null_count(&self, count: usize) {
        self.null_count.set(Some(count));
    }
}

impl From<TemporalArray> for TemporalArrayV {
    fn from(array: TemporalArray) -> Self {
        let len = array.len();
        TemporalArrayV {
            array,
            offset: 0,
            len,
            null_count: Cell::new(None)
        }
    }
}

impl From<Array> for TemporalArrayV {
    fn from(array: Array) -> Self {
        match array {
            Array::TemporalArray(arr) => {
                let len = arr.len();
                TemporalArrayV {
                    array: arr,
                    offset: 0,
                    len,
                    null_count: Cell::new(None)
                }
            }
            _ => panic!("Array is not a TemporalArray")
        }
    }
}

impl From<ArrayV> for TemporalArrayV {
    /// Converts an `ArrayView` to a `TemporalArrayView`, panicking if the array is not temporal.
    fn from(view: ArrayV) -> Self {
        let (array, offset, len) = view.as_tuple();
        match array {
            Array::TemporalArray(inner) => Self {
                array: inner,
                offset,
                len,
                null_count: Cell::new(None)
            },
            _ => panic!("From<ArrayView>: expected TemporalArray variant")
        }
    }
}

impl Debug for TemporalArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("TemporalArrayView")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("array", &self.array)
            .field("cached_null_count", &self.null_count.get())
            .finish()
    }
}

impl Display for TemporalArrayV {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let dtype = match &self.array {
            TemporalArray::Datetime32(_) => "Datetime32<i32>",
            TemporalArray::Datetime64(_) => "Datetime64<i64>",
            TemporalArray::Null => "Null"
        };

        writeln!(
            f,
            "TemporalArrayView<{dtype}> [{} values] (offset: {}, nulls: {})",
            self.len(),
            self.offset,
            self.null_count()
        )?;

        let max = self.len().min(MAX_PREVIEW);

        #[cfg(feature = "chrono")]
        {
            use chrono::{DateTime, Duration, NaiveDate};

            use crate::TimeUnit;
            let unit = match &self.array {
                TemporalArray::Datetime32(arr) => &arr.time_unit,
                TemporalArray::Datetime64(arr) => &arr.time_unit,
                TemporalArray::Null => &TimeUnit::Milliseconds
            };
            for i in 0..max {
                match self.get_i64(i) {
                    Some(val) => match unit {
                        TimeUnit::Seconds => {
                            let dt = DateTime::from_timestamp(val, 0)
                                .expect("Expected valid datetime value.");
                            writeln!(f, "  {dt}")?
                        }
                        TimeUnit::Milliseconds => {
                            let secs = val / 1_000;
                            let nsecs = ((val % 1_000) * 1_000_000) as u32;
                            let dt = DateTime::from_timestamp(secs, nsecs)
                                .expect("Expected valid datetime value.");
                            writeln!(f, "  {dt}")?
                        }
                        TimeUnit::Microseconds => {
                            let secs = val / 1_000_000;
                            let nsecs = ((val % 1_000_000) * 1_000) as u32;
                            let dt = DateTime::from_timestamp(secs, nsecs)
                                .expect("Expected valid datetime value.");
                            writeln!(f, "  {dt}")?
                        }
                        TimeUnit::Nanoseconds => {
                            let secs = val / 1_000_000_000;
                            let nsecs = (val % 1_000_000_000) as u32;
                            let dt = DateTime::from_timestamp(secs, nsecs)
                                .expect("Expected valid datetime value.");
                            writeln!(f, "  {dt}")?
                        }
                        TimeUnit::Days => {
                            let days = val;
                            let base = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                            let date = base.checked_add_signed(Duration::days(days));
                            match date {
                                Some(d) => writeln!(f, "  {d}")?,
                                None => writeln!(f, "  {days}d")?
                            }
                        }
                    },
                    None => writeln!(f, "  null")?
                }
            }
        }

        #[cfg(not(feature = "chrono"))]
        {
            use crate::TimeUnit;

            let unit = match &self.array {
                TemporalArray::Datetime32(arr) => &arr.time_unit,
                TemporalArray::Datetime64(arr) => &arr.time_unit,
                TemporalArray::Null => &TimeUnit::Milliseconds
            };
            let suffix = match unit {
                TimeUnit::Seconds => "s",
                TimeUnit::Milliseconds => "ms",
                TimeUnit::Microseconds => "µs",
                TimeUnit::Nanoseconds => "ns",
                TimeUnit::Days => "d"
            };
            for i in 0..max {
                match self.get_i64(i) {
                    Some(val) => writeln!(f, "  {}{}", val, suffix)?,
                    None => writeln!(f, "  null")?
                }
            }
        }

        if self.len() > MAX_PREVIEW {
            writeln!(f, "  ... ({} more)", self.len() - MAX_PREVIEW)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{Bitmask, DatetimeArray, TemporalArray};

    #[test]
    fn test_temporal_array_view_basic_indexing_and_slice() {
        let arr = DatetimeArray::<i64>::from_slice(&[10_000, 20_000, 30_000, 40_000], None);
        let temporal = TemporalArray::Datetime64(Arc::new(arr));
        let view = TemporalArrayV::new(temporal, 1, 2);

        assert_eq!(view.len(), 2);
        assert_eq!(view.offset, 1);
        assert_eq!(view.get_i64(0), Some(20_000));
        assert_eq!(view.get_i64(1), Some(30_000));
        assert_eq!(view.get_i64(2), None);

        let sub = view.slice(1, 1);
        assert_eq!(sub.len(), 1);
        assert_eq!(sub.get_i64(0), Some(30_000));
        assert_eq!(sub.get_i64(1), None);
    }

    #[test]
    fn test_temporal_array_view_null_count_and_cache() {
        let mut arr = DatetimeArray::<i64>::from_slice(&[1, 2, 3, 4], None);
        let mut mask = Bitmask::new_set_all(4, true);
        mask.set(2, false);
        arr.null_mask = Some(mask);

        let temporal = TemporalArray::Datetime64(Arc::new(arr));
        let view = TemporalArrayV::new(temporal, 0, 4);
        assert_eq!(view.null_count(), 1, "Null count should detect one null");
        assert_eq!(view.null_count(), 1);

        let view2 = view.slice(0, 2);
        assert_eq!(view2.null_count(), 0);
        let view3 = view.slice(2, 2);
        assert_eq!(view3.null_count(), 1);
    }

    #[test]
    fn test_temporal_array_view_with_supplied_null_count() {
        let arr = DatetimeArray::<i64>::from_slice(&[5, 6], None);
        let temporal = TemporalArray::Datetime64(Arc::new(arr));
        let view = TemporalArrayV::with_null_count(temporal, 0, 2, 99);
        assert_eq!(view.null_count(), 99);
        view.set_null_count(101);
        assert_eq!(view.null_count(), 101);
    }

    #[test]
    fn test_temporal_array_view_to_temporal_array_and_as_tuple() {
        let arr = DatetimeArray::<i64>::from_slice(&(10..20).collect::<Vec<_>>(), None);
        let temporal = TemporalArray::Datetime64(Arc::new(arr));
        let view = TemporalArrayV::new(temporal.clone(), 4, 3);
        let arr2 = view.to_temporal_array();
        if let TemporalArray::Datetime64(a2) = arr2 {
            assert_eq!(&a2.data[..], &[14, 15, 16]);
        } else {
            panic!("Unexpected variant");
        }
        let tup = view.as_tuple();
        assert_eq!(&tup.0, &temporal);
        assert_eq!(tup.1, 4);
        assert_eq!(tup.2, 3);
    }

    #[test]
    fn test_temporal_array_view_null_mask_view() {
        let mut arr = DatetimeArray::<i64>::from_slice(&[2, 4, 6], None);
        let mut mask = Bitmask::new_set_all(3, true);
        mask.set(0, false);
        arr.null_mask = Some(mask);

        let temporal = TemporalArray::Datetime64(Arc::new(arr));
        let view = TemporalArrayV::new(temporal, 1, 2);
        let mask_view = view.null_mask_view().expect("Should have mask");
        assert_eq!(mask_view.len(), 2);
        assert!(mask_view.get(0));
        assert!(mask_view.get(1));
    }

    #[test]
    fn test_temporal_array_view_from_temporal_array_and_array() {
        let arr = DatetimeArray::<i64>::from_slice(&[1, 2], None);
        let temporal = TemporalArray::Datetime64(Arc::new(arr));
        let view_from_temporal = TemporalArrayV::from(temporal.clone());
        assert_eq!(view_from_temporal.len(), 2);
        assert_eq!(view_from_temporal.get_i64(0), Some(1));

        let array = Array::TemporalArray(temporal);
        let view_from_array = TemporalArrayV::from(array);
        assert_eq!(view_from_array.len(), 2);
        assert_eq!(view_from_array.get_i64(1), Some(2));
    }

    #[test]
    #[should_panic(expected = "Array is not a TemporalArray")]
    fn test_temporal_array_view_from_array_panics_on_wrong_variant() {
        let array = Array::Null;
        let _view = TemporalArrayV::from(array);
    }
}
