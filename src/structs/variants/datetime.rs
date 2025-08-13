//! # DatetimeArray Module - *Mid-Level, Inner Typed DateTime Array*
//!
//! Arrow-compatible datetime/timestamp array implementation with optional null-mask,
//! 64-byte alignment, and efficient memory layout for analytical workloads.
//!
//! ## Overview
//! - Logical type: temporal values with a defined [`TimeUnit`] (seconds, milliseconds,
//!   microseconds, nanoseconds, days).
//! - Physical storage: numeric buffer (`Buffer<T>`) representing raw time offsets from
//!   the UNIX epoch or a base date, plus an optional bit-packed validity mask.
//! - Single generic type supports all Arrow datetime/timestamp variants via `Field`
//!   metadata, avoiding multiple specialised array structs.
//! - Integrates with Arrow FFI for zero-copy interop.
//!
//! ## Features
//! - **Construction** from slices, `Vec64` or plain `Vec` buffers, with optional null mask.
//! - **Mutation**: push, set, and bulk null insertion.
//! - **Iteration**: sequential and parallel (with `parallel_proc` feature).
//! - **Null handling**: optional validity mask with length validation.
//! - **Conversion**: when `chrono` feature is enabled, convert to native date/time
//!   values or `(year, month, day, hour, minute, second, millisecond, nanosecond)` tuples.
//!
//! ## Use Cases
//! - High-performance temporal analytics.
//! - FFI-based interchange with Apache Arrow or other columnar systems.
//! - Streaming or batch ingestion with incremental append.
//!
//! ## Performance Notes
//! - Prefer bulk pushes and slice construction over repeated single inserts.
//! - Parallel iteration is available with the `parallel_proc` feature.
//!
//! ## Related Types
//! - [`TimeUnit`]: enumerates supported time granularities.
//! - [`Bitmask`]: underlying null-mask storage.
//! - [`MaskedArray`]: trait defining the nullable array API.

use std::fmt::{Display, Formatter, Result as FmtResult};

#[cfg(feature = "datetime")]
use crate::Buffer;
use crate::enums::time_units::TimeUnit;
use crate::structs::allocator::Alloc64;
use crate::structs::vec64::Vec64;
use crate::traits::masked_array::MaskedArray;
use crate::traits::type_unions::Integer;
use crate::utils::validate_null_mask_len;
use crate::{
    Bitmask, Length, Offset, impl_arc_masked_array, impl_array_ref_deref, impl_masked_array
};

/// # DatetimeArray
///
/// Arrow-compatible datetime/timestamp array with 64-byte alignment and optional null mask.
///
/// ## Role
/// - Many will prefer the higher level `Array` type, which dispatches to this when
/// necessary. 
/// - Can be used as a standalone datetime array or as the datetime arm of `TemporalArray` / `Array`.
/// 
/// ## Description
/// - Stores temporal values as numeric offsets (`T: Integer`) from the UNIX epoch or a base date,
///   with units defined by [`TimeUnit`].
/// - A single struct supports all Arrow datetime/timestamp variants, with the exact logical
///   type determined by an associated `Field`’s `ArrowType`.
/// - `null_mask` is an optional bit-packed validity bitmap (`1 = valid`, `0 = null`).
/// - Implements [`MaskedArray`] for consistent nullable array behaviour.
/// - When `chrono` is enabled, provides conversion to native date/time representations.
///
/// ### Fields
/// - `data`: backing buffer storing raw temporal values.
/// - `null_mask`: optional bit-packed validity bitmap.
/// - `time_unit`: time units associated with the stored values.
///
/// ## Example
/// ```rust
/// use minarrow::{Bitmask, DatetimeArray, vec64};
/// use minarrow::enums::time_units::TimeUnit;
///
/// // Milliseconds since epoch, no nulls
/// let arr = DatetimeArray::<i64>::from_slice(&[1_700_000_000_000, 1_700_000_100_000], Some(TimeUnit::Milliseconds));
/// assert_eq!(arr.len(), 2);
/// assert_eq!(arr.value(0), Some(1_700_000_000_000));
///
/// // With nulls
/// let mask = Bitmask::from_bools(&[true, false, true]);
/// let arr_with_nulls = DatetimeArray::new(
///     vec64![1000_i64, 2000, 3000],
///     Some(mask),
///     Some(TimeUnit::Milliseconds)
/// );
/// assert_eq!(arr_with_nulls.value(1), None);
/// ```
#[cfg(feature = "datetime")]
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub struct DatetimeArray<T> {
    /// Backing buffer of time values (e.g., milliseconds since epoch).
    pub data: Buffer<T>,
    /// Optional null mask (bit-packed; 1=valid, 0=null).
    pub null_mask: Option<Bitmask>,
    /// The time units associated with the datatype
    pub time_unit: TimeUnit
}

impl<T: Integer> DatetimeArray<T> {
    /// Constructs a new, empty array.
    #[inline]
    pub fn new(
        data: impl Into<Buffer<T>>,
        null_mask: Option<Bitmask>,
        time_unit: Option<TimeUnit>
    ) -> Self {
        let data: Buffer<T> = data.into();
        validate_null_mask_len(data.len(), &null_mask);
        Self {
            data: data.into(),
            null_mask: null_mask,
            time_unit: time_unit.unwrap_or_default()
        }
    }

    /// Constructs an array with reserved capacity and optional null mask.
    ///
    /// # Arguments
    ///
    /// * `cap` - Capacity (number of elements) to reserve for the backing buffer.
    /// * `null_mask` - If true, allocates a null-mask bit vector.
    #[inline]
    pub fn with_capacity(cap: usize, null_mask: bool, time_unit: Option<TimeUnit>) -> Self {
        Self {
            data: Vec64::with_capacity(cap).into(),
            null_mask: if null_mask { Some(Bitmask::with_capacity(cap)) } else { None },
            time_unit: time_unit.unwrap_or_default()
        }
    }

    /// Constructs a new, empty array.
    #[inline]
    pub fn with_default_unit(time_unit: Option<TimeUnit>) -> Self {
        Self {
            data: Vec64::new().into(),
            null_mask: None,
            time_unit: time_unit.unwrap_or_default()
        }
    }

    /// Constructs an DateTimeArray from a slice (dense, no nulls).
    #[inline]
    pub fn from_slice(slice: &[T], time_unit: Option<TimeUnit>) -> Self {
        Self {
            data: Vec64(slice.to_vec_in(Alloc64)).into(),
            null_mask: None,
            time_unit: time_unit.unwrap_or_default()
        }
    }

    /// Returns the raw time value at index (e.g., ms since epoch), or None if null.
    #[inline]
    pub fn value(&self, idx: usize) -> Option<T> {
        if idx >= self.len() {
            None
        } else if self.is_null(idx) {
            None
        } else {
            Some(self.data[idx])
        }
    }

    /// Returns all raw values.
    #[inline]
    pub fn values(&self) -> &Buffer<T> {
        &self.data
    }

    /// Constructs a DatetimeArray by taking ownership of an existing buffer.
    ///
    /// # Arguments
    ///
    /// * `data` - Backing buffer of time values.
    /// * `null_mask` - Optional bit-packed null bitmap.
    /// * `time_unit` - Time unit metadata.
    #[inline]
    pub fn from_vec64(
        data: Vec64<T>,
        null_mask: Option<Bitmask>,
        time_unit: Option<TimeUnit>
    ) -> Self {
        Self {
            data: data.into(),
            null_mask,
            time_unit: time_unit.unwrap_or_default()
        }
    }

    /// Constructs a DatetimeArray by taking ownership of a standard Vec<T>.
    ///
    /// # Arguments
    ///
    /// * `data` - Standard heap-allocated buffer (unconditionally promoted).
    /// * `null_mask` - Optional bit-packed null bitmap.
    /// * `time_unit` - Time unit metadata.
    #[inline]
    pub fn from_vec(data: Vec<T>, null_mask: Option<Bitmask>, time_unit: Option<TimeUnit>) -> Self {
        Self::from_vec64(data.into(), null_mask, time_unit)
    }
}

// Chrono datetime conversion
#[cfg(feature = "chrono")]
impl DatetimeArray<i64> {
    /// Returns the value at index as chrono::NaiveDateTime
    ///
    /// Is the milliseconds since epoch UTC.
    #[inline]
    pub fn as_datetime(&self, idx: usize) -> Option<chrono::NaiveDateTime> {
        use chrono::{DateTime, TimeDelta, Utc};
        self.value(idx).map(|ms| {
            let epoch = DateTime::<Utc>::UNIX_EPOCH.naive_utc();
            epoch + TimeDelta::milliseconds(ms)
        })
    }

    /// Returns (year, month, day, hour, minute, second, millisecond) tuple.
    #[inline]
    pub fn tuple_dt(&self, idx: usize) -> Option<(i32, u32, u32, u32, u32, u32, u32, u32)> {
        use chrono::{Datelike, Timelike};
        self.as_datetime(idx).map(|dt| {
            (
                dt.date().year(),
                dt.date().month(),
                dt.date().day(),
                dt.time().hour(),
                dt.time().minute(),
                dt.time().second(),
                dt.time().nanosecond() / 1_000_000, // ms
                dt.time().nanosecond()
            )
        })
    }
}

impl_masked_array!(DatetimeArray, Integer, Buffer<T>, T, time_unit);
impl_array_ref_deref!(DatetimeArray<T>);
impl_arc_masked_array!(
    Inner = DatetimeArray<T>,
    T = T,
    Container = Buffer<T>,
    LogicalType = T,
    CopyType = T,
    BufferT = T,
    Variant = TemporalArray,
    Bound = Integer,
);

/// There are 2 options for displaying dates
/// One - is to enable the `chrono` feature and then they will display as native datetimes.
/// Second, is to not enable `chrono` and they will display as the value with a suffix e.g. " ms".
#[cfg(feature = "datetime")]
impl<T> Display for DatetimeArray<T>
where
    T: Integer + Display
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        use crate::traits::print::MAX_PREVIEW;

        let len = self.len();
        let nulls = self.null_count();
        writeln!(
            f,
            "DatetimeArray [{} values] (dtype: datetime[{:?}], nulls: {})",
            len, self.time_unit, nulls
        )?;

        write!(f, "[")?;

        #[cfg(feature = "chrono")]
        {
            use chrono::{DateTime, Duration, NaiveDate};
            for i in 0..usize::min(len, MAX_PREVIEW) {
                if i > 0 {
                    write!(f, ", ")?;
                }
                match self.value(i) {
                    Some(val) => match self.time_unit {
                        TimeUnit::Seconds => {
                            let v = val.to_i64().unwrap();
                            let dt = DateTime::from_timestamp(v, 0)
                                .expect("Expected valid datetime value.");
                            write!(f, "{dt}")
                        }
                        TimeUnit::Milliseconds => {
                            use chrono::DateTime;

                            let v = val.to_i64().unwrap();
                            let secs = v / 1_000;
                            let nsecs = ((v % 1_000) * 1_000_000) as u32;
                            let dt = DateTime::from_timestamp(secs, nsecs)
                                .expect("Expected valid datetime value.");
                            write!(f, "{dt}")
                        }
                        TimeUnit::Microseconds => {
                            let v = val.to_i64().unwrap();
                            let secs = v / 1_000_000;
                            let nsecs = ((v % 1_000_000) * 1_000) as u32;
                            let dt = DateTime::from_timestamp(secs, nsecs)
                                .expect("Expected valid datetime value.");
                            write!(f, "{dt}")
                        }
                        TimeUnit::Nanoseconds => {
                            let v = val.to_i64().unwrap();
                            let secs = v / 1_000_000_000;
                            let nsecs = (v % 1_000_000_000) as u32;
                            let dt = DateTime::from_timestamp(secs, nsecs)
                                .expect("Expected valid datetime value.");
                            write!(f, "{dt}")
                        }
                        TimeUnit::Days => {
                            let days = val.to_i64().unwrap();
                            let base = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                            let date = base.checked_add_signed(Duration::days(days));
                            match date {
                                Some(d) => write!(f, "{d}"),
                                None => write!(f, "{}d", val)
                            }
                        }
                    },
                    None => write!(f, "null")
                }?;
            }
        }

        #[cfg(not(feature = "chrono"))]
        {
            // Raw value plus suffix
            let suffix = match self.time_unit {
                TimeUnit::Seconds => "s",
                TimeUnit::Milliseconds => "ms",
                TimeUnit::Microseconds => "µs",
                TimeUnit::Nanoseconds => "ns",
                TimeUnit::Days => "d"
            };
            for i in 0..usize::min(len, MAX_PREVIEW) {
                if i > 0 {
                    write!(f, ", ")?;
                }
                match self.value(i) {
                    Some(val) => write!(f, "{}{}", val, suffix)?,
                    None => write!(f, "null")?
                }
            }
        }

        if len > MAX_PREVIEW {
            write!(f, ", … ({} total)", len)?;
        }

        write!(f, "]")
    }
}

#[cfg(feature = "datetime")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec64;

    #[test]
    fn test_new_and_with_capacity() {
        let arr = DatetimeArray::<i64>::default();
        assert_eq!(arr.len(), 0);
        assert!(arr.data.is_empty());
        assert!(arr.null_mask.is_none());

        let arr = DatetimeArray::<i64>::with_capacity(50, true, None);
        assert_eq!(arr.len(), 0);
        assert!(arr.data.capacity() >= 50);
        assert!(arr.null_mask.is_some());
        assert!(arr.null_mask.as_ref().unwrap().capacity() >= 50);
    }

    #[test]
    fn test_push_and_value_no_null_mask() {
        let mut arr = DatetimeArray::<i64>::with_capacity(2, false, None);
        arr.push(123);
        arr.push(456);
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.value(0), Some(123));
        assert_eq!(arr.value(1), Some(456));
        assert!(!arr.is_null(0));
        assert!(!arr.is_null(1));
    }

    #[test]
    fn test_push_and_value_with_null_mask() {
        let mut arr = DatetimeArray::<i64>::with_capacity(3, true, None);
        arr.push(1234);
        arr.push(5678);
        arr.push_null();
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.value(0), Some(1234));
        assert_eq!(arr.value(1), Some(5678));
        assert_eq!(arr.value(2), None);
        assert!(arr.is_null(2));
    }

    #[test]
    fn test_push_null_auto_mask() {
        let mut arr = DatetimeArray::<i64>::default();
        arr.push_null();
        assert_eq!(arr.len(), 1);
        assert!(arr.is_null(0));
        assert!(arr.null_mask.is_some());
        assert_eq!(arr.value(0), None);
    }

    #[test]
    fn test_set_and_set_null() {
        let mut arr = DatetimeArray::<i64>::with_capacity(3, true, None);
        arr.push(1);
        arr.push(2);
        arr.push(3);
        arr.set(1, 99);
        assert_eq!(arr.value(1), Some(99));
        arr.set_null(1);
        assert_eq!(arr.value(1), None);
        assert!(arr.is_null(1));
    }

    #[test]
    fn test_values_method() {
        let mut arr = DatetimeArray::<i32>::with_capacity(3, false, None);
        arr.push(10);
        arr.push(20);
        arr.push(30);
        assert_eq!(arr.values(), &vec64![10, 20, 30]);
    }

    #[test]
    fn test_out_of_bounds_value() {
        let mut arr = DatetimeArray::<i64>::with_capacity(1, true, None);
        arr.push(9);
        assert_eq!(arr.value(1), None);
        assert_eq!(arr.value(100), None);
    }

    #[test]
    fn test_is_empty() {
        let arr = DatetimeArray::<i64>::default();
        assert!(arr.is_empty());
        let mut arr = DatetimeArray::<i64>::with_capacity(2, true, None);
        arr.push(0);
        assert!(!arr.is_empty());
    }

    #[test]
    fn test_push_nulls_bulk() {
        let mut arr = DatetimeArray::<i64>::with_capacity(4, true, None);
        arr.push(100);
        arr.push_nulls(5);
        assert_eq!(arr.len(), 6);
        assert_eq!(arr.value(0), Some(100));
        for i in 1..6 {
            assert_eq!(arr.value(i), None);
        }
    }

    #[test]
    fn test_masked_and_mutable_trait() {
        let mut arr = DatetimeArray::<i64>::with_capacity(3, true, None);
        arr.push(1);
        arr.push(2);
        arr.push(3);
        assert_eq!(arr.get(0), Some(1));
        assert_eq!(arr.get(2), Some(3));
        arr.set_null(0);
        assert_eq!(arr.get(0), None);
        arr.set(2, 88);
        assert_eq!(arr.get(2), Some(88));
    }

    #[test]
    fn test_masked_trait_no_mask() {
        let mut arr = DatetimeArray::<i64>::with_capacity(2, false, None);
        arr.push(3);
        arr.push(5);
        assert_eq!(arr.get(0), Some(3));
        assert!(!arr.is_null(1));
    }

    #[test]
    fn test_unsigned_type() {
        let mut arr = DatetimeArray::<u64>::with_capacity(3, true, None);
        arr.push(1);
        arr.push(2);
        arr.push_null();
        assert_eq!(arr.value(0), Some(1));
        assert_eq!(arr.value(2), None);
    }

    #[cfg(feature = "chrono")]
    #[test]
    fn test_as_datetime_and_tuple_dt() {
        use chrono::{DateTime, Datelike, TimeDelta, Timelike, Utc};
        let mut arr = DatetimeArray::<i64>::with_capacity(2, true, None);
        let ms = 1_700_000_000_000;
        arr.push(ms);
        arr.push_null();

        let dt = arr.as_datetime(0).unwrap();
        let expected = DateTime::<Utc>::UNIX_EPOCH.naive_utc() + TimeDelta::milliseconds(ms);
        assert_eq!(dt, expected);

        let tuple = arr.tuple_dt(0).unwrap();
        assert_eq!(tuple.0, dt.year());
        assert_eq!(tuple.1, dt.month());
        assert_eq!(tuple.2, dt.day());
        assert_eq!(tuple.3, dt.hour());
        assert_eq!(tuple.4, dt.minute());
        assert_eq!(tuple.5, dt.second());
        assert_eq!(tuple.6, dt.nanosecond() / 1_000_000);
        assert_eq!(tuple.7, dt.nanosecond());

        assert!(arr.as_datetime(1).is_none());
        assert!(arr.tuple_dt(1).is_none());
    }

    #[test]
    fn test_datetime_array_slice() {
        use crate::enums::time_units::TimeUnit;

        let mut arr = DatetimeArray::<i64>::with_capacity(5, true, Some(TimeUnit::Milliseconds));
        arr.push(1000);
        arr.push(2000);
        arr.push(3000);
        arr.push_null(); // index 3 is null
        arr.push(5000);

        let sliced = arr.slice_clone(1, 3);
        assert_eq!(sliced.len(), 3);
        assert_eq!(sliced.value(0), Some(2000));
        assert_eq!(sliced.value(1), Some(3000));
        assert_eq!(sliced.value(2), None);
        assert!(sliced.is_null(2));
        assert_eq!(sliced.null_count(), 1);
        assert_eq!(sliced.time_unit, TimeUnit::Milliseconds);
    }
}

#[cfg(test)]
#[cfg(feature = "parallel_proc")]
mod parallel_tests {
    use rayon::prelude::*;

    use super::*;
    use crate::enums::time_units::TimeUnit;

    #[test]
    fn test_datetimearray_par_iter_all_valid_refs() {
        let arr =
            DatetimeArray::<i64>::from_slice(&[1000, 2000, 3000], Some(TimeUnit::Milliseconds));
        let out: Vec<&i64> = arr.par_iter().collect();
        let expected: Vec<i64> = vec![1000, 2000, 3000];
        for (i, val) in out.iter().enumerate() {
            assert_eq!(**val, expected[i]);
        }
    }

    #[test]
    fn test_datetimearray_par_iter_nulls() {
        let mut arr = DatetimeArray::<i64>::with_default_unit(Some(TimeUnit::Milliseconds));
        arr.data.extend_from_slice(&[10, 20, 30]);
        let mut mask = Bitmask::new_set_all(3, false);
        mask.set_bits_chunk(0, 0b0000_0011, 3);
        arr.null_mask = Some(mask);
        let out: Vec<&i64> = arr.par_iter().collect();
        let expected: Vec<i64> = vec![10, 20, 30];
        for (i, val) in out.iter().enumerate() {
            assert_eq!(**val, expected[i]);
        }
    }

    #[test]
    fn test_datetimearray_par_iter_opt_all_valid() {
        let arr = DatetimeArray::<i64>::from_slice(&[111, 222, 333], Some(TimeUnit::Seconds));
        let out: Vec<Option<&i64>> = arr.par_iter_opt().collect();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], Some(&111));
        assert_eq!(out[1], Some(&222));
        assert_eq!(out[2], Some(&333));
    }

    #[test]
    fn test_datetimearray_par_iter_opt_with_nulls() {
        let mut arr = DatetimeArray::<i64>::with_default_unit(Some(TimeUnit::Microseconds));
        arr.data.extend_from_slice(&[77, 88, 99, 111]);
        let mut mask = Bitmask::new_set_all(4, false);
        mask.set_bits_chunk(0, 0b0000_1011, 4);
        arr.null_mask = Some(mask);
        let out: Vec<Option<&i64>> = arr.par_iter_opt().collect();
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], Some(&77));
        assert_eq!(out[1], Some(&88));
        assert_eq!(out[2], None);
        assert_eq!(out[3], Some(&111));
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_datetimearray_par_iter_range_unchecked() {
        let arr = DatetimeArray::<i64>::from_slice(
            &[100, 200, 300, 400],
            Some(crate::enums::time_units::TimeUnit::Milliseconds)
        );
        let out: Vec<&i64> = unsafe { arr.par_iter_range_unchecked(1, 3).collect() };
        assert_eq!(*out[0], 200);
        assert_eq!(*out[1], 300);
    }
}
