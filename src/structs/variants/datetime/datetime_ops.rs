//! # **DatetimeOps Module** - *Datetime Operations for DatetimeArray*
//!
//! Datetime operations for `DatetimeArray` built on the `time` crate.
//!
//! ## Features
//! - **Arithmetic**: add/subtract durations, days, months, years with overflow checking
//! - **Duration calculations**: diff and absolute diff between datetime arrays
//! - **Comparisons**: temporal ordering (before/after) and range checks (between)
//! - **Component extraction**: year, month, day, hour, minute, second, weekday, quarter, ISO week
//! - **Truncation**: round down to year, month, week, day, hour, minute, second, or sub-second boundaries
//! - **Type casting**: convert between time units (seconds <-> milliseconds <-> microseconds <-> nanoseconds)
//!
//! ## Timezone Handling
//! `DatetimeArray` stores raw UTC timestamps as integers. Timezone information is metadata-only
//! and stored in `ArrowType::Timestamp(TimeUnit, Option<String>)` on the `Field` level.
//! For timezone operations, use `FieldArray::with_timezone()` and `FieldArray::to_utc()` which
//! update metadata without modifying the underlying UTC data.
//!
//! ## Requirements
//! Requires the `time` feature to be enabled.

#[cfg(feature = "datetime_ops")]
use time::Duration;

#[cfg(feature = "datetime_ops")]
use crate::{
    DatetimeArray,
    enums::{error::MinarrowError, time_units::TimeUnit},
    structs::variants::{boolean::BooleanArray, integer::IntegerArray},
    traits::{masked_array::MaskedArray, type_unions::Integer},
};
#[cfg(feature = "datetime_ops")]
use num_traits::FromPrimitive;

#[cfg(feature = "datetime_ops")]
impl<T: Integer + FromPrimitive> DatetimeArray<T> {
    /// Convert i64 value to OffsetDateTime based on time unit.
    #[inline(always)]
    fn i64_to_datetime(val_i64: i64, time_unit: TimeUnit) -> Option<time::OffsetDateTime> {
        use time::OffsetDateTime;
        match time_unit {
            TimeUnit::Seconds => OffsetDateTime::from_unix_timestamp(val_i64).ok(),
            TimeUnit::Milliseconds => {
                OffsetDateTime::from_unix_timestamp_nanos((val_i64 as i128) * 1_000_000).ok()
            }
            TimeUnit::Microseconds => {
                OffsetDateTime::from_unix_timestamp_nanos((val_i64 as i128) * 1_000).ok()
            }
            TimeUnit::Nanoseconds => {
                OffsetDateTime::from_unix_timestamp_nanos(val_i64 as i128).ok()
            }
            TimeUnit::Days => time::Date::from_julian_day((val_i64 + 2440588) as i32)
                .ok()
                .and_then(|date| date.with_hms(0, 0, 0).ok())
                .map(|dt| dt.assume_utc()),
        }
    }

    /// Convert OffsetDateTime back to i64 value based on time unit.
    #[inline(always)]
    fn datetime_to_i64(dt: time::OffsetDateTime, time_unit: TimeUnit) -> i64 {
        match time_unit {
            TimeUnit::Seconds => dt.unix_timestamp(),
            TimeUnit::Milliseconds => {
                dt.unix_timestamp() * 1_000i64 + (dt.nanosecond() / 1_000_000) as i64
            }
            TimeUnit::Microseconds => {
                dt.unix_timestamp() * 1_000_000i64 + (dt.nanosecond() / 1_000) as i64
            }
            TimeUnit::Nanoseconds => {
                dt.unix_timestamp() * 1_000_000_000i64 + dt.nanosecond() as i64
            }
            TimeUnit::Days => dt.unix_timestamp() / 86400i64,
        }
    }

    /// Helper to convert a value at index to OffsetDateTime.
    ///
    /// Performs bounds and null checks. For hot loops, prefer direct i64 conversion
    /// with `i64_to_datetime` after manual validation.
    fn value_to_datetime(&self, i: usize) -> Option<time::OffsetDateTime> {
        if self.is_null(i) || i >= self.len() {
            return None;
        }
        let val_i64 = self.data[i].to_i64()?;
        Self::i64_to_datetime(val_i64, self.time_unit)
    }

    // Arithmetic Operations

    /// Adds a duration to all datetime values in the array.
    ///
    /// # Example
    /// ```ignore
    /// use minarrow::{DatetimeArray, TimeUnit};
    /// use time::Duration;
    ///
    /// let arr = DatetimeArray::<i64>::from_slice(&[1000, 2000], Some(TimeUnit::Milliseconds));
    /// let result = arr.add_duration(Duration::seconds(5)).unwrap();
    /// ```
    pub fn add_duration(&self, duration: Duration) -> Result<Self, MinarrowError> {
        let mut result = self.clone();
        let len = result.len();
        let data = &self.data[..];

        // Convert duration to the array's time unit
        let duration_value: i64 =
            match self.time_unit {
                TimeUnit::Seconds => duration.whole_seconds(),
                TimeUnit::Milliseconds => {
                    duration.whole_milliseconds().try_into().map_err(|_| {
                        MinarrowError::Overflow {
                            value: format!("{} ms", duration.whole_milliseconds()),
                            target: "i64",
                        }
                    })?
                }
                TimeUnit::Microseconds => {
                    duration.whole_microseconds().try_into().map_err(|_| {
                        MinarrowError::Overflow {
                            value: format!("{} μs", duration.whole_microseconds()),
                            target: "i64",
                        }
                    })?
                }
                TimeUnit::Nanoseconds => duration.whole_nanoseconds().try_into().map_err(|_| {
                    MinarrowError::Overflow {
                        value: format!("{} ns", duration.whole_nanoseconds()),
                        target: "i64",
                    }
                })?,
                TimeUnit::Days => duration.whole_days(),
            };

        for i in 0..len {
            if !self.is_null(i) {
                // SAFETY: i is bounded by len, so access is safe
                let val = unsafe { *data.get_unchecked(i) };

                if let Some(val_i64) = val.to_i64() {
                    if let Some(new_val_i64) = val_i64.checked_add(duration_value) {
                        if let Some(new_val_t) = T::from_i64(new_val_i64) {
                            result.set(i, new_val_t);
                        } else {
                            result.set_null(i);
                        }
                    } else {
                        result.set_null(i); // Overflow
                    }
                } else {
                    result.set_null(i);
                }
            }
        }

        Ok(result)
    }

    /// Subtracts a duration from all datetime values in the array.
    pub fn sub_duration(&self, duration: Duration) -> Result<Self, MinarrowError> {
        self.add_duration(-duration)
    }

    /// Adds a number of days to all datetime values.
    pub fn add_days(&self, days: i64) -> Result<Self, MinarrowError> {
        self.add_duration(Duration::days(days))
    }

    /// Adds a number of months to all datetime values.
    ///
    /// Variable month length handling - if the day would be invalid in the target month
    /// (e.g., Jan 31 + 1 month), the day is clamped to the last valid day (Feb 28/29).
    /// Time-of-day is preserved.
    pub fn add_months(&self, months: i32) -> Result<Self, MinarrowError> {
        let mut result = self.clone();
        let len = result.len();
        let data = &self.data[..];
        let time_unit = self.time_unit; // Hoist time_unit outside loop

        for i in 0..len {
            if !self.is_null(i) {
                // SAFETY: i is bounded by len
                let val = unsafe { *data.get_unchecked(i) };

                if let Some(val_i64) = val.to_i64() {
                    if let Some(dt) = Self::i64_to_datetime(val_i64, time_unit) {
                        let date = dt.date();

                        // Calculate new year and month
                        let total_months = date.year() * 12 + (date.month() as i32) - 1 + months;
                        let new_year = total_months / 12;
                        let new_month = (total_months % 12 + 1) as u8;

                        // Try to construct new date
                        if let Ok(new_month_enum) = time::Month::try_from(new_month) {
                            let days_in_month = new_month_enum.length(new_year);
                            let day = date.day().min(days_in_month);
                            if let Ok(new_date) =
                                time::Date::from_calendar_date(new_year, new_month_enum, day)
                            {
                                let new_dt_primitive = new_date.with_time(dt.time());
                                let new_dt = new_dt_primitive.assume_utc();
                                let new_val_i64 = Self::datetime_to_i64(new_dt, time_unit);

                                if let Some(new_val_t) = T::from_i64(new_val_i64) {
                                    result.set(i, new_val_t);
                                } else {
                                    result.set_null(i);
                                }
                            } else {
                                result.set_null(i);
                            }
                        } else {
                            result.set_null(i);
                        }
                    } else {
                        result.set_null(i);
                    }
                } else {
                    result.set_null(i);
                }
            }
        }

        Ok(result)
    }

    /// Adds a number of years to all datetime values.
    pub fn add_years(&self, years: i32) -> Result<Self, MinarrowError> {
        self.add_months(years * 12)
    }

    // Duration Operations

    /// Calculate the duration between this datetime array and another.
    /// Returns an IntegerArray<i64> representing the difference in the specified unit.
    ///
    /// # Arguments
    /// * `other` - The datetime array to subtract from self
    /// * `unit` - The TimeUnit for the result (Seconds, Milliseconds, Microseconds, Nanoseconds)
    pub fn diff(&self, other: &Self, unit: TimeUnit) -> Result<IntegerArray<i64>, MinarrowError> {
        if self.len() != other.len() {
            return Err(MinarrowError::TypeError {
                from: "DatetimeArray",
                to: "IntegerArray",
                message: Some(format!(
                    "Array lengths do not match: {} vs {}",
                    self.len(),
                    other.len()
                )),
            });
        }

        let mut result =
            IntegerArray::with_capacity(self.len(), self.is_nullable() || other.is_nullable());

        for i in 0..self.len() {
            if self.is_null(i) || other.is_null(i) {
                result.push_null();
            } else {
                let self_dt = self.value_to_datetime(i);
                let other_dt = other.value_to_datetime(i);

                if let (Some(a), Some(b)) = (self_dt, other_dt) {
                    let diff_duration = a - b;

                    let diff_value = match unit {
                        TimeUnit::Seconds => diff_duration.whole_seconds(),
                        TimeUnit::Milliseconds => diff_duration.whole_milliseconds() as i64,
                        TimeUnit::Microseconds => diff_duration.whole_microseconds() as i64,
                        TimeUnit::Nanoseconds => diff_duration.whole_nanoseconds() as i64,
                        TimeUnit::Days => diff_duration.whole_days(),
                    };

                    result.push(diff_value);
                } else {
                    result.push_null();
                }
            }
        }

        Ok(result)
    }

    /// Calculate the absolute duration between elements (always positive).
    pub fn abs_diff(
        &self,
        other: &Self,
        unit: TimeUnit,
    ) -> Result<IntegerArray<i64>, MinarrowError> {
        let diff = self.diff(other, unit)?;
        let mut result = IntegerArray::with_capacity(diff.len(), diff.is_nullable());

        for i in 0..diff.len() {
            if diff.is_null(i) {
                result.push_null();
            } else {
                result.push(diff.data[i].abs());
            }
        }

        Ok(result)
    }

    // Comparison Operations

    /// Compares this array with another, returning a boolean array indicating
    /// where values in `self` are before values in `other`.
    pub fn is_before(&self, other: &Self) -> Result<BooleanArray<()>, MinarrowError> {
        if self.len() != other.len() {
            return Err(MinarrowError::TypeError {
                from: "DatetimeArray",
                to: "BooleanArray",
                message: Some(format!(
                    "Array lengths do not match: {} vs {}",
                    self.len(),
                    other.len()
                )),
            });
        }

        let mut result = BooleanArray::with_capacity(self.len(), true);

        for i in 0..self.len() {
            if self.is_null(i) || other.is_null(i) {
                result.push_null();
            } else {
                let self_dt = self.value_to_datetime(i);
                let other_dt = other.value_to_datetime(i);

                if let (Some(a), Some(b)) = (self_dt, other_dt) {
                    result.push(if a < b { true } else { false });
                } else {
                    result.push_null();
                }
            }
        }

        Ok(result)
    }

    /// Returns a boolean array indicating where values are after `other`.
    pub fn is_after(&self, other: &Self) -> Result<BooleanArray<()>, MinarrowError> {
        if self.len() != other.len() {
            return Err(MinarrowError::TypeError {
                from: "DatetimeArray",
                to: "BooleanArray",
                message: Some(format!(
                    "Array lengths do not match: {} vs {}",
                    self.len(),
                    other.len()
                )),
            });
        }

        let mut result = BooleanArray::with_capacity(self.len(), true);

        for i in 0..self.len() {
            if self.is_null(i) || other.is_null(i) {
                result.push_null();
            } else {
                let self_dt = self.value_to_datetime(i);
                let other_dt = other.value_to_datetime(i);

                if let (Some(a), Some(b)) = (self_dt, other_dt) {
                    result.push(if a > b { true } else { false });
                } else {
                    result.push_null();
                }
            }
        }

        Ok(result)
    }

    /// Returns a boolean array indicating where values fall within the range [start, end].
    pub fn between(&self, start: &Self, end: &Self) -> Result<BooleanArray<()>, MinarrowError> {
        if self.len() != start.len() || self.len() != end.len() {
            return Err(MinarrowError::TypeError {
                from: "DatetimeArray",
                to: "BooleanArray",
                message: Some("Array lengths do not match".to_string()),
            });
        }

        let mut result = BooleanArray::with_capacity(self.len(), true);

        for i in 0..self.len() {
            if self.is_null(i) || start.is_null(i) || end.is_null(i) {
                result.push_null();
            } else {
                let self_dt = self.value_to_datetime(i);
                let start_dt = start.value_to_datetime(i);
                let end_dt = end.value_to_datetime(i);

                if let (Some(val), Some(s), Some(e)) = (self_dt, start_dt, end_dt) {
                    result.push(if val >= s && val <= e { true } else { false });
                } else {
                    result.push_null();
                }
            }
        }

        Ok(result)
    }

    // Component Extraction Operations

    /// Extracts the year component from all datetime values.
    pub fn year(&self) -> IntegerArray<i32> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.value_to_datetime(i) {
                result.push(dt.year());
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the month component (1-12) from all datetime values.
    pub fn month(&self) -> IntegerArray<u8> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.value_to_datetime(i) {
                result.push(dt.month() as u8);
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the day of month (1-31) from all datetime values.
    pub fn day(&self) -> IntegerArray<u8> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.value_to_datetime(i) {
                result.push(dt.day());
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the hour component (0-23) from all datetime values.
    pub fn hour(&self) -> IntegerArray<u8> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.value_to_datetime(i) {
                result.push(dt.hour());
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the minute component (0-59) from all datetime values.
    pub fn minute(&self) -> IntegerArray<u8> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.value_to_datetime(i) {
                result.push(dt.minute());
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the second component (0-59) from all datetime values.
    pub fn second(&self) -> IntegerArray<u8> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.value_to_datetime(i) {
                result.push(dt.second());
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the weekday (1=Sunday, 2=Monday, ..., 7=Saturday) from all datetime values.
    pub fn weekday(&self) -> IntegerArray<u8> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.value_to_datetime(i) {
                result.push(dt.weekday().number_from_sunday());
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the day of year (1-366) from all datetime values.
    pub fn day_of_year(&self) -> IntegerArray<u16> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.value_to_datetime(i) {
                result.push(dt.ordinal());
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the ISO week number (1-53) from all datetime values.
    pub fn iso_week(&self) -> IntegerArray<u8> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.value_to_datetime(i) {
                result.push(dt.iso_week());
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the quarter (1-4) from all datetime values.
    pub fn quarter(&self) -> IntegerArray<u8> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.value_to_datetime(i) {
                let month = dt.month() as u8;
                let quarter = ((month - 1) / 3) + 1;
                result.push(quarter);
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Extracts the week of year (0-53) from all datetime values.
    /// Week 0 contains days before the first Sunday.
    pub fn week_of_year(&self) -> IntegerArray<u8> {
        let mut result = IntegerArray::with_capacity(self.len(), self.is_nullable());

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.value_to_datetime(i) {
                let day_of_year = dt.ordinal();
                let weekday = dt.weekday().number_from_sunday();
                // Calculate week based on day of year and weekday (Sunday = 1)
                let week = ((day_of_year + 7 - weekday as u16) / 7) as u8;
                result.push(week);
            } else {
                result.push_null();
            }
        }

        result
    }

    /// Returns boolean array indicating whether each datetime's year is a leap year.
    pub fn is_leap_year(&self) -> BooleanArray<()> {
        let mut result = BooleanArray::with_capacity(self.len(), self.is_nullable());

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.value_to_datetime(i) {
                let year = dt.year();
                let is_leap = time::util::is_leap_year(year);
                result.push(is_leap);
            } else {
                result.push_null();
            }
        }

        result
    }

    // Rounding/Truncating Operations

    /// Truncate/floor datetime values to the start of the specified unit.
    ///
    /// # Arguments
    /// * `unit` - The unit to truncate to (Day, Hour, Minute, Second)
    pub fn truncate(&self, unit: &str) -> Result<Self, MinarrowError> {
        let mut result = self.clone();
        let len = result.len();
        let time_unit = self.time_unit; // Hoist time_unit outside loop

        for i in 0..len {
            if !self.is_null(i) {
                if let Some(val_i64) = self.data[i].to_i64() {
                    if let Some(dt) = Self::i64_to_datetime(val_i64, time_unit) {
                        let truncated_dt = match unit {
                            "year" => {
                                time::Date::from_calendar_date(dt.year(), time::Month::January, 1)
                                    .ok()
                                    .and_then(|d| d.with_hms(0, 0, 0).ok())
                                    .map(|pdt| pdt.assume_utc())
                            }
                            "month" => time::Date::from_calendar_date(dt.year(), dt.month(), 1)
                                .ok()
                                .and_then(|d| d.with_hms(0, 0, 0).ok())
                                .map(|pdt| pdt.assume_utc()),
                            "day" => dt.date().with_hms(0, 0, 0).ok().map(|pdt| pdt.assume_utc()),
                            "hour" => dt
                                .date()
                                .with_hms(dt.hour(), 0, 0)
                                .ok()
                                .map(|pdt| pdt.assume_utc()),
                            "minute" => dt
                                .date()
                                .with_hms(dt.hour(), dt.minute(), 0)
                                .ok()
                                .map(|pdt| pdt.assume_utc()),
                            "second" => dt
                                .date()
                                .with_hms(dt.hour(), dt.minute(), dt.second())
                                .ok()
                                .map(|pdt| pdt.assume_utc()),
                            _ => {
                                return Err(MinarrowError::TypeError {
                                    from: "String",
                                    to: "TimeUnit",
                                    message: Some(format!("Invalid truncation unit: {}", unit)),
                                });
                            }
                        };

                        if let Some(new_dt) = truncated_dt {
                            let new_val_i64 = Self::datetime_to_i64(new_dt, time_unit);
                            if let Some(new_val_t) = T::from_i64(new_val_i64) {
                                result.set(i, new_val_t);
                            } else {
                                result.set_null(i);
                            }
                        } else {
                            result.set_null(i);
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    // Truncation Shorthand Methods

    /// Truncate to microsecond boundaries.
    ///
    /// Only meaningful for nanosecond precision arrays.
    pub fn us(&self) -> Self {
        let mut result = self.clone();

        // Only meaningful for nanosecond precision
        if self.time_unit == TimeUnit::Nanoseconds {
            let len = result.len();
            let data = &self.data[..];

            for i in 0..len {
                if !self.is_null(i) {
                    // SAFETY: i is bounded by len
                    let val = unsafe { *data.get_unchecked(i) };
                    if let Some(val_i64) = val.to_i64() {
                        // Truncate to microseconds (1000 nanoseconds)
                        let truncated = (val_i64 / 1_000) * 1_000;
                        if let Some(new_val_t) = T::from_i64(truncated) {
                            result.set(i, new_val_t);
                        }
                    }
                }
            }
        }

        result
    }

    /// Truncate to millisecond boundaries.
    ///
    /// Meaningful for nanosecond and microsecond precision arrays.
    pub fn ms(&self) -> Self {
        let mut result = self.clone();
        let len = result.len();
        let data = &self.data[..];

        match self.time_unit {
            TimeUnit::Nanoseconds => {
                for i in 0..len {
                    if !self.is_null(i) {
                        // SAFETY: i is bounded by len
                        let val = unsafe { *data.get_unchecked(i) };
                        if let Some(val_i64) = val.to_i64() {
                            // Truncate to milliseconds (1_000_000 nanoseconds)
                            let truncated = (val_i64 / 1_000_000) * 1_000_000;
                            if let Some(new_val_t) = T::from_i64(truncated) {
                                result.set(i, new_val_t);
                            }
                        }
                    }
                }
            }
            TimeUnit::Microseconds => {
                for i in 0..len {
                    if !self.is_null(i) {
                        // SAFETY: i is bounded by len
                        let val = unsafe { *data.get_unchecked(i) };
                        if let Some(val_i64) = val.to_i64() {
                            // Truncate to milliseconds (1_000 microseconds)
                            let truncated = (val_i64 / 1_000) * 1_000;
                            if let Some(new_val_t) = T::from_i64(truncated) {
                                result.set(i, new_val_t);
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        result
    }

    /// Truncate to second boundaries.
    pub fn sec(&self) -> Self {
        self.truncate("second").unwrap_or_else(|_| self.clone())
    }

    /// Truncate to minute boundaries.
    pub fn min(&self) -> Self {
        self.truncate("minute").unwrap_or_else(|_| self.clone())
    }

    /// Truncate to hour boundaries.
    pub fn hr(&self) -> Self {
        self.truncate("hour").unwrap_or_else(|_| self.clone())
    }

    /// Truncate to week boundaries (Sunday 00:00:00).
    pub fn week(&self) -> Self {
        let mut result = self.clone();
        let len = result.len();
        let time_unit = self.time_unit; // Hoist time_unit outside loop

        for i in 0..len {
            if !self.is_null(i) {
                if let Some(val_i64) = self.data[i].to_i64() {
                    if let Some(dt) = Self::i64_to_datetime(val_i64, time_unit) {
                        // Get the current weekday (1=Sunday, 2=Monday, ..., 7=Saturday)
                        let weekday = dt.weekday().number_from_sunday();
                        // Calculate days to subtract to get to Sunday
                        let days_to_sunday = (weekday - 1) as i64;

                        // Subtract days and zero out time
                        if let Some(week_start) =
                            dt.checked_sub(time::Duration::days(days_to_sunday))
                        {
                            if let Some(week_start_dt) = week_start.date().with_hms(0, 0, 0).ok() {
                                let truncated_dt = week_start_dt.assume_utc();
                                let new_val_i64 = Self::datetime_to_i64(truncated_dt, time_unit);

                                if let Some(new_val_t) = T::from_i64(new_val_i64) {
                                    result.set(i, new_val_t);
                                } else {
                                    result.set_null(i);
                                }
                            }
                        }
                    }
                }
            }
        }

        result
    }

    // Type Casting Operations

    /// Cast this DatetimeArray to a different TimeUnit.
    ///
    /// This converts the internal representation while preserving the logical datetime values.
    pub fn cast_time_unit(&self, new_unit: TimeUnit) -> Result<Self, MinarrowError> {
        let mut result =
            Self::with_capacity(self.len(), self.is_nullable(), Some(new_unit.clone()));

        for i in 0..self.len() {
            if self.is_null(i) {
                result.push_null();
            } else if let Some(dt) = self.value_to_datetime(i) {
                // Convert to new unit
                let new_val: i64 = match new_unit {
                    TimeUnit::Seconds => dt.unix_timestamp(),
                    TimeUnit::Milliseconds => {
                        dt.unix_timestamp() * 1_000i64 + (dt.nanosecond() / 1_000_000) as i64
                    }
                    TimeUnit::Microseconds => {
                        dt.unix_timestamp() * 1_000_000i64 + (dt.nanosecond() / 1_000) as i64
                    }
                    TimeUnit::Nanoseconds => {
                        dt.unix_timestamp() * 1_000_000_000i64 + dt.nanosecond() as i64
                    }
                    TimeUnit::Days => dt.unix_timestamp() / 86400i64,
                };

                if let Some(new_val_t) = T::from_i64(new_val) {
                    result.push(new_val_t);
                } else {
                    result.push_null();
                }
            } else {
                result.push_null();
            }
        }

        Ok(result)
    }
}

#[cfg(all(test, feature = "datetime_ops"))]
mod tests {
    use super::*;
    use time::Duration;
    use vec64::vec64;

    #[test]
    fn test_add_duration() {
        let arr = DatetimeArray::<i64>::from_slice(&[1000, 2000, 3000], Some(TimeUnit::Seconds));
        let result = arr.add_duration(Duration::seconds(10)).unwrap();

        assert_eq!(result.value(0), Some(1010));
        assert_eq!(result.value(1), Some(2010));
        assert_eq!(result.value(2), Some(3010));
    }

    #[test]
    fn test_sub_duration() {
        let arr = DatetimeArray::<i64>::from_slice(&[1000, 2000, 3000], Some(TimeUnit::Seconds));
        let result = arr.sub_duration(Duration::seconds(10)).unwrap();

        assert_eq!(result.value(0), Some(990));
        assert_eq!(result.value(1), Some(1990));
        assert_eq!(result.value(2), Some(2990));
    }

    #[test]
    fn test_add_days() {
        let arr = DatetimeArray::<i64>::from_slice(&[0, 86400, 172800], Some(TimeUnit::Seconds));
        let result = arr.add_days(1).unwrap();

        assert_eq!(result.value(0), Some(86400));
        assert_eq!(result.value(1), Some(172800));
        assert_eq!(result.value(2), Some(259200));
    }

    #[test]
    fn test_is_before() {
        let arr1 = DatetimeArray::<i64>::from_slice(&[1000, 2000, 3000], Some(TimeUnit::Seconds));
        let arr2 = DatetimeArray::<i64>::from_slice(&[1500, 2500, 2500], Some(TimeUnit::Seconds));

        let result = arr1.is_before(&arr2).unwrap();

        assert_eq!(result.get(0), Some(true));
        assert_eq!(result.get(1), Some(true));
        assert_eq!(result.get(2), Some(false));
    }

    #[test]
    fn test_is_after() {
        let arr1 = DatetimeArray::<i64>::from_slice(&[1000, 2000, 3000], Some(TimeUnit::Seconds));
        let arr2 = DatetimeArray::<i64>::from_slice(&[1500, 2500, 2500], Some(TimeUnit::Seconds));

        let result = arr1.is_after(&arr2).unwrap();

        assert_eq!(result.get(0), Some(false));
        assert_eq!(result.get(1), Some(false));
        assert_eq!(result.get(2), Some(true));
    }

    #[test]
    fn test_component_extraction() {
        // 2023-11-14 22:13:20 UTC
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        assert_eq!(arr.year().get(0), Some(2023));
        assert_eq!(arr.month().get(0), Some(11));
        assert_eq!(arr.day().get(0), Some(14));
        assert_eq!(arr.hour().get(0), Some(22));
        assert_eq!(arr.minute().get(0), Some(13));
        assert_eq!(arr.second().get(0), Some(20));
    }

    // Arithmetic Operations Tests

    #[test]
    fn test_add_months() {
        // 2023-01-15 -> add 2 months -> 2023-03-15
        let timestamp = 1_673_740_800i64; // 2023-01-15 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));
        let result = arr.add_months(2).unwrap();

        // Verify we got March 15th, 2023
        assert_eq!(result.year().get(0), Some(2023));
        assert_eq!(result.month().get(0), Some(3));
        assert_eq!(result.day().get(0), Some(15));
    }

    #[test]
    fn test_add_months_overflow_year() {
        // 2023-11-15 -> add 3 months -> 2024-02-15
        let timestamp = 1_700_006_400i64; // 2023-11-15 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));
        let result = arr.add_months(3).unwrap();

        assert_eq!(result.year().get(0), Some(2024));
        assert_eq!(result.month().get(0), Some(2));
        assert_eq!(result.day().get(0), Some(15));
    }

    #[test]
    fn test_add_months_end_of_month() {
        // 2023-01-31 -> add 1 month -> 2023-02-28 (not 02-31)
        let timestamp = 1_675_123_200i64; // 2023-01-31 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));
        let result = arr.add_months(1).unwrap();

        assert_eq!(result.year().get(0), Some(2023));
        assert_eq!(result.month().get(0), Some(2));
        assert_eq!(result.day().get(0), Some(28)); // Clamped to Feb 28
    }

    #[test]
    fn test_add_years() {
        let timestamp = 1_700_000_000i64; // 2023-11-14 22:13:20 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));
        let result = arr.add_years(2).unwrap();

        assert_eq!(result.year().get(0), Some(2025));
        assert_eq!(result.month().get(0), Some(11));
        assert_eq!(result.day().get(0), Some(14));
    }

    // Duration Operations Tests

    #[test]
    fn test_diff() {
        let arr1 = DatetimeArray::<i64>::from_slice(&[1000, 2000, 3000], Some(TimeUnit::Seconds));
        let arr2 = DatetimeArray::<i64>::from_slice(&[1500, 2500, 2500], Some(TimeUnit::Seconds));

        let result = arr1.diff(&arr2, TimeUnit::Seconds).unwrap();

        assert_eq!(result.get(0), Some(-500));
        assert_eq!(result.get(1), Some(-500));
        assert_eq!(result.get(2), Some(500));
    }

    #[test]
    fn test_abs_diff() {
        let arr1 = DatetimeArray::<i64>::from_slice(&[1000, 2000, 3000], Some(TimeUnit::Seconds));
        let arr2 = DatetimeArray::<i64>::from_slice(&[1500, 2500, 2500], Some(TimeUnit::Seconds));

        let result = arr1.abs_diff(&arr2, TimeUnit::Seconds).unwrap();

        assert_eq!(result.get(0), Some(500));
        assert_eq!(result.get(1), Some(500));
        assert_eq!(result.get(2), Some(500));
    }

    #[test]
    fn test_diff_different_units() {
        let arr1 =
            DatetimeArray::<i64>::from_slice(&[1_000_000, 2_000_000], Some(TimeUnit::Milliseconds));
        let arr2 =
            DatetimeArray::<i64>::from_slice(&[1_500_000, 2_500_000], Some(TimeUnit::Milliseconds));

        let result = arr1.diff(&arr2, TimeUnit::Milliseconds).unwrap();

        assert_eq!(result.get(0), Some(-500_000));
        assert_eq!(result.get(1), Some(-500_000));
    }

    // Comparison Operations Tests

    #[test]
    fn test_between() {
        let arr =
            DatetimeArray::<i64>::from_slice(&[1000, 2000, 3000, 4000], Some(TimeUnit::Seconds));
        let start =
            DatetimeArray::<i64>::from_slice(&[1500, 1500, 1500, 1500], Some(TimeUnit::Seconds));
        let end =
            DatetimeArray::<i64>::from_slice(&[3500, 3500, 3500, 3500], Some(TimeUnit::Seconds));

        let result = arr.between(&start, &end).unwrap();

        assert_eq!(result.get(0), Some(false)); // 1000 < 1500
        assert_eq!(result.get(1), Some(true)); // 1500 <= 2000 <= 3500
        assert_eq!(result.get(2), Some(true)); // 1500 <= 3000 <= 3500
        assert_eq!(result.get(3), Some(false)); // 4000 > 3500
    }

    // Component Extraction Tests

    #[test]
    fn test_weekday() {
        // 2023-11-14 is a Tuesday (1=Sunday, 2=Monday, 3=Tuesday, ...)
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        assert_eq!(arr.weekday().get(0), Some(3)); // Tuesday = 3
    }

    #[test]
    fn test_weekday_all_days() {
        // Test all days of the week
        let timestamps = vec![
            1_699_747_200i64, // 2023-11-12 Sunday
            1_699_833_600i64, // 2023-11-13 Monday
            1_699_920_000i64, // 2023-11-14 Tuesday
            1_700_006_400i64, // 2023-11-15 Wednesday
            1_700_092_800i64, // 2023-11-16 Thursday
            1_700_179_200i64, // 2023-11-17 Friday
            1_700_265_600i64, // 2023-11-18 Saturday
        ];
        let arr = DatetimeArray::<i64>::from_slice(&timestamps, Some(TimeUnit::Seconds));
        let weekdays = arr.weekday();

        assert_eq!(weekdays.get(0), Some(1)); // Sunday
        assert_eq!(weekdays.get(1), Some(2)); // Monday
        assert_eq!(weekdays.get(2), Some(3)); // Tuesday
        assert_eq!(weekdays.get(3), Some(4)); // Wednesday
        assert_eq!(weekdays.get(4), Some(5)); // Thursday
        assert_eq!(weekdays.get(5), Some(6)); // Friday
        assert_eq!(weekdays.get(6), Some(7)); // Saturday
    }

    #[test]
    fn test_day_of_year() {
        // 2023-11-14 is the 318th day of the year
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        assert_eq!(arr.day_of_year().get(0), Some(318));
    }

    #[test]
    fn test_iso_week() {
        // 2023-11-14 is in ISO week 46
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        assert_eq!(arr.iso_week().get(0), Some(46));
    }

    #[test]
    fn test_quarter() {
        let timestamps = vec![
            1_672_531_200i64, // 2023-01-01 -> Q1
            1_680_307_200i64, // 2023-04-01 -> Q2
            1_688_169_600i64, // 2023-07-01 -> Q3
            1_696_118_400i64, // 2023-10-01 -> Q4
        ];
        let arr = DatetimeArray::<i64>::from_slice(&timestamps, Some(TimeUnit::Seconds));
        let quarters = arr.quarter();

        assert_eq!(quarters.get(0), Some(1));
        assert_eq!(quarters.get(1), Some(2));
        assert_eq!(quarters.get(2), Some(3));
        assert_eq!(quarters.get(3), Some(4));
    }

    #[test]
    fn test_week_of_year() {
        // 2023-11-14 is in week 46
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        let week = arr.week_of_year().get(0).unwrap();
        assert!(week >= 45 && week <= 47); // Allow some variation in week calculation
    }

    #[test]
    fn test_is_leap_year() {
        let timestamps = vec![
            1_672_531_200i64, // 2023-01-01 -> not leap
            1_704_067_200i64, // 2024-01-01 -> leap year
        ];
        let arr = DatetimeArray::<i64>::from_slice(&timestamps, Some(TimeUnit::Seconds));
        let leap_years = arr.is_leap_year();

        assert_eq!(leap_years.get(0), Some(false));
        assert_eq!(leap_years.get(1), Some(true));
    }

    // Truncation Operations Tests

    #[test]
    fn test_truncate_to_year() {
        let timestamp = 1_700_000_000i64; // 2023-11-14 22:13:20 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        let result = arr.truncate("year").unwrap();

        assert_eq!(result.year().get(0), Some(2023));
        assert_eq!(result.month().get(0), Some(1));
        assert_eq!(result.day().get(0), Some(1));
        assert_eq!(result.hour().get(0), Some(0));
        assert_eq!(result.minute().get(0), Some(0));
        assert_eq!(result.second().get(0), Some(0));
    }

    #[test]
    fn test_truncate_to_month() {
        let timestamp = 1_700_000_000i64; // 2023-11-14 22:13:20 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        let result = arr.truncate("month").unwrap();

        assert_eq!(result.year().get(0), Some(2023));
        assert_eq!(result.month().get(0), Some(11));
        assert_eq!(result.day().get(0), Some(1));
        assert_eq!(result.hour().get(0), Some(0));
    }

    #[test]
    fn test_truncate_to_day() {
        let timestamp = 1_700_000_000i64; // 2023-11-14 22:13:20 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        let result = arr.truncate("day").unwrap();

        assert_eq!(result.hour().get(0), Some(0));
        assert_eq!(result.minute().get(0), Some(0));
        assert_eq!(result.second().get(0), Some(0));
    }

    #[test]
    fn test_truncate_to_hour() {
        let timestamp = 1_700_000_000i64; // 2023-11-14 22:13:20 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        let result = arr.truncate("hour").unwrap();

        assert_eq!(result.hour().get(0), Some(22));
        assert_eq!(result.minute().get(0), Some(0));
        assert_eq!(result.second().get(0), Some(0));
    }

    #[test]
    fn test_truncate_shorthand_methods() {
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        // Test sec(), min(), hr()
        let sec_result = arr.sec();
        let min_result = arr.min();
        let hr_result = arr.hr();

        assert_eq!(sec_result.second().get(0), Some(20));
        assert_eq!(min_result.minute().get(0), Some(13));
        assert_eq!(min_result.second().get(0), Some(0));
        assert_eq!(hr_result.hour().get(0), Some(22));
        assert_eq!(hr_result.minute().get(0), Some(0));
    }

    #[test]
    fn test_truncate_week() {
        // 2023-11-14 is a Tuesday; week should start on Sunday 2023-11-12
        let timestamp = 1_700_000_000i64; // 2023-11-14 22:13:20 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        let result = arr.week();

        assert_eq!(result.day().get(0), Some(12)); // Sunday
        assert_eq!(result.hour().get(0), Some(0));
        assert_eq!(result.minute().get(0), Some(0));
    }

    #[test]
    fn test_truncate_subsecond_us() {
        let arr = DatetimeArray::<i64>::from_slice(&[1_234_567], Some(TimeUnit::Nanoseconds));
        let result = arr.us();

        // Should truncate to nearest microsecond (1000 ns)
        assert_eq!(result.value(0), Some(1_234_000));
    }

    #[test]
    fn test_truncate_subsecond_ms() {
        let arr = DatetimeArray::<i64>::from_slice(&[1_234_567_890], Some(TimeUnit::Nanoseconds));
        let result = arr.ms();

        // Should truncate to nearest millisecond (1_000_000 ns)
        assert_eq!(result.value(0), Some(1_234_000_000));
    }

    // Type Casting Tests

    #[test]
    fn test_cast_time_unit_seconds_to_milliseconds() {
        let arr = DatetimeArray::<i64>::from_slice(&[1, 2, 3], Some(TimeUnit::Seconds));
        let result = arr.cast_time_unit(TimeUnit::Milliseconds).unwrap();

        assert_eq!(result.value(0), Some(1_000));
        assert_eq!(result.value(1), Some(2_000));
        assert_eq!(result.value(2), Some(3_000));
        assert_eq!(result.time_unit, TimeUnit::Milliseconds);
    }

    #[test]
    fn test_cast_time_unit_milliseconds_to_seconds() {
        let arr =
            DatetimeArray::<i64>::from_slice(&[1_000, 2_000, 3_000], Some(TimeUnit::Milliseconds));
        let result = arr.cast_time_unit(TimeUnit::Seconds).unwrap();

        assert_eq!(result.value(0), Some(1));
        assert_eq!(result.value(1), Some(2));
        assert_eq!(result.value(2), Some(3));
        assert_eq!(result.time_unit, TimeUnit::Seconds);
    }

    #[test]
    fn test_cast_time_unit_preserves_datetime() {
        // Cast should preserve the actual datetime, just change representation
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        let as_ms = arr.cast_time_unit(TimeUnit::Milliseconds).unwrap();
        let back_to_sec = as_ms.cast_time_unit(TimeUnit::Seconds).unwrap();

        assert_eq!(back_to_sec.value(0), Some(timestamp));
    }

    // Null Handling and Edge Cases

    #[test]
    fn test_operations_with_nulls() {
        use crate::Bitmask;

        let mask = Bitmask::from_bools(&[true, false, true]);
        let arr = DatetimeArray::new(
            vec64![1000i64, 2000, 3000],
            Some(mask),
            Some(TimeUnit::Seconds),
        );

        let result = arr.add_duration(Duration::seconds(10)).unwrap();

        assert_eq!(result.value(0), Some(1010));
        assert_eq!(result.value(1), None); // null preserved
        assert_eq!(result.value(2), Some(3010));
    }

    #[test]
    fn test_comparison_with_nulls() {
        use crate::Bitmask;

        let mask1 = Bitmask::from_bools(&[true, false, true]);
        let arr1 = DatetimeArray::new(
            vec64![1000i64, 2000, 3000],
            Some(mask1),
            Some(TimeUnit::Seconds),
        );

        let arr2 = DatetimeArray::<i64>::from_slice(&[1500, 2500, 2500], Some(TimeUnit::Seconds));

        let result = arr1.is_before(&arr2).unwrap();

        assert_eq!(result.get(0), Some(true));
        assert_eq!(result.get(1), None); // null input -> null output
        assert_eq!(result.get(2), Some(false));
    }

    #[test]
    fn test_diff_length_mismatch() {
        let arr1 = DatetimeArray::<i64>::from_slice(&[1000, 2000], Some(TimeUnit::Seconds));
        let arr2 = DatetimeArray::<i64>::from_slice(&[1500], Some(TimeUnit::Seconds));

        let result = arr1.diff(&arr2, TimeUnit::Seconds);
        assert!(result.is_err());
    }

    #[test]
    fn test_component_extraction_with_nulls() {
        use crate::Bitmask;

        let mask = Bitmask::from_bools(&[true, false]);
        let arr = DatetimeArray::new(
            vec64![1_700_000_000i64, 1_700_086_400],
            Some(mask),
            Some(TimeUnit::Seconds),
        );

        let years = arr.year();
        assert_eq!(years.get(0), Some(2023));
        assert_eq!(years.get(1), None);
    }

    // Edge Case Tests

    #[test]
    fn test_leap_year_feb_29() {
        // 2024-02-29 00:00:00 UTC (leap year)
        let timestamp = 1_709_164_800i64;
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        assert_eq!(arr.year().get(0), Some(2024));
        assert_eq!(arr.month().get(0), Some(2));
        assert_eq!(arr.day().get(0), Some(29));
        assert_eq!(arr.is_leap_year().get(0), Some(true));
    }

    #[test]
    fn test_add_months_leap_year_feb_29() {
        // 2024-02-29 -> add 1 month -> 2024-03-29
        let timestamp = 1_709_164_800i64; // 2024-02-29 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));
        let result = arr.add_months(1).unwrap();

        assert_eq!(result.year().get(0), Some(2024));
        assert_eq!(result.month().get(0), Some(3));
        assert_eq!(result.day().get(0), Some(29));
    }

    #[test]
    fn test_add_months_leap_year_to_non_leap() {
        // 2024-02-29 -> add 12 months -> 2025-02-28 (clamped, not leap year)
        let timestamp = 1_709_164_800i64; // 2024-02-29 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));
        let result = arr.add_months(12).unwrap();

        assert_eq!(result.year().get(0), Some(2025));
        assert_eq!(result.month().get(0), Some(2));
        assert_eq!(result.day().get(0), Some(28)); // Clamped to Feb 28
        assert_eq!(result.is_leap_year().get(0), Some(false));
    }

    #[test]
    fn test_add_years_leap_year() {
        // 2024-02-29 -> add 4 years -> 2028-02-29 (both leap years)
        let timestamp = 1_709_164_800i64; // 2024-02-29 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));
        let result = arr.add_years(4).unwrap();

        assert_eq!(result.year().get(0), Some(2028));
        assert_eq!(result.month().get(0), Some(2));
        assert_eq!(result.day().get(0), Some(29));
    }

    #[test]
    fn test_add_years_leap_to_non_leap() {
        // 2024-02-29 -> add 1 year -> 2025-02-28 (clamped)
        let timestamp = 1_709_164_800i64; // 2024-02-29 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));
        let result = arr.add_years(1).unwrap();

        assert_eq!(result.year().get(0), Some(2025));
        assert_eq!(result.month().get(0), Some(2));
        assert_eq!(result.day().get(0), Some(28)); // Clamped
    }

    #[test]
    fn test_year_boundary_dec_31_to_jan_1() {
        // 2023-12-31 -> add 1 day -> 2024-01-01
        let timestamp = 1_704_067_199i64; // 2023-12-31 23:59:59 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));
        let result = arr.add_days(1).unwrap();

        assert_eq!(result.year().get(0), Some(2024));
        assert_eq!(result.month().get(0), Some(1));
        assert_eq!(result.day().get(0), Some(1));
    }

    #[test]
    fn test_month_boundary_jan_31_to_feb_1() {
        // 2024-01-31 -> add 1 day -> 2024-02-01
        let timestamp = 1_706_659_200i64; // 2024-01-31 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));
        let result = arr.add_days(1).unwrap();

        assert_eq!(result.year().get(0), Some(2024));
        assert_eq!(result.month().get(0), Some(2));
        assert_eq!(result.day().get(0), Some(1));
    }

    #[test]
    fn test_negative_months() {
        // 2023-03-15 -> subtract 3 months -> 2022-12-15
        let timestamp = 1_678_838_400i64; // 2023-03-15 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));
        let result = arr.add_months(-3).unwrap();

        assert_eq!(result.year().get(0), Some(2022));
        assert_eq!(result.month().get(0), Some(12));
        assert_eq!(result.day().get(0), Some(15));
    }

    #[test]
    fn test_negative_years() {
        // 2023-11-14 -> subtract 5 years -> 2018-11-14
        let timestamp = 1_700_000_000i64;
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));
        let result = arr.add_years(-5).unwrap();

        assert_eq!(result.year().get(0), Some(2018));
        assert_eq!(result.month().get(0), Some(11));
        assert_eq!(result.day().get(0), Some(14));
    }

    #[test]
    fn test_century_leap_year_2000() {
        // 2000 is a leap year (divisible by 400)
        let timestamp = 951_868_800i64; // 2000-03-01 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));
        let prev_day = arr.add_days(-1).unwrap();

        assert_eq!(prev_day.year().get(0), Some(2000));
        assert_eq!(prev_day.month().get(0), Some(2));
        assert_eq!(prev_day.day().get(0), Some(29)); // Feb 29 exists
    }

    #[test]
    fn test_century_non_leap_year_1900() {
        // 1900 was not a leap year (divisible by 100 but not 400)
        // March 1, 1900 minus 1 day should be Feb 28, not Feb 29
        let timestamp = -2_203_891_200i64; // 1900-03-01 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));
        let prev_day = arr.add_days(-1).unwrap();

        assert_eq!(prev_day.year().get(0), Some(1900));
        assert_eq!(prev_day.month().get(0), Some(2));
        assert_eq!(prev_day.day().get(0), Some(28)); // Feb 28, not 29
    }

    #[test]
    fn test_extreme_timestamp_min() {
        // Test with a very old date (year 1)
        let timestamp = -62_135_596_800i64; // 0001-01-01 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        assert_eq!(arr.year().get(0), Some(1));
        assert_eq!(arr.month().get(0), Some(1));
        assert_eq!(arr.day().get(0), Some(1));
    }

    #[test]
    fn test_extreme_timestamp_far_future() {
        // Test with a far future date (year 3000)
        let timestamp = 32_503_680_000i64; // 3000-01-01 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        assert_eq!(arr.year().get(0), Some(3000));
        assert_eq!(arr.month().get(0), Some(1));
        assert_eq!(arr.day().get(0), Some(1));
    }

    #[test]
    fn test_different_time_units_precision() {
        // Test that different time units maintain precision correctly
        let arr_sec = DatetimeArray::<i64>::from_slice(&[1_700_000_000], Some(TimeUnit::Seconds));
        let arr_ms = arr_sec.cast_time_unit(TimeUnit::Milliseconds).unwrap();
        let arr_us = arr_ms.cast_time_unit(TimeUnit::Microseconds).unwrap();
        let arr_ns = arr_us.cast_time_unit(TimeUnit::Nanoseconds).unwrap();

        // Convert back to seconds
        let back_to_sec = arr_ns.cast_time_unit(TimeUnit::Seconds).unwrap();

        assert_eq!(back_to_sec.value(0), Some(1_700_000_000));
    }

    #[test]
    fn test_week_start_on_sunday() {
        // 2023-11-12 is Sunday, week should start on itself
        let timestamp = 1_699_747_200i64; // 2023-11-12 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        let result = arr.week();

        assert_eq!(result.day().get(0), Some(12)); // Same day (Sunday)
        assert_eq!(result.hour().get(0), Some(0));
    }

    #[test]
    fn test_week_start_on_saturday() {
        // 2023-11-18 is Saturday, week should start on Sunday 2023-11-12
        let timestamp = 1_700_265_600i64; // 2023-11-18 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[timestamp], Some(TimeUnit::Seconds));

        let result = arr.week();

        assert_eq!(result.day().get(0), Some(12)); // Previous Sunday
    }

    #[test]
    fn test_add_months_varying_lengths() {
        // Test adding months across months of different lengths
        // Jan (31) -> Feb (28/29) -> Mar (31) -> Apr (30)
        let jan_31 = 1_675_123_200i64; // 2023-01-31 00:00:00 UTC
        let arr = DatetimeArray::<i64>::from_slice(&[jan_31], Some(TimeUnit::Seconds));

        // Jan 31 + 1 month = Feb 28 (clamped)
        let feb = arr.add_months(1).unwrap();
        assert_eq!(feb.month().get(0), Some(2));
        assert_eq!(feb.day().get(0), Some(28));

        // Feb 28 + 1 month = Mar 28
        let mar = feb.add_months(1).unwrap();
        assert_eq!(mar.month().get(0), Some(3));
        assert_eq!(mar.day().get(0), Some(28));
    }
}
