use std::fmt::{Display, Formatter, Result as FmtResult};

/// Unified time unit enumeration.
///
/// # Purpose
/// - Combines time units for both `DatetimeArray32` and `DatetimeArray64`.
/// - Confirm time since epoch units, or a raw duration value *(depending on the `ArrowType`
/// that's attached to `Field` during `FieldArray` construction)*.
/// - Avoids proliferating variants that require explicit handling throughout match statements.
/// 
/// # Behaviour
/// - Unit values are stored on the `DatetimeArray`, enabling variant-specific logic.
/// - When transmitted over FFI, an `Apache Arrow`- produces compatible native format.
#[derive(PartialEq, Clone, Debug, Default)]
pub enum TimeUnit {
    /// Seconds for Apache Arrow `Time32` and `Time64` units.
    Seconds,
    /// Milliseconds for Apache Arrow `Time32` and `Time64` units.
    Milliseconds,
    /// Microseconds for Apache Arrow `Time32` and `Time64` units.
    Microseconds,
    /// Nanoseconds for Apache Arrow `Time32` and `Time64` units.
    Nanoseconds,
    /// Default = days unspecified
    ///
    /// Apache Arrow's `Date32` and `Date64` types use days implicitly.
    #[default]
    Days
}

/// Inner Arrow discriminant for representing interval types
/// 
/// Attach via `ArrowType` to `Field` when your `DatetimeArray<T>`
/// T-integer represents an interval, rather than an epoch value.
/// Then, it will materialise as an `Interval` *Apache Arrow* type
/// when sent over FFI.
#[derive(PartialEq, Clone, Debug)]
pub enum IntervalUnit {
    YearMonth,
    DaysTime,
    MonthDaysNs
}

impl Display for TimeUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            TimeUnit::Seconds => f.write_str("Seconds"),
            TimeUnit::Milliseconds => f.write_str("Milliseconds"),
            TimeUnit::Microseconds => f.write_str("Microseconds"),
            TimeUnit::Nanoseconds => f.write_str("Nanoseconds"),
            TimeUnit::Days => f.write_str("Days")
        }
    }
}

impl Display for IntervalUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            IntervalUnit::YearMonth => f.write_str("YearMonth"),
            IntervalUnit::DaysTime => f.write_str("DaysTime"),
            IntervalUnit::MonthDaysNs => f.write_str("MonthDaysNs")
        }
    }
}
