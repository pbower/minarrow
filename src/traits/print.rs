//! # **Print Module** - *Pretty Printing with Attitude*
//! 
//! Contains implementations of the Display trait
//! and an additional `Print` trait which wraps it to provide
//! `myobj.print()` for any object that implements it.
use std::fmt::{self, Display, Formatter};

#[cfg(feature = "datetime")]
use crate::{DatetimeArray, Integer, TemporalArray};
use crate::{Array, Buffer,  Float, NumericArray, TextArray};

pub (crate) const MAX_PREVIEW: usize = 50;

/// # Print
/// 
/// Loaded print trait for pretty printing tables
/// 
/// Provides a more convenient way to activate `Display`
/// for other types such as arrays via `myarr.print()`,
/// avoiding the need to write `println!("{}", myarr);`
pub trait Print {
    #[inline]
    fn print(&self)
    where
        Self: Display
    {
        println!("{}", self);
    }
}

impl<T: Display> Print for T where T: Display {}

// Helper functions

pub (crate) fn value_to_string(arr: &Array, idx: usize) -> String {
    // Null checks (handles absent mask too)
    if let Some(mask) = arr.null_mask() {
        if !mask.get(idx) {
            return "null".into();
        }
    }
    match arr {
        // ------------------------- numeric ------------------------------
        Array::NumericArray(inner) => match inner {
            NumericArray::Int32(a) => a.data[idx].to_string(),
            NumericArray::Int64(a) => a.data[idx].to_string(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int8(a) => a.data[idx].to_string(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::Int16(a) => a.data[idx].to_string(),
            NumericArray::UInt32(a) => a.data[idx].to_string(),
            NumericArray::UInt64(a) => a.data[idx].to_string(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt8(a) => a.data[idx].to_string(),
            #[cfg(feature = "extended_numeric_types")]
            NumericArray::UInt16(a) => a.data[idx].to_string(),
            NumericArray::Float32(a) => format_float(a.data[idx] as f64),
            NumericArray::Float64(a) => format_float(a.data[idx]),
            NumericArray::Null => "null".into()
        },
        // ------------------------- boolean ------------------------------
        Array::BooleanArray(b) => {
            let bit = b.data.get(idx);
            bit.to_string()
        }
        // ------------------------- string / categorical -----------------
        Array::TextArray(inner) => match inner {
            TextArray::String32(s) => string_value(&s.offsets, &s.data, idx),
            #[cfg(feature = "large_string")]
            TextArray::String64(s) => string_value(&s.offsets, &s.data, idx),
            TextArray::Categorical32(cat) => {
                let key = cat.data[idx] as usize;
                cat.unique_values[key].clone()
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical8(cat) => {
                let key = cat.data[idx] as usize;
                cat.unique_values[key].clone()
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical16(cat) => {
                let key = cat.data[idx] as usize;
                cat.unique_values[key].clone()
            }
            #[cfg(feature = "extended_categorical")]
            TextArray::Categorical64(cat) => {
                let key = cat.data[idx] as usize;
                cat.unique_values[key].clone()
            }
            TextArray::Null => "null".into()
        },
        // ------------------------- datetime -----------------------------
        #[cfg(feature = "datetime")]
        Array::TemporalArray(inner) => match inner {
            TemporalArray::Datetime32(dt) => format_datetime_value(dt, idx),
            TemporalArray::Datetime64(dt) => format_datetime_value(dt, idx),
            TemporalArray::Null => "null".into()
        },
        // ------------------------- fallback -----------------------------
        Array::Null => "null".into()
    }
}

fn string_value<T: Copy>(
    offsets: &Buffer<T>,
    data: &Buffer<u8>,
    idx: usize
) -> String
where
    T: Copy + Into<u64>
{
    // Convert to u64, then to usize (explicitly)
    let start = offsets[idx].into() as usize;
    let end = offsets[idx + 1].into() as usize;
    let slice = &data[start..end];

    // Safety: Arrow guarantees valid UTF-8 encoding
    let s = unsafe { std::str::from_utf8_unchecked(slice) };
    s.to_string()
}

pub (crate) fn print_rule(f: &mut Formatter<'_>, idx_width: usize, col_widths: &[usize]) -> fmt::Result {
    write!(f, "+{:-<w$}+", "", w = idx_width + 2)?; // idx column (+2 for spaces)
    for &w in col_widths {
        write!(f, "{:-<w$}+", "", w = w + 2)?; // +2 for spaces
    }
    writeln!(f)
}

pub (crate) fn print_header_row(
    f: &mut Formatter<'_>,
    idx_width: usize,
    headers: &[String],
    col_widths: &[usize]
) -> fmt::Result {
    write!(f, "| {hdr:^w$} |", hdr = "idx", w = idx_width)?;
    for (hdr, &w) in headers.iter().zip(col_widths) {
        write!(f, " {hdr:^w$} |", hdr = hdr, w = w)?;
    }
    writeln!(f)
}

pub (crate) fn print_ellipsis_row(
    f: &mut Formatter<'_>,
    idx_width: usize,
    col_widths: &[usize]
) -> fmt::Result {
    write!(f, "| {dots:^w$} |", dots = "…", w = idx_width)?;
    for &w in col_widths {
        write!(f, " {dots:^w$} |", dots = "…", w = w)?;
    }
    writeln!(f)
}

/// Formats floating point numbers:
/// - Keeps up to 6 decimal digits
/// - Trims trailing zeroes and unnecessary decimal point
#[inline]
pub (crate) fn format_float<T: Float + Display>(v: T) -> String {
    let s = format!("{:.6}", v);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

#[cfg(feature = "datetime")]
fn format_datetime_value<T>(arr: &DatetimeArray<T>, idx: usize) -> String
where
    T: Integer + std::fmt::Display,
{
    // Null check

    use crate::MaskedArray;
    if arr.is_null(idx) {
        return "null".into();
    }

    #[cfg(feature = "chrono")]
    {
        use chrono::{DateTime, NaiveDate, Utc};

        use crate::TimeUnit;

        match arr.time_unit {
            TimeUnit::Seconds => {
                let secs = arr.data[idx].to_i64().unwrap();
                DateTime::<Utc>::from_timestamp(secs, 0)
                    .map(|dt| dt.to_string())
                    .unwrap_or_else(|| format!("{secs}s"))
            }
            TimeUnit::Milliseconds => {
                let v = arr.data[idx].to_i64().unwrap();
                let secs = v / 1_000;
                let nsecs = ((v % 1_000) * 1_000_000) as u32;
                DateTime::<Utc>::from_timestamp(secs, nsecs)
                    .map(|dt| dt.to_string())
                    .unwrap_or_else(|| format!("{v}ms"))
            }
            TimeUnit::Microseconds => {
                let v = arr.data[idx].to_i64().unwrap();
                let secs = v / 1_000_000;
                let nsecs = ((v % 1_000_000) * 1_000) as u32;
                DateTime::<Utc>::from_timestamp(secs, nsecs)
                    .map(|dt| dt.to_string())
                    .unwrap_or_else(|| format!("{v}µs"))
            }
            TimeUnit::Nanoseconds => {
                let v = arr.data[idx].to_i64().unwrap();
                let secs = v / 1_000_000_000;
                let nsecs = (v % 1_000_000_000) as u32;
                DateTime::<Utc>::from_timestamp(secs, nsecs)
                    .map(|dt| dt.to_string())
                    .unwrap_or_else(|| format!("{v}ns"))
            }
            TimeUnit::Days => {
                let days = arr.data[idx].to_i64().unwrap();
                let base = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                base.checked_add_signed(chrono::Duration::days(days))
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| format!("{days}d"))
            }
        }
    }
    #[cfg(not(feature = "chrono"))]
    {
        use crate::TimeUnit;

        let v = arr.data[idx];
        let suffix = match arr.time_unit {
            TimeUnit::Seconds => "s",
            TimeUnit::Milliseconds => "ms",
            TimeUnit::Microseconds => "µs",
            TimeUnit::Nanoseconds => "ns",
            TimeUnit::Days => "d",
        };
        format!("{v}{suffix}")
    }
}
