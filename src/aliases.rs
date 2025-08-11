use crate::{
    Array, Bitmask, BooleanArray, CategoricalArray, Field, FieldArray, FloatArray, IntegerArray, StringArray, Table
};
#[cfg(feature = "cube")]
use crate::TableV;
#[cfg(feature = "datetime")]
use crate::DatetimeArray;

#[cfg(feature = "chunked")]
use crate::SuperTable;

/// Standard Arrow `Record Batch`
///
/// # Description
/// - Standard columnar table batch with named columns (`FieldArray`),
/// a fixed number of rows, and an optional logical table name.
/// - All columns are required to be equal length and have consistent schema.
/// - Supports zero-copy slicing, efficient iteration, and bulk operations.
/// - Equivalent to the `RecordBatch` in *Apache Arrow*.
/// 
/// # Structure
/// - `cols`: A vector of `FieldArray`, each representing a column with metadata and data.
/// - `n_rows`: The logical number of rows (guaranteed equal for all columns).
/// - `name`: Optional logical name or alias for this table instance.
///
/// # Usage
/// - Use `Table` as a general-purpose, in-memory columnar data container.
/// - Good for analytics, transformation pipelines, and FFI or Arrow interoperability.
/// - For batched/partitioned tables, see [`ChunkedTable`] or windowed/chunked abstractions.
///
/// # Notes
/// - Table instances are typically lightweight to clone and pass by value.
/// - For mutation, construct a new table or replace individual columns as needed.
/// 
/// # Example
/// ```rust
/// use minarrow::{arr_i32, arr_str32, FieldArray, Print, vec64, aliases::RecordBatch};
///
/// let col1 = FieldArray::from_inner("numbers", arr_i32![1, 2, 3]);
/// let col2 = FieldArray::from_inner("letters", arr_str32!["x", "y", "z"]);
///
/// let mut tbl = RecordBatch::new("Demo".into(), vec![col1, col2].into());
/// tbl.print();
/// ```
pub type RecordBatch = Table;

/// Batched (windowed/chunked) table - collection of `Tables`.
///
/// ### Data structure
/// Each Table represents a record batch with schema consistency enforced.
/// Windows/row-chunks are tracked as Vec<Table>.
///
/// - `batches`: Ordered record batches.
/// - `schema`: schema of the first batch, cached for fast access.
/// - `n_rows`: total number of rows across all batches.
/// - `name`: logical group name.
///
/// ### Use cases
/// Useful for cases such as :
///     1. Streaming *(and mini-batch processing)*
///     2. Reading from multiple memory mapped arrow files from disk
///     3. In-memory analytics, where chunks can be used as a source for
///     parallelism, or windowing, etc., depending on use case semantics.
#[cfg(feature = "chunked")]
pub type ChunkedTable = SuperTable;

// ----------------- Array Views --------------------------------
//
// Combined zero-copy views, and/or windows that facilitate
// slicing only the required array portions that hold onto the
// parameters to support reconstruction.
//
// -----------------------------------------------------------------

/// The `ArrayView` offset lower bound, for a windowed view.
/// Set to `0` for the whole set.
pub type Offset = usize;

/// The logical length of the `ArrayView`.
/// Set to `arr.len()` for the whole set.
pub type Length = usize;

/// Dictionary field length for a `DictionaryArray`
pub type DictLength = usize;

/// Raw bytes data length for a `StringArray`
///
/// Physical length, rather than logical offsets.\
/// Useful when keying back into it downstream.
pub type BytesLength = usize;

// Top-level type

/// Windowed ***V**iew **T**uple* for Array, when one isn't using the 
/// full `ArrayView` abstraction, doesn't want to use it, couple a
/// function signature to it or doesn't have that feature enabled.
pub type ArrayVT<'a> = (&'a Array, Offset, Length);

/// Windowed ***V**iew **T**uple* for Bitmask
pub type BitmaskVT<'a> = (&'a Bitmask, Offset, Length);

/// Subset per respective table within the cube
/// 
/// Respects the means in which each table is windowed,
/// e.g., if offsets and lengths are different due to category lengths,
/// time windows etc.
#[cfg(feature = "cube")]
pub type CubeV = Vec<TableV>;

// Low-level types

// Also see `crate::bitmask_view::BitmaskView`

/// Logical windowed ***A**rray **V**iew **T**uple* over a Boolean array.
pub type BooleanAVT<'a, T> = (&'a BooleanArray<T>, Offset, Length);

/// Logical windowed ***A**rray **V**iew **T**uple* over a categorical array.
pub type CategoricalAVT<'a, T> = (&'a CategoricalArray<T>, Offset, Length);

/// Logical windowed ***A**rray **V**iew **T**uple* over a categorical array, including dictionary length.
pub type CategoricalAVTExt<'a, T> = (&'a CategoricalArray<T>, Offset, Length, DictLength);

/// Logical windowed ***A**rray **V**iew **T**uple* over a UTF-8 string array.
pub type StringAVT<'a, T> = (&'a StringArray<T>, Offset, Length);

/// Logical windowed ***A**rray **V**iew **T**uple* over a UTF-8 string array, including byte length.
pub type StringAVTExt<'a, T> = (&'a StringArray<T>, Offset, Length, BytesLength);

/// Logical windowed ***A**rray **V**iew **T**uple* over a datetime array.
#[cfg(feature = "datetime")]
pub type DatetimeAVT<'a, T> = (&'a DatetimeArray<T>, Offset, Length);

// When working with Float and Integer, `as_slice` or `slice_tuple` off those arrays is already sufficient,
// and the recommended pattern. Slice &[T] already pre-slices to a 1:1 index <-> memory layout.
// Regardless, they are included below for completeness.
//
// The others are different, because they include context on the source object that's lost when
// moving to slice, and/or hold different physical vs. logical layouts.

/// Logical windowed ***A**rray **V**iew **T**uple* over a primitive integer array.
pub type IntegerAVT<'a, T> = (&'a IntegerArray<T>, Offset, Length);

/// Logical windowed ***A**rray **V**iew **T**uple* over a primitive float array.
pub type FloatAVT<'a, T> = (&'a FloatArray<T>, Offset, Length);

/// Logical windowed ***A**rray **V**iew **T**uple* over an `Array` and its `Field`: *((array, offset, len), field)*.
///
/// Available if desired, but it's recommended to avoid due to reduced clarity and ergonomics (e.g., `.0.1` access).\
/// In many cases, there are cleaner ways to retain a `Field` for reconstruction without coupling or polluting downstream APIs.
pub type FieldAVT<'a> = ((&'a Array, Offset, Length), &'a Field);

// ----------------- Standard Aliases --------------------------------

// Less syllables

pub type IntArr<T> = IntegerArray<T>;
pub type FltArr<T> = FloatArray<T>;
pub type StrArr<T> = StringArray<T>;
pub type CatArr<T> = CategoricalArray<T>;
#[cfg(feature = "datetime")]
pub type DtArr<T> = DatetimeArray<T>;
pub type BoolArr = BooleanArray<()>;
pub type FA = FieldArray;