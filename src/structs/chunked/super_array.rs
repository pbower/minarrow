//! # **SuperArray** - *Holds multiple arrays for chunked data partitioning, streaming + fast memIO*
//!
//! Contains SuperArray, a higher-order container representing a logical column split into multiple immutable `FieldArray` chunks.
//!
//! ## Overview
//! - Equivalent to Apache Arrow's `ChunkedArray`.
//! - Stores an ordered list of `FieldArray` segments, each with identical field metadata.
//! - Chunk lengths may vary.
//! - A solid fit for append-only patterns, partitioned storage, and streaming data ingestion.

use std::fmt::{Display, Formatter};
use std::iter::FromIterator;
#[cfg(feature = "views")]
use std::sync::Arc;

#[cfg(feature = "views")]
use crate::ArrayV;
#[cfg(feature = "views")]
use crate::SuperArrayV;
use crate::enums::{error::MinarrowError, shape_dim::ShapeDim};
use crate::ffi::arrow_dtype::ArrowType;
#[cfg(feature = "size")]
use crate::traits::byte_size::ByteSize;
use crate::traits::{concatenate::Concatenate, shape::Shape};
use crate::{Array, Field, FieldArray};

/// Strategy for rechunking arrays and tables.
///
/// Defines how to redistribute data across chunks/batches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RechunkStrategy {
    /// Rechunk into uniform chunks of the specified element/row count.
    Count(usize),
    /// Rechunk targeting a specific memory size per chunk in bytes.
    #[cfg(feature = "size")]
    Memory(usize),
    /// Rechunk using a default size of 8192 elements/rows.
    Auto,
}

/// # SuperArray
///
/// Higher-order container for multiple immutable `FieldArray` segments.
///
/// ## Description
/// - Stores an ordered sequence of `FieldArray` chunks, each with identical field metadata.
/// - Equivalent to Apache Arrow’s `ChunkedArray` when sent over FFI, where it is treated
///   as a single logical column.
/// - It can also serve as an unbounded or continuously growing
///   collection of segments, making it useful for streaming ingestion and partitioned storage.
/// - Chunk lengths may vary without restriction.
///
/// ## Example
/// ```ignore
/// // Create from multiple chunks with matching metadata
/// let sa = SuperArray::from_chunks(vec![fa("col", &[1, 2], 0), fa("col", &[3], 0)]);
///
/// assert_eq!(sa.len(), 3);         // total rows across chunks
/// assert_eq!(sa.n_chunks(), 2);    // number of chunks
/// assert_eq!(sa.field().name, "col");
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SuperArray {
    arrays: Vec<FieldArray>,
}

impl SuperArray {
    // Constructors

    /// Constructs an empty ChunkedArray.
    #[inline]
    pub fn new() -> Self {
        Self { arrays: Vec::new() }
    }

    /// Constructs a ChunkedArray from `FieldArray` chunks.
    /// Panics if chunks is empty or metadata/type/nullable mismatch is found.
    pub fn from_field_array_chunks(chunks: Vec<FieldArray>) -> Self {
        assert!(
            !chunks.is_empty(),
            "from_field_array_chunks: input chunks cannot be empty"
        );
        let field = &chunks[0].field;
        for (i, fa) in chunks.iter().enumerate().skip(1) {
            assert_eq!(
                fa.field.dtype, field.dtype,
                "Chunk {i} ArrowType mismatch (expected {:?}, got {:?})",
                field.dtype, fa.field.dtype
            );
            assert_eq!(
                fa.field.nullable, field.nullable,
                "Chunk {i} nullability mismatch"
            );
            assert_eq!(
                fa.field.name, field.name,
                "Chunk {i} field name mismatch (expected '{}', got '{}')",
                field.name, fa.field.name
            );
        }
        Self { arrays: chunks }
    }

    /// Construct from `Vec<FieldArray>`.
    pub fn from_chunks(chunks: Vec<FieldArray>) -> Self {
        Self::from_field_array_chunks(chunks)
    }

    /// Materialises a `ChunkedArray` from an existing slice of `ArrayView` tuples,
    /// using the provided field metadata (applied to all slices).
    ///
    /// Panics if the slice list is empty, or if any slice's type or nullability
    /// does not match the provided field.
    #[cfg(feature = "views")]
    pub fn from_slices(slices: &[ArrayV], field: Arc<Field>) -> Self {
        assert!(!slices.is_empty(), "from_slices requires non-empty slice");

        let mut out = Vec::with_capacity(slices.len());
        for (i, view) in slices.iter().enumerate() {
            assert_eq!(
                view.array.arrow_type(),
                field.dtype,
                "Slice {i} ArrowType does not match field"
            );
            assert_eq!(
                view.array.is_nullable(),
                field.nullable,
                "Slice {i} nullability does not match field"
            );
            out.push(FieldArray {
                field: field.clone(),
                array: view.array.slice_clone(view.offset, view.len()),
                null_count: view.null_count(),
            });
        }

        Self { arrays: out }
    }

    /// Returns a zero-copy view of this chunked array over the window `[offset..offset+len)`.
    ///
    /// If the chunks are fragmented in memory, access patterns may result in
    /// degraded cache locality and reduced SIMD optimisation.
    #[cfg(feature = "views")]
    pub fn slice(&self, offset: usize, len: usize) -> SuperArrayV {
        assert!(offset + len <= self.len(), "slice out of bounds");

        let mut remaining = len;
        let mut off = offset;
        let mut slices = Vec::new();
        let field = self.field().clone();

        for fa in &self.arrays {
            let this_len = fa.len();
            if off >= this_len {
                off -= this_len;
                continue;
            }

            let take = remaining.min(this_len - off);
            slices.push(ArrayV::new(fa.array.clone(), off, take));
            remaining -= take;

            if remaining == 0 {
                break;
            }
            off = 0;
        }

        SuperArrayV {
            slices,
            len,
            field: field.into(),
        }
    }

    // Concatenation

    /// Materialises a contiguous `Array` holding all rows.
    pub fn copy_to_array(&self) -> Array {
        assert!(
            !self.arrays.is_empty(),
            "to_array() called on empty ChunkedArray"
        );

        // Use the Concatenate trait to combine all FieldArrays
        let result = self
            .arrays
            .iter()
            .cloned()
            .reduce(|acc, arr| acc.concat(arr).expect("Failed to concatenate arrays"))
            .expect("Expected at least one array");

        result.array
    }

    // Metadata

    /// Returns the field metadata from the first chunk (guaranteed by constructor).
    #[inline]
    pub fn field(&self) -> &Field {
        &self.arrays[0].field
    }

    /// Returns the Arrow physical type.
    #[inline]
    pub fn arrow_type(&self) -> ArrowType {
        self.arrays[0].field.dtype.clone()
    }

    /// Returns the nullability flag.
    #[inline]
    pub fn is_nullable(&self) -> bool {
        self.arrays[0].field.nullable
    }

    /// Returns the number of logical chunks.
    #[inline]
    pub fn n_chunks(&self) -> usize {
        self.arrays.len()
    }

    /// Returns total logical length (sum of all chunk lengths).
    pub fn len(&self) -> usize {
        self.arrays.iter().map(|c| c.len()).sum()
    }

    /// Returns true if the array has no chunks or all chunks are empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_chunks() == 0 || self.len() == 0
    }

    // Chunk Access

    /// Returns a read-only reference to all underlying chunks.
    #[inline]
    pub fn chunks(&self) -> &[FieldArray] {
        &self.arrays
    }
    /// Returns a mutable reference to the underlying chunks.
    #[inline]
    pub fn chunks_mut(&mut self) -> &mut [FieldArray] {
        &mut self.arrays
    }

    /// Returns a reference to a specific chunk, if it exists.
    #[inline]
    pub fn chunk(&self, idx: usize) -> Option<&FieldArray> {
        self.arrays.get(idx)
    }

    // Mutation

    /// Validates and appends a new chunk.
    ///
    /// # Panics
    /// If the chunk does not match the expected type or nullability.
    pub fn push(&mut self, chunk: FieldArray) {
        if self.arrays.is_empty() {
            self.arrays.push(chunk);
        } else {
            let f = &self.arrays[0].field;
            assert_eq!(chunk.field.dtype, f.dtype, "Chunk ArrowType mismatch");
            assert_eq!(
                chunk.field.nullable, f.nullable,
                "Chunk nullability mismatch"
            );
            assert_eq!(chunk.field.name, f.name, "Chunk field name mismatch");
            self.arrays.push(chunk);
        }
    }

    /// Inserts rows from another SuperArray (or Array) at the specified index.
    ///
    /// This is an **O(n)** operation where n is the number of elements in the chunk
    /// containing the insertion point.
    ///
    /// # Arguments
    /// * `index` - Global row position before which to insert (0 = prepend, len() = append)
    /// * `other` - SuperArray or Array to insert (via `Into<SuperArray>`)
    ///
    /// # Requirements
    /// - Field metadata (name, type, nullability) must match
    /// - `index` must be <= `self.len()`
    ///
    /// # Strategy
    /// Finds the chunk containing the insertion point and inserts all of `other`'s data
    /// into that chunk. This may cause the target chunk to grow significantly.
    ///
    /// # Errors
    /// - `IndexError` if index > len()
    /// - Schema mismatch if field metadata doesn't match
    pub fn insert_rows(
        &mut self,
        index: usize,
        other: impl Into<SuperArray>,
    ) -> Result<(), MinarrowError> {
        let other = other.into();
        let total_len = self.len();

        // Validate index
        if index > total_len {
            return Err(MinarrowError::IndexError(format!(
                "Index {} out of bounds for SuperArray of length {}",
                index, total_len
            )));
        }

        // If other is empty, nothing to do
        if other.is_empty() {
            return Ok(());
        }

        // Validate schema match
        if !self.arrays.is_empty() {
            let self_field = &self.arrays[0].field;
            let other_field = &other.arrays[0].field;

            if self_field.name != other_field.name {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "SuperArray",
                    to: "SuperArray",
                    message: Some(format!(
                        "Field name mismatch: '{}' vs '{}'",
                        self_field.name, other_field.name
                    )),
                });
            }

            if self_field.dtype != other_field.dtype {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "SuperArray",
                    to: "SuperArray",
                    message: Some(format!(
                        "Type mismatch: {:?} vs {:?}",
                        self_field.dtype, other_field.dtype
                    )),
                });
            }

            if self_field.nullable != other_field.nullable {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "SuperArray",
                    to: "SuperArray",
                    message: Some(format!(
                        "Nullable mismatch: {} vs {}",
                        self_field.nullable, other_field.nullable
                    )),
                });
            }
        }

        // Handle empty self - just append other's chunks
        if self.arrays.is_empty() {
            self.arrays = other.arrays;
            return Ok(());
        }

        // Find which chunk contains the insertion index
        let mut cumulative = 0;
        let mut chunk_idx = None;
        for (i, chunk) in self.arrays.iter().enumerate() {
            let chunk_len = chunk.len();

            // Check if index falls within this chunk or at its boundary
            if index <= cumulative + chunk_len {
                chunk_idx = Some((i, index - cumulative));
                break;
            }

            cumulative += chunk_len;
        }

        let (target_idx, local_index) = chunk_idx.ok_or_else(|| {
            MinarrowError::IndexError(format!("Failed to find chunk for index {}", index))
        })?;

        let target_chunk_len = self.arrays[target_idx].len();

        // Handle edge cases: prepend or append to a chunk without splitting
        if local_index == 0 {
            // Insert before target chunk
            let mut new_chunks = Vec::with_capacity(self.arrays.len() + other.arrays.len());
            new_chunks.extend(self.arrays.drain(0..target_idx));
            new_chunks.extend(other.arrays.into_iter());
            new_chunks.extend(self.arrays.drain(..));
            self.arrays = new_chunks;
        } else if local_index == target_chunk_len {
            // Insert after target chunk
            let mut new_chunks = Vec::with_capacity(self.arrays.len() + other.arrays.len());
            new_chunks.extend(self.arrays.drain(0..=target_idx));
            new_chunks.extend(other.arrays.into_iter());
            new_chunks.extend(self.arrays.drain(..));
            self.arrays = new_chunks;
        } else {
            // Split the target chunk at the insertion point
            let target_chunk = self.arrays.remove(target_idx);
            let mut split_chunks = target_chunk.array.split(local_index, &target_chunk.field)?;

            // Build new chunk list: left chunk + other's chunks + right chunk
            let mut new_chunks = Vec::with_capacity(self.arrays.len() + other.arrays.len() + 2);

            // Add chunks before target
            new_chunks.extend(self.arrays.drain(0..target_idx));

            // Add left half of split
            new_chunks.extend(split_chunks.arrays.drain(0..1));

            // Add other's chunks
            new_chunks.extend(other.arrays.into_iter());

            // Add right half of split
            new_chunks.extend(split_chunks.arrays.drain(..));

            // Add remaining chunks after target
            new_chunks.extend(self.arrays.drain(..));

            self.arrays = new_chunks;
        }

        Ok(())
    }

    /// Rechunks the array according to the specified strategy.
    ///
    /// Redistributes data across chunks using an efficient incremental approach
    /// that avoids full materialization:
    /// - `Count(n)`: Creates chunks of `n` elements. The last chunk may be smaller.
    /// - `Auto`: Uses a default size of 8192 elements
    /// - `Memory(bytes)`: Targets a specific memory size per chunk
    ///
    /// # Arguments
    /// * `strategy` - The rechunking strategy to use
    ///
    /// # Errors
    /// - Returns `IndexError` if `Count(0)` is specified
    /// - Returns `IndexError` if memory-based calculation results in 0 chunk size
    ///
    /// # Example
    /// ```ignore
    /// // Rechunk into 1024-element chunks
    /// array.rechunk(RechunkStrategy::Count(1024))?;
    ///
    /// // Rechunk with default size
    /// array.rechunk(RechunkStrategy::Auto)?;
    ///
    /// // Target 64KB per chunk
    /// array.rechunk(RechunkStrategy::Memory(65536))?;
    /// ```
    pub fn rechunk(&mut self, strategy: RechunkStrategy) -> Result<(), MinarrowError> {
        if self.arrays.is_empty() || self.len() == 0 {
            return Ok(());
        }

        // Determine chunk size based on strategy
        let chunk_size = match strategy {
            RechunkStrategy::Count(size) => {
                if size == 0 {
                    return Err(MinarrowError::IndexError(
                        "Count chunk size must be greater than 0".to_string(),
                    ));
                }
                size
            }
            RechunkStrategy::Auto => 8192,
            #[cfg(feature = "size")]
            RechunkStrategy::Memory(bytes_per_chunk) => {
                let total_bytes = self.est_bytes();
                let total_len = self.len();

                if total_bytes == 0 {
                    return Err(MinarrowError::IndexError(
                        "Cannot rechunk: array has 0 estimated bytes".to_string(),
                    ));
                }

                ((bytes_per_chunk * total_len) / total_bytes).max(1)
            }
        };

        // Fast path: single chunk already at target size
        if self.arrays.len() == 1 && self.arrays[0].len() == chunk_size {
            return Ok(());
        }

        let field = self.arrays[0].field.clone();
        let mut new_chunks = Vec::new();
        let mut accumulator: Option<FieldArray> = None;

        // Process each existing chunk
        for chunk in self.arrays.drain(..) {
            let mut remaining = chunk;

            while remaining.len() > 0 {
                if let Some(ref mut acc) = accumulator {
                    let acc_len = acc.len();
                    let needed = chunk_size - acc_len;

                    if remaining.len() <= needed {
                        // Entire remaining chunk fits in accumulator
                        *acc = acc.clone().concat(remaining)?;

                        // If accumulator is now full, emit it
                        if acc.len() == chunk_size {
                            new_chunks.push(accumulator.take().unwrap());
                        }
                        break; // consumed remaining
                    } else {
                        // Split remaining chunk to complete accumulator
                        let split_result = remaining.array.split(needed, &field)?;
                        let mut parts = split_result.into_arrays();
                        let to_add = parts.remove(0);
                        remaining = parts.remove(0);

                        // Complete and emit the accumulator
                        *acc = acc.clone().concat(to_add)?;
                        new_chunks.push(accumulator.take().unwrap());
                    }
                } else {
                    // No accumulator - start processing remaining
                    if remaining.len() == chunk_size {
                        // Exact fit - use remaining as-is
                        new_chunks.push(remaining);
                        break;
                    } else if remaining.len() > chunk_size {
                        // Split off one chunk_size portion
                        let split_result = remaining.array.split(chunk_size, &field)?;
                        let mut parts = split_result.into_arrays();
                        new_chunks.push(parts.remove(0));
                        remaining = parts.remove(0);
                    } else {
                        // Remaining becomes new accumulator
                        accumulator = Some(remaining);
                        break;
                    }
                }
            }
        }

        // Emit any remaining accumulator as final chunk
        if let Some(final_chunk) = accumulator {
            new_chunks.push(final_chunk);
        }

        self.arrays = new_chunks;
        Ok(())
    }

    /// Rechunks only the first `up_to_index` elements, leaving the rest untouched.
    ///
    /// This is useful for streaming scenarios where new data is being appended
    /// and you want to rechunk stable data while leaving recent additions alone.
    ///
    /// # Arguments
    /// * `up_to_index` - Rechunk only elements before this index
    /// * `strategy` - The rechunking strategy to use
    ///
    /// # Errors
    /// - Returns `IndexError` if `up_to_index` is greater than array length
    /// - Returns same errors as `rechunk()` for invalid strategies
    ///
    /// # Example
    /// ```ignore
    /// // Rechunk first 1000 elements, leave the rest untouched
    /// array.rechunk_to(1000, RechunkStrategy::Count(512))?;
    /// ```
    pub fn rechunk_to(
        &mut self,
        up_to_index: usize,
        strategy: RechunkStrategy,
    ) -> Result<(), MinarrowError> {
        let total_len = self.len();

        if up_to_index > total_len {
            return Err(MinarrowError::IndexError(format!(
                "rechunk_to index {} out of bounds for array of length {}",
                up_to_index, total_len
            )));
        }

        if up_to_index == 0 || self.arrays.is_empty() {
            return Ok(());
        }

        if up_to_index == total_len {
            // Rechunk everything
            return self.rechunk(strategy);
        }

        // Find which chunks contain the data up to up_to_index
        let mut current_offset = 0;
        let mut split_point = 0;

        for (i, chunk) in self.arrays.iter().enumerate() {
            let chunk_end = current_offset + chunk.len();
            if chunk_end > up_to_index {
                split_point = i;
                break;
            }
            current_offset = chunk_end;
        }

        // Extract chunks to rechunk and chunks to keep
        let mut to_rechunk = self.arrays.drain(..=split_point).collect::<Vec<_>>();
        let keep_chunks = self.arrays.drain(..).collect::<Vec<_>>();

        // If the split chunk needs to be divided
        if current_offset < up_to_index {
            let split_chunk = to_rechunk.pop().unwrap();
            let split_at = up_to_index - current_offset;
            let field = split_chunk.field.clone();

            let split_result = split_chunk.array.split(split_at, &field)?;
            let mut parts = split_result.into_arrays();
            to_rechunk.push(parts.remove(0));
            self.arrays.push(parts.remove(0));
        }

        // Rechunk the selected portion
        self.arrays.extend(keep_chunks);
        let mut temp = SuperArray::from_field_array_chunks(to_rechunk);
        temp.rechunk(strategy)?;

        // Reconstruct rechunked portion + untouched portion
        let mut result = temp.arrays;
        result.extend(self.arrays.drain(..));
        self.arrays = result;

        Ok(())
    }

    /// Consumes the SuperArray and returns the underlying Vec<FieldArray> chunks.
    #[inline]
    pub fn into_arrays(self) -> Vec<FieldArray> {
        self.arrays
    }
}

impl Default for SuperArray {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<FieldArray> for SuperArray {
    fn from_iter<T: IntoIterator<Item = FieldArray>>(iter: T) -> Self {
        let chunks: Vec<FieldArray> = iter.into_iter().collect();
        Self::from_field_array_chunks(chunks)
    }
}

// FieldArray -> ChunkedArray (Vec<FieldArray> of single entry)
impl From<FieldArray> for SuperArray {
    fn from(field_array: FieldArray) -> Self {
        SuperArray {
            arrays: vec![field_array],
        }
    }
}

impl Shape for SuperArray {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
    }
}

impl Concatenate for SuperArray {
    /// Concatenates two SuperArrays by appending all chunks from `other` to `self`.
    ///
    /// # Requirements
    /// - Both SuperArrays must have the same field metadata (name, type, nullability)
    ///
    /// # Returns
    /// A new SuperArray containing all chunks from `self` followed by all chunks from `other`
    ///
    /// # Errors
    /// - `IncompatibleTypeError` if field metadata doesn't match
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // If both are empty, return empty
        if self.arrays.is_empty() && other.arrays.is_empty() {
            return Ok(SuperArray::new());
        }

        // If one is empty, return the other
        if self.arrays.is_empty() {
            return Ok(other);
        }
        if other.arrays.is_empty() {
            return Ok(self);
        }

        // Validate field metadata matches
        let self_field = &self.arrays[0].field;
        let other_field = &other.arrays[0].field;

        if self_field.name != other_field.name {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "SuperArray",
                to: "SuperArray",
                message: Some(format!(
                    "Field name mismatch: '{}' vs '{}'",
                    self_field.name, other_field.name
                )),
            });
        }

        if self_field.dtype != other_field.dtype {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "SuperArray",
                to: "SuperArray",
                message: Some(format!(
                    "Field '{}' type mismatch: {:?} vs {:?}",
                    self_field.name, self_field.dtype, other_field.dtype
                )),
            });
        }

        if self_field.nullable != other_field.nullable {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "SuperArray",
                to: "SuperArray",
                message: Some(format!(
                    "Field '{}' nullable mismatch: {} vs {}",
                    self_field.name, self_field.nullable, other_field.nullable
                )),
            });
        }

        // Concatenate chunks
        let mut result_arrays = self.arrays;
        result_arrays.extend(other.arrays);

        Ok(SuperArray {
            arrays: result_arrays,
        })
    }
}

impl Display for SuperArray {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "SuperArray \"{}\" [{} rows, {} chunks] (dtype: {})",
            self.field().name,
            self.len(),
            self.n_chunks(),
            self.field().dtype
        )?;

        for (i, chunk) in self.arrays.iter().enumerate() {
            writeln!(
                f,
                "  ├─ Chunk {i}: {} rows, nulls: {}",
                chunk.len(),
                chunk.null_count
            )?;
            let indent = "    │ ";
            for line in format!("{}", chunk.array).lines() {
                writeln!(f, "{indent}{line}")?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "views")]
    use crate::NumericArray;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{Array, Field, FieldArray, Vec64};

    fn field(name: &str, dtype: ArrowType, nullable: bool) -> Field {
        Field {
            name: name.to_string(),
            dtype,
            nullable,
            metadata: Default::default(),
        }
    }

    fn int_array(data: &[i32]) -> Array {
        Array::from_int32(crate::IntegerArray::<i32> {
            data: Vec64::from_slice(data).into(),
            null_mask: None,
        })
    }

    fn fa(name: &str, data: &[i32], null_count: usize) -> FieldArray {
        FieldArray {
            field: field(name, ArrowType::Int32, false).into(),
            array: int_array(data),
            null_count: null_count,
        }
    }

    #[test]
    fn test_new_and_push() {
        let mut ca = SuperArray::new();
        assert_eq!(ca.n_chunks(), 0);
        ca.push(fa("a", &[1, 2, 3], 0));
        assert_eq!(ca.n_chunks(), 1);
        assert_eq!(ca.len(), 3);
        ca.push(fa("a", &[4, 5], 0));
        assert_eq!(ca.n_chunks(), 2);
        assert_eq!(ca.len(), 5);
    }

    #[test]
    #[should_panic(expected = "Chunk ArrowType mismatch")]
    fn test_type_mismatch() {
        let mut ca = SuperArray::new();
        ca.push(fa("a", &[1, 2, 3], 0));
        let wrong = FieldArray {
            field: field("a", ArrowType::Float64, false).into(),
            array: Array::from_float64(crate::FloatArray::<f64> {
                data: Vec64::from_slice(&[1.0, 2.0]).into(),
                null_mask: None,
            }),
            null_count: 0,
        };
        ca.push(wrong);
    }

    #[test]
    #[should_panic(expected = "Chunk field name mismatch")]
    fn test_name_mismatch() {
        let mut ca = SuperArray::new();
        ca.push(fa("a", &[1, 2, 3], 0));
        ca.push(fa("b", &[4, 5], 0)); // wrong name
    }

    #[test]
    fn test_from_field_array_chunks() {
        let c1 = fa("a", &[1, 2, 3], 0);
        let c2 = fa("a", &[4], 0);
        let ca = SuperArray::from_field_array_chunks(vec![c1.clone(), c2.clone()].into());
        assert_eq!(ca.n_chunks(), 2);
        assert_eq!(ca.len(), 4);
        assert_eq!(ca.field().name, "a");
    }

    #[test]
    #[should_panic(expected = "from_field_array_chunks: input chunks cannot be empty")]
    fn test_from_field_array_chunks_empty() {
        let _ = SuperArray::from_field_array_chunks(Vec::new());
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_slice_and_materialise() {
        use crate::NumericArray;

        let c1 = fa("a", &[10, 20, 30], 0);
        let c2 = fa("a", &[40, 50], 0);
        let ca = SuperArray::from_field_array_chunks(vec![c1.clone(), c2.clone()].into());
        let sl = ca.slice(2, 3);
        assert_eq!(sl.len, 3);
        let arr = sl.copy_to_array();
        if let Array::NumericArray(NumericArray::Int32(ia)) = arr {
            assert_eq!(&*ia.data, &[30, 40, 50]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_from_slices() {
        let c1 = fa("a", &[10, 20, 30], 0);
        let c2 = fa("a", &[40, 50], 0);
        let ca = SuperArray::from_field_array_chunks(vec![c1.clone(), c2.clone()].into());

        let sl = ca.slice(1, 4);
        let slices = &sl.slices;
        let field = c1.field.clone();
        let ca2 = SuperArray::from_slices(slices, field);
        assert_eq!(ca2.n_chunks(), 2);
        assert_eq!(ca2.len(), 4);
    }

    #[test]
    fn test_is_empty_and_default() {
        let ca = SuperArray::default();
        assert!(ca.is_empty());
        let ca2 = SuperArray::from_chunks(vec![fa("a", &[1], 0)].into());
        assert!(!ca2.is_empty());
    }

    #[test]
    fn test_metadata_accessors() {
        let ca = SuperArray::from_chunks(vec![fa("z", &[1, 2, 3, 4], 0)].into());
        assert_eq!(ca.arrow_type(), ArrowType::Int32);
        assert!(!ca.is_nullable());
        assert_eq!(ca.field().name, "z");
        assert_eq!(ca.chunks().len(), 1);
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_insert_rows_into_first_chunk() {
        let mut ca = SuperArray::from_chunks(vec![fa("a", &[1, 2, 3], 0), fa("a", &[4, 5], 0)]);

        let other = SuperArray::from_chunks(vec![fa("a", &[99, 88], 0)]);

        ca.insert_rows(1, other).unwrap();

        assert_eq!(ca.len(), 7);
        assert_eq!(ca.n_chunks(), 4);

        let result = ca.copy_to_array();
        if let Array::NumericArray(NumericArray::Int32(ia)) = result {
            assert_eq!(&*ia.data, &[1, 99, 88, 2, 3, 4, 5]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_insert_rows_into_second_chunk() {
        let mut ca = SuperArray::from_chunks(vec![fa("a", &[1, 2], 0), fa("a", &[3, 4, 5], 0)]);

        let other = SuperArray::from_chunks(vec![fa("a", &[99], 0)]);

        ca.insert_rows(3, other).unwrap();

        assert_eq!(ca.len(), 6);
        assert_eq!(ca.n_chunks(), 4);

        let result = ca.copy_to_array();
        if let Array::NumericArray(NumericArray::Int32(ia)) = result {
            assert_eq!(&*ia.data, &[1, 2, 3, 99, 4, 5]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_insert_rows_prepend() {
        let mut ca = SuperArray::from_chunks(vec![fa("a", &[1, 2, 3], 0)]);

        let other = SuperArray::from_chunks(vec![fa("a", &[99], 0)]);

        ca.insert_rows(0, other).unwrap();

        assert_eq!(ca.len(), 4);

        let result = ca.copy_to_array();
        if let Array::NumericArray(NumericArray::Int32(ia)) = result {
            assert_eq!(&*ia.data, &[99, 1, 2, 3]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_insert_rows_append() {
        let mut ca = SuperArray::from_chunks(vec![fa("a", &[1, 2, 3], 0)]);

        let other = SuperArray::from_chunks(vec![fa("a", &[99], 0)]);

        ca.insert_rows(3, other).unwrap();

        assert_eq!(ca.len(), 4);

        let result = ca.copy_to_array();
        if let Array::NumericArray(NumericArray::Int32(ia)) = result {
            assert_eq!(&*ia.data, &[1, 2, 3, 99]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[test]
    fn test_insert_rows_schema_mismatch() {
        let mut ca = SuperArray::from_chunks(vec![fa("a", &[1, 2, 3], 0)]);

        let wrong_name = FieldArray {
            field: field("b", ArrowType::Int32, false).into(),
            array: int_array(&[99]),
            null_count: 0,
        };
        let other = SuperArray::from_chunks(vec![wrong_name]);

        let result = ca.insert_rows(0, other);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_rows_out_of_bounds() {
        let mut ca = SuperArray::from_chunks(vec![fa("a", &[1, 2, 3], 0)]);
        let other = SuperArray::from_chunks(vec![fa("a", &[99], 0)]);

        let result = ca.insert_rows(10, other);
        assert!(result.is_err());
    }

    #[test]
    fn test_rechunk_uniform() {
        let mut ca = SuperArray::from_chunks(vec![
            fa("a", &[1, 2, 3], 0),
            fa("a", &[4, 5], 0),
            fa("a", &[6, 7, 8, 9], 0),
        ]);

        ca.rechunk(RechunkStrategy::Count(3)).unwrap();

        assert_eq!(ca.n_chunks(), 3);
        assert_eq!(ca.len(), 9);
        assert_eq!(ca.arrays[0].len(), 3);
        assert_eq!(ca.arrays[1].len(), 3);
        assert_eq!(ca.arrays[2].len(), 3);

        let result = ca.copy_to_array();
        if let Array::NumericArray(NumericArray::Int32(ia)) = result {
            assert_eq!(&*ia.data, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[test]
    fn test_rechunk_auto() {
        let mut ca = SuperArray::from_chunks(vec![fa("a", &[1, 2, 3], 0), fa("a", &[4, 5], 0)]);

        ca.rechunk(RechunkStrategy::Auto).unwrap();

        assert_eq!(ca.n_chunks(), 1);
        assert_eq!(ca.len(), 5);
    }

    #[test]
    #[cfg(feature = "size")]
    fn test_rechunk_by_memory() {
        let mut ca = SuperArray::from_chunks(vec![
            fa("a", &[1, 2, 3, 4, 5, 6, 7, 8], 0),
            fa("a", &[9, 10, 11, 12], 0),
        ]);

        // i32 is 4 bytes, so 16 bytes = 4 elements
        ca.rechunk(RechunkStrategy::Memory(16)).unwrap();

        assert_eq!(ca.len(), 12);
        assert!(ca.n_chunks() >= 3);

        let result = ca.copy_to_array();
        if let Array::NumericArray(NumericArray::Int32(ia)) = result {
            assert_eq!(&*ia.data, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        } else {
            panic!("Expected Int32");
        }
    }

    #[test]
    fn test_rechunk_uniform_zero_error() {
        let mut ca = SuperArray::from_chunks(vec![fa("a", &[1, 2, 3], 0)]);

        let result = ca.rechunk(RechunkStrategy::Count(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_rechunk_empty_array() {
        let mut ca = SuperArray::new();
        ca.rechunk(RechunkStrategy::Auto).unwrap();
        assert_eq!(ca.n_chunks(), 0);
    }

    #[test]
    fn test_rechunk_preserves_data_order() {
        let mut ca = SuperArray::from_chunks(vec![
            fa("a", &[10, 20], 0),
            fa("a", &[30], 0),
            fa("a", &[40, 50, 60], 0),
        ]);

        ca.rechunk(RechunkStrategy::Count(2)).unwrap();

        let result = ca.copy_to_array();
        if let Array::NumericArray(NumericArray::Int32(ia)) = result {
            assert_eq!(&*ia.data, &[10, 20, 30, 40, 50, 60]);
        } else {
            panic!("Expected Int32");
        }
    }
}
