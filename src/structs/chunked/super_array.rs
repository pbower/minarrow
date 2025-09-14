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

use crate::traits::shape::Shape;
use crate::enums::shape_dim::ShapeDim;
#[cfg(feature = "views")]
use crate::ArrayV;
#[cfg(feature = "views")]
use crate::SuperArrayV;
#[cfg(feature = "datetime")]
use crate::enums::time_units::TimeUnit;
use crate::ffi::arrow_dtype::ArrowType;
use crate::traits::masked_array::MaskedArray;
use crate::traits::type_unions::{Float, Integer};
use crate::{
    Array, Bitmask, BooleanArray, CategoricalArray, Field, FieldArray, FloatArray, IntegerArray,
    NumericArray, StringArray, TextArray, Vec64
};
#[cfg(feature = "datetime")]
use crate::{DatetimeArray, TemporalArray};

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
    arrays: Vec<FieldArray>
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
        assert!(!chunks.is_empty(), "from_field_array_chunks: input chunks cannot be empty");
        let field = &chunks[0].field;
        for (i, fa) in chunks.iter().enumerate().skip(1) {
            assert_eq!(
                fa.field.dtype, field.dtype,
                "Chunk {i} ArrowType mismatch (expected {:?}, got {:?})",
                field.dtype, fa.field.dtype
            );
            assert_eq!(fa.field.nullable, field.nullable, "Chunk {i} nullability mismatch");
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
                null_count: view.null_count()
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

        SuperArrayV { slices, len, field: field.into() }
    }

    // Concatenation

    /// Materialises a contiguous `Array` holding all rows.
    pub fn copy_to_array(&self) -> Array {
        assert!(!self.arrays.is_empty(), "to_array() called on empty ChunkedArray");
        match &self.arrays[0].array {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(_) => self.concat_integer::<i8>(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(_) => self.concat_integer::<i16>(),
                NumericArray::Int32(_) => self.concat_integer::<i32>(),
                NumericArray::Int64(_) => self.concat_integer::<i64>(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(_) => self.concat_integer::<u8>(),
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(_) => self.concat_integer::<u16>(),
                NumericArray::UInt32(_) => self.concat_integer::<u32>(),
                NumericArray::UInt64(_) => self.concat_integer::<u64>(),
                NumericArray::Float32(_) => self.concat_float::<f32>(),
                NumericArray::Float64(_) => self.concat_float::<f64>(),
                NumericArray::Null => unreachable!()
            },
            Array::BooleanArray(_) => self.concat_bool(),
            Array::TextArray(inner) => match inner {
                TextArray::String32(_) => self.concat_string::<u32>(),
                #[cfg(feature = "large_string")]
                TextArray::String64(_) => self.concat_string::<u64>(),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical8(_) => self.concat_dictionary::<u8>(),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(_) => self.concat_dictionary::<u16>(),
                TextArray::Categorical32(_) => self.concat_dictionary::<u32>(),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(_) => self.concat_dictionary::<u64>(),
                TextArray::Null => unreachable!()
            },
            #[cfg(feature = "datetime")]
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(_) => self.concat_datetime::<i32>(),
                TemporalArray::Datetime64(_) => self.concat_datetime::<i64>(),
                TemporalArray::Null => unreachable!()
            },
            Array::Null => unreachable!()
        }
    }

    /// Concatenates 2 or more integer numerical arrays, producing a single Array.
    /// The input arrays must all be the same underlying type.
    ///
    /// # Panics
    /// Panics if an input is not a numerical array of the correct type.
    fn concat_integer<T>(&self) -> Array
    where
        T: Integer + Default + Copy + 'static
    {
        let total: usize = self.arrays.iter().map(|c| c.len()).sum();
        let mut data = Vec64::<T>::with_capacity(total);
        let mut null_mask: Option<Bitmask> = None;

        for c in &self.arrays {
            let src = match &c.array {
                Array::NumericArray(inner) => match inner {
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::Int8(a) => unsafe {
                        &*(a.as_ref() as *const _ as *const IntegerArray<T>)
                    },
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::Int16(a) => unsafe {
                        &*(a.as_ref() as *const _ as *const IntegerArray<T>)
                    },
                    NumericArray::Int32(a) => unsafe {
                        &*(a.as_ref() as *const _ as *const IntegerArray<T>)
                    },
                    NumericArray::Int64(a) => unsafe {
                        &*(a.as_ref() as *const _ as *const IntegerArray<T>)
                    },
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::UInt8(a) => unsafe {
                        &*(a.as_ref() as *const _ as *const IntegerArray<T>)
                    },
                    #[cfg(feature = "extended_numeric_types")]
                    NumericArray::UInt16(a) => unsafe {
                        &*(a.as_ref() as *const _ as *const IntegerArray<T>)
                    },
                    NumericArray::UInt32(a) => unsafe {
                        &*(a.as_ref() as *const _ as *const IntegerArray<T>)
                    },
                    NumericArray::UInt64(a) => unsafe {
                        &*(a.as_ref() as *const _ as *const IntegerArray<T>)
                    },
                    NumericArray::Null => unreachable!(),
                    _ => unreachable!()
                },
                _ => unreachable!("concat_integer called on non-numerical array")
            };

            let dst_before = data.len();
            data.extend_from_slice(&src.data);
            if src.is_nullable() {
                null_mask.get_or_insert_with(|| Bitmask::new_set_all(total, true));
                concat_null_masks_bitmask(
                    null_mask.as_mut().unwrap(),
                    dst_before,
                    src.null_mask.as_ref(),
                    src.len()
                );
            }
        }

        let out = IntegerArray::<T> { data: data.into(), null_mask };

        match &self.arrays[0].array {
            Array::NumericArray(inner) => match inner {
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int8(_) => {
                    Array::from_int8(unsafe { std::mem::transmute::<_, IntegerArray<i8>>(out) })
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::Int16(_) => {
                    Array::from_int16(unsafe { std::mem::transmute::<_, IntegerArray<i16>>(out) })
                }
                NumericArray::Int32(_) => {
                    Array::from_int32(unsafe { std::mem::transmute::<_, IntegerArray<i32>>(out) })
                }
                NumericArray::Int64(_) => {
                    Array::from_int64(unsafe { std::mem::transmute::<_, IntegerArray<i64>>(out) })
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt8(_) => {
                    Array::from_uint8(unsafe { std::mem::transmute::<_, IntegerArray<u8>>(out) })
                }
                #[cfg(feature = "extended_numeric_types")]
                NumericArray::UInt16(_) => {
                    Array::from_uint16(unsafe { std::mem::transmute::<_, IntegerArray<u16>>(out) })
                }
                NumericArray::UInt32(_) => {
                    Array::from_uint32(unsafe { std::mem::transmute::<_, IntegerArray<u32>>(out) })
                }
                NumericArray::UInt64(_) => {
                    Array::from_uint64(unsafe { std::mem::transmute::<_, IntegerArray<u64>>(out) })
                }
                NumericArray::Null => unreachable!(),
                _ => unreachable!()
            },
            _ => unreachable!("concat_integer called on non-numerical array")
        }
    }

    /// Concatenates 2 or more float numerical arrays, producing a single Array.
    /// The input arrays must all be the same underlying type.
    ///
    /// # Panics
    /// Panics if an input is not a numerical array of the correct type.
    fn concat_float<T>(&self) -> Array
    where
        T: Float + Default + Copy + 'static
    {
        let total: usize = self.arrays.iter().map(|c| c.len()).sum();
        let mut data = Vec64::<T>::with_capacity(total);
        let mut null_mask: Option<Bitmask> = None;

        for c in &self.arrays {
            let src = match &c.array {
                Array::NumericArray(inner) => match inner {
                    NumericArray::Float32(a) => unsafe {
                        &*(a.as_ref() as *const _ as *const FloatArray<T>)
                    },
                    NumericArray::Float64(a) => unsafe {
                        &*(a.as_ref() as *const _ as *const FloatArray<T>)
                    },
                    NumericArray::Null => unreachable!(),
                    _ => unreachable!()
                },
                _ => unreachable!("concat_float called on non-numerical array")
            };
            let dst_before = data.len();
            data.extend_from_slice(&src.data);
            if src.is_nullable() {
                null_mask.get_or_insert_with(|| Bitmask::new_set_all(total, true));
                concat_null_masks_bitmask(
                    null_mask.as_mut().unwrap(),
                    dst_before,
                    src.null_mask.as_ref(),
                    src.len()
                );
            }
        }

        let out = FloatArray::<T> { data: data.into(), null_mask };

        match &self.arrays[0].array {
            Array::NumericArray(inner) => match inner {
                NumericArray::Float32(_) => {
                    Array::from_float32(unsafe { std::mem::transmute::<_, FloatArray<f32>>(out) })
                }
                NumericArray::Float64(_) => {
                    Array::from_float64(unsafe { std::mem::transmute::<_, FloatArray<f64>>(out) })
                }
                NumericArray::Null => unreachable!(),
                _ => unreachable!()
            },
            _ => unreachable!("concat_float called on non-numerical array")
        }
    }

    /// Concatenates 2 boolean arrays
    fn concat_bool(&self) -> Array {
        let total_len: usize = self.arrays.iter().map(|c| c.len()).sum();

        // Construct bit-packed buffer for data.
        let mut data = Vec64::<u8>::with_capacity((total_len + 7) / 8);
        let mut null_mask: Option<Bitmask> = None;
        let mut dst_len = 0;

        for c in &self.arrays {
            let src = match &c.array {
                Array::BooleanArray(a) => a,
                _ => unreachable!()
            };
            let bytes = (src.len() + 7) / 8;
            for b in 0..bytes {
                data.push(src.data.as_ref()[b]);
            }
            if src.is_nullable() {
                null_mask.get_or_insert_with(|| Bitmask::new_set_all(total_len, true));
                concat_null_masks_bitmask(
                    null_mask.as_mut().unwrap(),
                    dst_len,
                    src.null_mask.as_ref(),
                    src.len()
                );
            }
            dst_len += src.len();
        }

        // Finalise as Bitmask for API.
        let bit_data = Bitmask::from_bytes(&data, total_len);
        let out = BooleanArray::from_bitmask(bit_data, null_mask);
        Array::BooleanArray(out.into())
    }

    /// Concatenates 2 string arrays
    fn concat_string<O>(&self) -> Array
    where
        O: crate::traits::type_unions::Integer + num_traits::Unsigned
    {
        let mut values = Vec64::<u8>::new();
        let mut offsets = Vec64::<O>::with_capacity(1);
        let mut null_mask: Option<Bitmask> = None;
        offsets.push(O::zero());
        let mut total_rows = 0usize;
        for c in &self.arrays {
            let src = match &c.array {
                Array::TextArray(inner) => match inner {
                    TextArray::String32(a) => unsafe { &*(a as *const _ as *const StringArray<O>) },
                    #[cfg(feature = "large_string")]
                    TextArray::String64(a) => unsafe { &*(a as *const _ as *const StringArray<O>) },
                    _ => unreachable!()
                },
                _ => unreachable!()
            };
            let base = values.len();
            values.extend_from_slice(&src.data);
            for i in 1..src.offsets.len() {
                let off = src.offsets[i].to_usize() + base;
                offsets.push(O::from_usize(off));
            }
            if src.is_nullable() {
                null_mask.get_or_insert_with(|| Bitmask::new_set_all(total_rows + src.len(), true));
                concat_null_masks_bitmask(
                    null_mask.as_mut().unwrap(),
                    total_rows,
                    src.null_mask.as_ref(),
                    src.len()
                );
            }
            total_rows += src.len();
        }
        let out = StringArray::<O>::from_parts(offsets, values, null_mask);
        match &self.arrays[0].array {
            Array::TextArray(inner) => match inner {
                TextArray::String32(_) => {
                    Array::from_string32(unsafe { std::mem::transmute::<_, StringArray<u32>>(out) })
                }
                #[cfg(feature = "large_string")]
                TextArray::String64(_) => {
                    Array::from_string64(unsafe { std::mem::transmute::<_, StringArray<u64>>(out) })
                }
                _ => unreachable!()
            },
            _ => unreachable!()
        }
    }

    /// Concatenates 2 dict arrays
    fn concat_dictionary<Idx>(&self) -> Array
    where
        Idx: crate::traits::type_unions::Integer + Default + Copy
    {
        use std::collections::HashMap;
        let mut dict: Vec64<String> = Vec64::new();
        let mut dict_map: HashMap<String, Idx> = HashMap::new();
        let mut indices = Vec64::<Idx>::new();
        let mut null_mask: Option<Bitmask> = None;
        let mut dst_rows = 0usize;
        for c in &self.arrays {
            let src = match &c.array {
                Array::TextArray(inner) => match inner {
                    #[cfg(feature = "extended_categorical")]
                    TextArray::Categorical8(a) => unsafe {
                        &*(a as *const _ as *const CategoricalArray<Idx>)
                    },
                    #[cfg(feature = "extended_categorical")]
                    TextArray::Categorical16(a) => unsafe {
                        &*(a as *const _ as *const CategoricalArray<Idx>)
                    },
                    TextArray::Categorical32(a) => unsafe {
                        &*(a as *const _ as *const CategoricalArray<Idx>)
                    },
                    #[cfg(feature = "extended_categorical")]
                    TextArray::Categorical64(a) => unsafe {
                        &*(a as *const _ as *const CategoricalArray<Idx>)
                    },
                    _ => unreachable!()
                },
                _ => unreachable!()
            };
            for &idx in &src.data {
                let str_val = &src.unique_values[idx.to_usize()];
                let new_idx = *dict_map.entry(str_val.clone()).or_insert_with(|| {
                    let i = dict.len();
                    dict.push(str_val.clone());
                    Idx::from_usize(i)
                });
                indices.push(new_idx);
            }
            if src.is_nullable() {
                null_mask.get_or_insert_with(|| Bitmask::new_set_all(dst_rows + src.len(), true));
                concat_null_masks_bitmask(
                    null_mask.as_mut().unwrap(),
                    dst_rows,
                    src.null_mask.as_ref(),
                    src.len()
                );
            }
            dst_rows += src.len();
        }
        let out = CategoricalArray::<Idx>::from_parts(indices, dict, null_mask);
        match &self.arrays[0].array {
            Array::TextArray(inner) => match inner {
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical8(_) => Array::from_categorical8(unsafe {
                    std::mem::transmute::<_, CategoricalArray<u8>>(out)
                }),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical16(_) => Array::from_categorical16(unsafe {
                    std::mem::transmute::<_, CategoricalArray<u16>>(out)
                }),
                TextArray::Categorical32(_) => Array::from_categorical32(unsafe {
                    std::mem::transmute::<_, CategoricalArray<u32>>(out)
                }),
                #[cfg(feature = "extended_categorical")]
                TextArray::Categorical64(_) => Array::from_categorical64(unsafe {
                    std::mem::transmute::<_, CategoricalArray<u64>>(out)
                }),
                _ => unreachable!()
            },
            _ => unreachable!()
        }
    }

    /// Concatenates 2 datetime arrays
    #[cfg(feature = "datetime")]
    fn concat_datetime<T>(&self) -> Array
    where
        T: Integer + Default + Copy
    {
        let total: usize = self.arrays.iter().map(|c| c.len()).sum();
        let mut data = Vec64::<T>::with_capacity(total);
        let mut null_mask: Option<Bitmask> = None;
        let mut time_unit: Option<TimeUnit> = None;

        for c in &self.arrays {
            let src = match &c.array {
                Array::TemporalArray(inner) => match inner {
                    TemporalArray::Datetime32(a) => {
                        if time_unit.is_none() {
                            time_unit = Some(a.time_unit.clone());
                        } else {
                            assert_eq!(
                                time_unit,
                                Some(a.time_unit.clone()),
                                "Mismatched TimeUnit across chunks"
                            );
                        }
                        unsafe { &*(a as *const _ as *const DatetimeArray<T>) }
                    }
                    TemporalArray::Datetime64(a) => {
                        if time_unit.is_none() {
                            time_unit = Some(a.time_unit.clone());
                        } else {
                            assert_eq!(
                                time_unit,
                                Some(a.time_unit.clone()),
                                "Mismatched TimeUnit across chunks"
                            );
                        }
                        unsafe { &*(a as *const _ as *const DatetimeArray<T>) }
                    }
                    TemporalArray::Null => unreachable!()
                },
                _ => unreachable!()
            };

            let dst_before = data.len();
            data.extend_from_slice(&src.data);

            if src.is_nullable() {
                null_mask.get_or_insert_with(|| Bitmask::new_set_all(total, true));
                concat_null_masks_bitmask(
                    null_mask.as_mut().unwrap(),
                    dst_before,
                    src.null_mask.as_ref(),
                    src.len()
                );
            }
        }

        let out = DatetimeArray::<T> {
            data: data.into(),
            null_mask,
            time_unit: time_unit.expect("Expected time unit")
        };

        match &self.arrays[0].array {
            Array::TemporalArray(inner) => match inner {
                TemporalArray::Datetime32(_) => Array::from_datetime_i32(unsafe {
                    std::mem::transmute::<_, crate::DatetimeArray<i32>>(out)
                }),
                TemporalArray::Datetime64(_) => Array::from_datetime_i64(unsafe {
                    std::mem::transmute::<_, crate::DatetimeArray<i64>>(out)
                }),
                TemporalArray::Null => unreachable!()
            },
            _ => unreachable!()
        }
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
            assert_eq!(chunk.field.nullable, f.nullable, "Chunk nullability mismatch");
            assert_eq!(chunk.field.name, f.name, "Chunk field name mismatch");
            self.arrays.push(chunk);
        }
    }
}

/// Concatenates Bitmask null masks into a single mask for the output array.
fn concat_null_masks_bitmask(
    dst: &mut Bitmask,
    dst_len_before: usize,
    src_mask: Option<&Bitmask>,
    src_len: usize
) {
    if let Some(src) = src_mask {
        dst.ensure_capacity(dst_len_before + src_len);
        for i in 0..src_len {
            let valid = src.get(i);
            dst.set(dst_len_before + i, valid);
        }
    } else {
        for i in 0..src_len {
            dst.set(dst_len_before + i, true);
        }
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
        SuperArray { arrays: vec![field_array] }
    }
}

impl Shape for SuperArray {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank1(self.len())
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
            writeln!(f, "  ├─ Chunk {i}: {} rows, nulls: {}", chunk.len(), chunk.null_count)?;
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
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{Array, Field, FieldArray, Vec64};

    fn field(name: &str, dtype: ArrowType, nullable: bool) -> Field {
        Field {
            name: name.to_string(),
            dtype,
            nullable,
            metadata: Default::default()
        }
    }

    fn int_array(data: &[i32]) -> Array {
        Array::from_int32(crate::IntegerArray::<i32> {
            data: Vec64::from_slice(data).into(),
            null_mask: None
        })
    }

    fn fa(name: &str, data: &[i32], null_count: usize) -> FieldArray {
        FieldArray {
            field: field(name, ArrowType::Int32, false).into(),
            array: int_array(data),
            null_count: null_count
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
                null_mask: None
            }),
            null_count: 0
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
}
