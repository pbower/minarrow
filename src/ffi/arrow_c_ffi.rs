//! # **Arrow-C-FFI Module** - *Share data to another language and/or run-time**
//!
//! Implements the *Apache Arrow* **C Data Interface** for Minarrow, enabling zero-copy
//! data exchange across language boundaries.  
//! Compatible with any runtime implementing the Arrow C interface, including Python, C++,
//! Java, and others.
//!
//! ## Features
//! - **Export**: Convert Minarrow arrays into Arrow C `ArrowArray` and `ArrowSchema` pointers.
//! - **Import**: Construct Minarrow arrays directly from Arrow C pointers.
//! - **Type fidelity**: Preserves physical layout, logical type metadata, and temporal units.
//! - **FFI safety**: Backing buffers are reference-counted to ensure lifetime validity
//!   across the boundary, with explicit release functions for deallocation.
//!
//! ## Usage
//! 1. Ensure the array has an associated `Field` or is wrapped in a `FieldArray`.
//! 2. For temporal arrays, set the correct `ArrowType` with appropriate `TimeUnit` or `IntervalUnit`.
//! 3. Call [`export_to_c`] to obtain Arrow-compatible pointers for FFI transmission.
//! 4. On the receiving side, reconstruct a Minarrow array with [`import_from_c`].
//!
//! ## Examples
//! - Rust -> Arrow: `cargo run --example apache-ffi`
//! - Planned: Python, C++, and other language bindings.
//!
//! ## Notes
//! - Dictionary-encoded (categorical) arrays are supported, including index type mapping.
//! - UTF-8 and large UTF-8 string arrays preserve offset and value buffer ordering.
//! - Temporal arrays validate logical type <-> physical storage alignment prior to export.
//!- `pyo3` normally abstracts pointer handling and lifetime management when integrating
//!   with Python; we do not yet use it, but once integrated, instead of manual `Arc` reference
//!  count handling and explicit clean-up, one will be able to instead leverage automatic,
//!  Python-owned lifetimes.
//!
//! ## Trademark Notice
//! *Apache Arrow* is a trademark of the Apache Software Foundation, used here under
//! fair-use to implement its published interoperability standard as per
//! https://www.apache.org/foundation/marks/ .

use std::ffi::{CString, c_void};
use std::sync::Arc;
use std::{ptr, slice};

use crate::structs::buffer::Buffer;
use crate::structs::shared_buffer::SharedBuffer;
use crate::ffi::arrow_dtype::ArrowType;
use crate::ffi::arrow_dtype::CategoricalIndexType;
use crate::ffi::schema::Schema;
use crate::{
    Array, Bitmask, BooleanArray, CategoricalArray, Field, Float, FloatArray, Integer,
    IntegerArray, MaskedArray, StringArray, TextArray, Vec64, vec64,
};
#[cfg(feature = "datetime")]
use crate::{DatetimeArray, IntervalUnit, TemporalArray, TimeUnit};

// Provides compatibility with the cross-platform `Apache Arrow` standard
// via the `C Data Interface` specification:
// https://arrow.apache.org/docs/format/CDataInterface.html

// The `C Data Interface` supports only individual arrays. To transmit tables,
// pass each column separately along with its corresponding `Schema`, constructed
// using the relevant `Field` definitions and metadata.
//
// On the receiving side, these arrays and the schema can combine to reconstruct
// an Arrow `RecordBatch`. This process can repeat to transmit multiple `RecordBatch`
// instances or once to transmit a single array—depending on the use case.

/// ArrowArray as per the Arrow C spec
/// 
/// 1. Box::new(ArrowArray::empty()) - allocates ~80 bytes for the metadata struct
/// 2. PyArrow's _export_to_c fills in the struct, setting buffers to point to PyArrow's memory
/// 3. ForeignBuffer stores this Box and uses the buffers pointer for zero-copy access
/// 4. The data never moves - we just hold a pointer to it
#[repr(C)]
pub struct ArrowArray {
    pub length: i64,
    pub null_count: i64,
    pub offset: i64,
    pub n_buffers: i64,
    pub n_children: i64,
    pub buffers: *mut *const u8,
    pub children: *mut *mut ArrowArray,
    pub dictionary: *mut ArrowArray,
    pub release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    pub private_data: *mut c_void,
}

impl ArrowArray {
    /// Creates an empty ArrowArray for receiving FFI data.
    /// Used when importing from external sources (e.g., PyArrow).
    pub fn empty() -> Self {
        Self {
            length: 0,
            null_count: 0,
            offset: 0,
            n_buffers: 0,
            n_children: 0,
            buffers: ptr::null_mut(),
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        }
    }
}

/// ArrowSchema as per the Arrow C spec
#[repr(C)]
#[derive(Clone)]
pub struct ArrowSchema {
    pub format: *const i8,
    pub name: *const i8,
    pub metadata: *const i8,
    pub flags: i64,
    pub n_children: i64,
    pub children: *mut *mut ArrowSchema,
    pub dictionary: *mut ArrowSchema,
    pub release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    pub private_data: *mut c_void,
}

impl ArrowSchema {
    /// Creates an empty ArrowSchema for receiving FFI data.
    /// Used when importing from external sources (e.g., PyArrow).
    pub fn empty() -> Self {
        Self {
            format: ptr::null(),
            name: ptr::null(),
            metadata: ptr::null(),
            flags: 0,
            n_children: 0,
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        }
    }
}

/// Keep buffers and data alive for the C Data Interface.
/// Ensures backing data and all pointers while referenced by ArrowArray.private_data.
struct Holder {
    #[allow(dead_code)] // These hold values at runtime
    array: Arc<Array>,
    _schema: Box<ArrowSchema>,
    #[allow(dead_code)]
    buf_ptrs: Vec64<*const u8>,
    #[allow(dead_code)]
    name_cstr: CString,
    #[allow(dead_code)]
    format_cstr: CString,
    #[allow(dead_code)]
    metadata_cstr: Option<CString>,
}

/// Wrapper that owns a foreign ArrowArray's buffer memory and calls release on drop.
/// Implements `AsRef<[u8]>` for use with `SharedBuffer::from_owner()`.
///
/// This enables zero-copy FFI by:
/// 1. Holding the ArrowArray (with its release callback) alive
/// 2. Providing access to the buffer data as a byte slice
/// 3. Calling the release callback when dropped, freeing the foreign memory
struct ForeignBuffer {
    /// Raw pointer to the buffer data
    ptr: *const u8,
    /// Length in bytes
    len: usize,
    /// The ArrowArray we need to release when done.
    /// Only one ForeignBuffer per ArrowArray should hold this.
    array: Option<Box<ArrowArray>>,
}

impl AsRef<[u8]> for ForeignBuffer {
    fn as_ref(&self) -> &[u8] {
        if self.len == 0 || self.ptr.is_null() {
            return &[];
        }
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for ForeignBuffer {
    fn drop(&mut self) {
        if let Some(mut arr_box) = self.array.take() {
            if let Some(release) = arr_box.release {
                unsafe { release(arr_box.as_mut() as *mut ArrowArray) };
            }
        }
    }
}

// Required for SharedBuffer::from_owner()
unsafe impl Send for ForeignBuffer {}
unsafe impl Sync for ForeignBuffer {}

/// Releases memory for ArrowArray by deallocating Holder and zeroing the structure.
/// # Safety
/// Caller must ensure this is only called once per ArrowArray.
unsafe extern "C" fn release_arrow_array(arr: *mut ArrowArray) {
    if arr.is_null() || (unsafe { &*arr }).release.is_none() {
        return;
    }
    let _: Box<Holder> = unsafe { Box::from_raw((*arr).private_data as *mut Holder) };
    unsafe { ptr::write_bytes(arr, 0, 1) };
}

/// Releases memory for ArrowSchema by zeroing the structure.
/// # Safety
/// Caller must ensure this is only called once per ArrowSchema.
unsafe extern "C" fn release_arrow_schema(s: *mut ArrowSchema) {
    if s.is_null() || (unsafe { &*s }).release.is_none() {
        return;
    }
    unsafe { ptr::write_bytes(s, 0, 1) };
}

/// Constructs the Arrow C FFI format string for the given ArrowType.
pub fn fmt_c(dtype: ArrowType) -> CString {
    #[cfg(feature = "datetime")]
    if let ArrowType::Timestamp(u, tz) = &dtype {
        let unit_str = match u {
            TimeUnit::Seconds => "tss:",
            TimeUnit::Milliseconds => "tsm:",
            TimeUnit::Microseconds => "tsu:",
            TimeUnit::Nanoseconds => "tsn:",
            TimeUnit::Days => panic!("Timestamp(Days) is invalid in Arrow C format"),
        };
        let tz_str = tz.as_deref().unwrap_or("");
        let format_str = format!("{}{}", unit_str, tz_str);
        return CString::new(format_str).expect("CString formatting failed: invalid bytes");
    }

    let bytes: &'static [u8] = match dtype {
        ArrowType::Null => b"n",
        ArrowType::Boolean => b"b",

        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int8 => b"c",
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt8 => b"C",
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::Int16 => b"s",
        #[cfg(feature = "extended_numeric_types")]
        ArrowType::UInt16 => b"S",

        ArrowType::Int32 => b"i",
        ArrowType::UInt32 => b"I",
        ArrowType::Int64 => b"l",
        ArrowType::UInt64 => b"L",
        ArrowType::Float32 => b"f",
        ArrowType::Float64 => b"g",

        ArrowType::String => b"u",
        #[cfg(feature = "large_string")]
        ArrowType::LargeString => b"U",

        // ---- datetime ----
        #[cfg(feature = "datetime")]
        ArrowType::Date32 => b"tdD",
        #[cfg(feature = "datetime")]
        ArrowType::Date64 => b"tdm",

        #[cfg(feature = "datetime")]
        ArrowType::Time32(u) => match u {
            TimeUnit::Seconds => b"tts",
            TimeUnit::Milliseconds => b"ttm",
            _ => panic!("Time32 supports Seconds or Milliseconds only"),
        },
        #[cfg(feature = "datetime")]
        ArrowType::Time64(u) => match u {
            TimeUnit::Microseconds => b"ttu",
            TimeUnit::Nanoseconds => b"ttn",
            _ => panic!("Time64 supports Microseconds or Nanoseconds only"),
        },

        #[cfg(feature = "datetime")]
        ArrowType::Duration32(u) => match u {
            TimeUnit::Seconds => b"tDs",
            TimeUnit::Milliseconds => b"tDm",
            _ => panic!("Duration32 supports Seconds or Milliseconds only"),
        },
        #[cfg(feature = "datetime")]
        ArrowType::Duration64(u) => match u {
            TimeUnit::Microseconds => b"tDu",
            TimeUnit::Nanoseconds => b"tDn",
            _ => panic!("Duration64 supports Microseconds or Nanoseconds only"),
        },

        #[cfg(feature = "datetime")]
        ArrowType::Timestamp(_, _) => {
            unreachable!("Timestamp case handled above")
        }

        #[cfg(feature = "datetime")]
        ArrowType::Interval(u) => match u {
            IntervalUnit::YearMonth => b"tiM",
            IntervalUnit::DaysTime => b"tiD",
            IntervalUnit::MonthDaysNs => b"tin",
        },

        // ---- dictionary (categorical) ----
        ArrowType::Dictionary(idx) => match idx {
            #[cfg(feature = "extended_categorical")]
            CategoricalIndexType::UInt8 => b"C",
            #[cfg(feature = "extended_categorical")]
            CategoricalIndexType::UInt16 => b"S",
            CategoricalIndexType::UInt32 => b"I",
            #[cfg(feature = "extended_categorical")]
            CategoricalIndexType::UInt64 => b"L",
        },
    };

    CString::new(bytes).expect("CString formatting failed: invalid bytes")
}

#[cfg(feature = "datetime")]
fn validate_temporal_field(array: &Array, dtype: &ArrowType) {
    use crate::{TemporalArray, TimeUnit};

    match (array, dtype) {
        // Date32 requires days since epoch in i32
        (Array::TemporalArray(TemporalArray::Datetime32(arr)), ArrowType::Date32) => {
            assert!(
                arr.time_unit == TimeUnit::Days,
                "FFI export: Field=Date32 requires Datetime32(Days); got {:?}",
                arr.time_unit
            );
        }
        // Date64 requires milliseconds since epoch in i64
        (Array::TemporalArray(TemporalArray::Datetime64(arr)), ArrowType::Date64) => {
            assert!(
                arr.time_unit == TimeUnit::Milliseconds,
                "FFI export: Field=Date64 requires Datetime64(Milliseconds); got {:?}",
                arr.time_unit
            );
        }
        // Time32 requires i32 with the exact unit
        (Array::TemporalArray(TemporalArray::Datetime32(arr)), ArrowType::Time32(u)) => {
            assert!(
                arr.time_unit == *u,
                "FFI export: Field=Time32({u:?}) requires Datetime32({u:?}); got {:?}",
                arr.time_unit
            );
        }
        // Time64 requires i64 with the exact unit
        (Array::TemporalArray(TemporalArray::Datetime64(arr)), ArrowType::Time64(u)) => {
            assert!(
                arr.time_unit == *u,
                "FFI export: Field=Time64({u:?}) requires Datetime64({u:?}); got {:?}",
                arr.time_unit
            );
        }
        // Timestamp requires i64 with the exact unit
        (Array::TemporalArray(TemporalArray::Datetime64(arr)), ArrowType::Timestamp(u, _tz)) => {
            assert!(
                arr.time_unit == *u,
                "FFI export: Field=Timestamp({u:?}) requires Datetime64({u:?}); got {:?}",
                arr.time_unit
            );
        }
        // Duration32 requires i32 with the exact unit
        (Array::TemporalArray(TemporalArray::Datetime32(arr)), ArrowType::Duration32(u)) => {
            assert!(
                arr.time_unit == *u,
                "FFI export: Field=Duration32({u:?}) requires Datetime32({u:?}); got {:?}",
                arr.time_unit
            );
        }
        // Duration64 requires i64 with the exact unit
        (Array::TemporalArray(TemporalArray::Datetime64(arr)), ArrowType::Duration64(u)) => {
            assert!(
                arr.time_unit == *u,
                "FFI export: Field=Duration64({u:?}) requires Datetime64({u:?}); got {:?}",
                arr.time_unit
            );
        }
        // Interval mapping enforced at type-selection time; nothing to assert here yet.
        _ => {}
    }
}

/// Exports a Minarrow array to Arrow C Data Interface pointers.
pub fn export_to_c(array: Arc<Array>, schema: Schema) -> (*mut ArrowArray, *mut ArrowSchema) {
    #[cfg(feature = "datetime")]
    {
        let field_ty = &schema.fields[0].dtype;
        // Validate temporal logical type <-> physical unit before export.
        validate_temporal_field(&*array, field_ty);
    }

    match &*array {
        Array::TextArray(TextArray::String32(s)) => {
            export_string_array_to_c(&array, schema, s.len() as i64)
        }
        #[cfg(feature = "large_string")]
        Array::TextArray(TextArray::String64(s)) => {
            export_string_array_to_c(&array, schema, s.len() as i64)
        }
        Array::TextArray(TextArray::Categorical32(cat)) => export_categorical_array_to_c(
            &array,
            schema,
            cat.data.len() as i64,
            &cat.unique_values,
            32,
        ),
        #[cfg(feature = "extended_categorical")]
        Array::TextArray(TextArray::Categorical8(cat)) => export_categorical_array_to_c(
            &array,
            schema,
            cat.data.len() as i64,
            &cat.unique_values,
            8,
        ),
        #[cfg(feature = "extended_categorical")]
        Array::TextArray(TextArray::Categorical16(cat)) => export_categorical_array_to_c(
            &array,
            schema,
            cat.data.len() as i64,
            &cat.unique_values,
            16,
        ),
        #[cfg(feature = "extended_categorical")]
        Array::TextArray(TextArray::Categorical64(cat)) => export_categorical_array_to_c(
            &array,
            schema,
            cat.data.len() as i64,
            &cat.unique_values,
            64,
        ),

        _ => {
            let (data_ptr, len, _) = array.data_ptr_and_byte_len();
            let (mask_ptr, _) = array
                .null_mask_ptr_and_byte_len()
                .unwrap_or((ptr::null(), 0));
            let mut buf_ptrs = vec64![mask_ptr, data_ptr];
            let name_cstr = CString::new(schema.fields[0].name.clone()).unwrap();
            check_alignment(&mut buf_ptrs, len as i64);
            create_arrow_export(array, schema, buf_ptrs, 2, len as i64, name_cstr)
        }
    }
}

/// Exports a Utf8 or LargeUtf8 string array to Arrow C format.
fn export_string_array_to_c(
    array: &Arc<Array>,
    schema: Schema,
    len: i64,
) -> (*mut ArrowArray, *mut ArrowSchema) {
    let (offsets_ptr, _) = array.offsets_ptr_and_len().unwrap();
    let (values_ptr, values_len, _) = array.data_ptr_and_byte_len();
    let (null_ptr, _) = array
        .null_mask_ptr_and_byte_len()
        .unwrap_or((ptr::null(), 0));
    // Arrow expects: [null, offsets, values]
    // For values buffer, only use it if non-empty; otherwise use null to avoid sentinel pointers
    let values_buf_ptr = if values_len > 0 { values_ptr } else { ptr::null() };
    let mut buf_ptrs = vec64![null_ptr, offsets_ptr, values_buf_ptr];
    let name_cstr = CString::new(schema.fields[0].name.clone()).unwrap();
    check_alignment(&mut buf_ptrs, len);
    create_arrow_export(array.clone(), schema, buf_ptrs, 3, len, name_cstr)
}

/// Exports a categorical array and its dictionary in Arrow C format.
fn export_categorical_array_to_c(
    array: &Arc<Array>,
    schema: Schema,
    codes_len: i64,
    unique_values: &Vec64<String>,
    index_bits: usize,
) -> (*mut ArrowArray, *mut ArrowSchema) {
    let codes_ptr = array.data_ptr_and_byte_len().0;
    let null_ptr = array
        .null_mask_ptr_and_byte_len()
        .map_or(ptr::null(), |(p, _)| p);

    let mut buf_ptrs = vec64![null_ptr, codes_ptr];
    check_alignment(&mut buf_ptrs, codes_len);

    // Export dictionary as string array with correct buffer order
    let dict_offsets: Vec64<u32> = {
        let mut offsets = Vec64::with_capacity(unique_values.len() + 1);
        let mut total = 0u32;
        offsets.push(0);
        for s in unique_values {
            total = total
                .checked_add(s.len() as u32)
                .expect("String data too large for u32 offset");
            offsets.push(total);
        }
        offsets
    };
    let dict_data: Vec64<u8> = unique_values
        .iter()
        .flat_map(|s| s.as_bytes())
        .copied()
        .collect();

    let dict_array = StringArray {
        offsets: dict_offsets.into(),
        data: dict_data.into(),
        null_mask: None,
    };

    let dict_schema = Schema::from(vec![Field::new(
        "dictionary",
        ArrowType::String,
        false,
        None,
    )]);
    let dict_array_arc = Arc::new(Array::TextArray(TextArray::String32(Arc::new(dict_array))));
    let (dict_arr_ptr, dict_schema_ptr) = export_to_c(dict_array_arc, dict_schema);

    let name_cstr = CString::new(schema.fields[0].name.clone()).unwrap();

    let mut field = schema.fields[0].clone();
    field.dtype = match index_bits {
        #[cfg(all(feature = "extended_categorical", feature = "extended_numeric_types"))]
        8 => ArrowType::Dictionary(crate::ffi::arrow_dtype::CategoricalIndexType::UInt8),
        #[cfg(all(feature = "extended_categorical", feature = "extended_numeric_types"))]
        16 => ArrowType::Dictionary(crate::ffi::arrow_dtype::CategoricalIndexType::UInt16),
        32 => ArrowType::Dictionary(crate::ffi::arrow_dtype::CategoricalIndexType::UInt32),
        #[cfg(feature = "extended_categorical")]
        64 => ArrowType::Dictionary(crate::ffi::arrow_dtype::CategoricalIndexType::UInt64),
        _ => panic!("Invalid index bits for categorical array"),
    };
    let format_cstr = fmt_c(field.dtype.clone());
    let format_ptr = format_cstr.as_ptr();

    let metadata_cstr = if field.metadata.is_empty() {
        None
    } else {
        let flat = field
            .metadata
            .iter()
            .flat_map(|(k, v)| vec![k.as_str(), v.as_str()])
            .collect::<Vec64<_>>()
            .join("\u{0001}");
        Some(CString::new(flat).unwrap())
    };
    let metadata_ptr = metadata_cstr
        .as_ref()
        .map(|c| c.as_ptr())
        .unwrap_or(ptr::null());

    let arr = Box::new(ArrowArray {
        length: codes_len,
        null_count: if buf_ptrs[0].is_null() { 0 } else { -1 },
        offset: 0,
        n_buffers: 2,
        n_children: 0,
        buffers: buf_ptrs.as_mut_ptr(),
        children: ptr::null_mut(),
        dictionary: dict_arr_ptr,
        release: Some(release_arrow_array),
        private_data: ptr::null_mut(),
    });

    let flags = if field.nullable { 1 } else { 0 };
    let schema_box = Box::new(ArrowSchema {
        format: format_ptr,
        name: name_cstr.as_ptr(),
        metadata: metadata_ptr,
        flags,
        n_children: 0,
        children: ptr::null_mut(),
        dictionary: dict_schema_ptr,
        release: Some(release_arrow_schema),
        private_data: ptr::null_mut(),
    });

    let holder = Box::new(Holder {
        array: array.clone(),
        _schema: schema_box.clone(),
        buf_ptrs,
        name_cstr,
        format_cstr,
        metadata_cstr,
    });

    let arr_ptr = Box::into_raw(arr);
    unsafe {
        (*arr_ptr).private_data = Box::into_raw(holder) as *mut c_void;
    }
    (arr_ptr, Box::into_raw(schema_box))
}

/// Imports a Minarrow array from ArrowArray and ArrowSchema C pointers.
/// # Safety
/// Both pointers must be valid and follow the Arrow C Data Interface specification.
pub unsafe fn import_from_c(arr_ptr: *const ArrowArray, sch_ptr: *const ArrowSchema) -> Arc<Array> {
    if arr_ptr.is_null() || sch_ptr.is_null() {
        panic!("FFI import_from_c: null pointer");
    }
    let arr = unsafe { &*arr_ptr };
    let sch = unsafe { &*sch_ptr };
    let fmt = unsafe { std::ffi::CStr::from_ptr(sch.format).to_bytes() };
    let is_dict = !arr.dictionary.is_null() || !sch.dictionary.is_null();

    let dtype = match fmt {
        b"n" => ArrowType::Null,
        b"b" => ArrowType::Boolean,
        #[cfg(feature = "extended_numeric_types")]
        b"c" => ArrowType::Int8,
        #[cfg(feature = "extended_numeric_types")]
        b"C" => ArrowType::UInt8,
        #[cfg(feature = "extended_numeric_types")]
        b"s" => ArrowType::Int16,
        #[cfg(feature = "extended_numeric_types")]
        b"S" => ArrowType::UInt16,
        b"i" => ArrowType::Int32,
        b"I" => ArrowType::UInt32,
        b"l" => ArrowType::Int64,
        b"L" => ArrowType::UInt64,
        b"f" => ArrowType::Float32,
        b"g" => ArrowType::Float64,
        b"u" => ArrowType::String,
        #[cfg(feature = "large_string")]
        b"U" => ArrowType::LargeString,
        #[cfg(feature = "datetime")]
        b"tdD" => ArrowType::Date32,
        #[cfg(feature = "datetime")]
        b"tdm" => ArrowType::Date64,
        #[cfg(feature = "datetime")]
        b"tts" => ArrowType::Time32(crate::TimeUnit::Seconds),
        #[cfg(feature = "datetime")]
        b"ttm" => ArrowType::Time32(crate::TimeUnit::Milliseconds),
        #[cfg(feature = "datetime")]
        b"ttu" => ArrowType::Time64(crate::TimeUnit::Microseconds),
        #[cfg(feature = "datetime")]
        b"ttn" => ArrowType::Time64(crate::TimeUnit::Nanoseconds),
        #[cfg(feature = "datetime")]
        b"tDs" => ArrowType::Duration32(crate::TimeUnit::Seconds),
        #[cfg(feature = "datetime")]
        b"tDm" => ArrowType::Duration32(crate::TimeUnit::Milliseconds),
        #[cfg(feature = "datetime")]
        b"tDu" => ArrowType::Duration64(crate::TimeUnit::Microseconds),
        #[cfg(feature = "datetime")]
        b"tDn" => ArrowType::Duration64(crate::TimeUnit::Nanoseconds),
        #[cfg(feature = "datetime")]
        b"tiM" => ArrowType::Interval(IntervalUnit::YearMonth),
        #[cfg(feature = "datetime")]
        b"tiD" => ArrowType::Interval(IntervalUnit::DaysTime),
        #[cfg(feature = "datetime")]
        b"tin" => ArrowType::Interval(IntervalUnit::MonthDaysNs),
        #[cfg(feature = "datetime")]
        _ if fmt.starts_with(b"tss")
            || fmt.starts_with(b"tsm")
            || fmt.starts_with(b"tsu")
            || fmt.starts_with(b"tsn") =>
        {
            let unit = match &fmt[..3] {
                b"tss" => crate::TimeUnit::Seconds,
                b"tsm" => crate::TimeUnit::Milliseconds,
                b"tsu" => crate::TimeUnit::Microseconds,
                b"tsn" => crate::TimeUnit::Nanoseconds,
                _ => unreachable!(),
            };
            let tz = if fmt.len() > 4 {
                let tz_bytes = &fmt[4..];
                let tz_str = String::from_utf8_lossy(tz_bytes).into_owned();
                if tz_str.is_empty() {
                    None
                } else {
                    Some(tz_str)
                }
            } else {
                None
            };
            ArrowType::Timestamp(unit, tz)
        }
        o => panic!("unsupported format {:?}", o),
    };

    // if the array owns a dictionary, map the physical index dtype ➜ CategoricalIndexType
    let maybe_cat_index = if is_dict {
        Some(match dtype {
            #[cfg(feature = "extended_numeric_types")]
            #[cfg(feature = "extended_categorical")]
            ArrowType::Int8 | ArrowType::UInt8 => CategoricalIndexType::UInt8,
            #[cfg(feature = "extended_numeric_types")]
            #[cfg(feature = "extended_categorical")]
            ArrowType::Int16 | ArrowType::UInt16 => CategoricalIndexType::UInt16,
            ArrowType::Int32 | ArrowType::UInt32 => CategoricalIndexType::UInt32,
            #[cfg(feature = "extended_numeric_types")]
            #[cfg(feature = "extended_categorical")]
            ArrowType::Int64 | ArrowType::UInt64 => CategoricalIndexType::UInt64,
            _ => panic!(
                "FFI import_from_c: unsupported dictionary index type {:?}",
                dtype
            ),
        })
    } else {
        None
    };

    if let Some(idx_ty) = maybe_cat_index {
        // SAFETY: we just verified pointers, types and dictionary presence
        return unsafe { import_categorical(arr, sch, idx_ty) };
    }

    if is_dict {
        unsafe {
            import_categorical(
                arr,
                sch,
                match dtype {
                    ArrowType::Dictionary(i) => i,
                    _ => panic!("Expected Dictionary type"),
                },
            )
        }
    } else {
        // Pass None for ownership - this function copies data because it's used for:
        // 1. Recursive imports of dictionary arrays inside categoricals (the parent
        //    ArrowArray's release callback frees the nested dictionary's memory)
        // 2. Legacy callers that don't transfer ownership
        // For zero-copy top-level imports, use import_from_c_owned instead.
        match dtype {
            ArrowType::Boolean => unsafe { import_boolean(arr, None) },
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int8 => unsafe { import_integer::<i8>(arr, None, Array::from_int8) },
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt8 => unsafe { import_integer::<u8>(arr, None, Array::from_uint8) },
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int16 => unsafe { import_integer::<i16>(arr, None, Array::from_int16) },
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt16 => unsafe { import_integer::<u16>(arr, None, Array::from_uint16) },
            ArrowType::Int32 => unsafe { import_integer::<i32>(arr, None, Array::from_int32) },
            ArrowType::UInt32 => unsafe { import_integer::<u32>(arr, None, Array::from_uint32) },
            ArrowType::Int64 => unsafe { import_integer::<i64>(arr, None, Array::from_int64) },
            ArrowType::UInt64 => unsafe { import_integer::<u64>(arr, None, Array::from_uint64) },
            ArrowType::Float32 => unsafe { import_float::<f32>(arr, None, Array::from_float32) },
            ArrowType::Float64 => unsafe { import_float::<f64>(arr, None, Array::from_float64) },
            ArrowType::String => unsafe { import_utf8::<u32>(arr, None) },
            #[cfg(feature = "large_string")]
            ArrowType::LargeString => unsafe { import_utf8::<u64>(arr, None) },
            #[cfg(feature = "datetime")]
            ArrowType::Date32 => unsafe { import_datetime::<i32>(arr, None, crate::TimeUnit::Days) },
            #[cfg(feature = "datetime")]
            ArrowType::Date64 => unsafe {
                import_datetime::<i64>(arr, None, crate::TimeUnit::Milliseconds)
            },
            #[cfg(feature = "datetime")]
            ArrowType::Time32(u) => unsafe { import_datetime::<i32>(arr, None, u) },
            #[cfg(feature = "datetime")]
            ArrowType::Time64(u) => unsafe { import_datetime::<i64>(arr, None, u) },
            #[cfg(feature = "datetime")]
            ArrowType::Timestamp(u, _tz) => unsafe { import_datetime::<i64>(arr, None, u) },
            #[cfg(feature = "datetime")]
            ArrowType::Duration32(u) => unsafe { import_datetime::<i32>(arr, None, u) },
            #[cfg(feature = "datetime")]
            ArrowType::Duration64(u) => unsafe { import_datetime::<i64>(arr, None, u) },
            #[cfg(feature = "datetime")]
            ArrowType::Interval(_u) => {
                panic!("FFI import_from_c: Arrow Interval types are not yet supported");
            }
            ArrowType::Null => {
                panic!("FFI import_from_c: Arrow Null arrays types are not yet supported")
            }
            ArrowType::Dictionary(idx) => {
                if arr.dictionary.is_null() || sch.dictionary.is_null() {
                    panic!(
                        "FFI import_from_c: dictionary pointers missing for dictionary-encoded array"
                    );
                }
                unsafe { import_categorical(arr, sch, idx) }
            }
        }
    }
}

/// Imports a Minarrow array from owned ArrowArray and ArrowSchema C pointers.
///
/// This is the zero-copy version that takes ownership of the ArrowArray.
/// The release callback will be called when the imported array is dropped.
///
/// Returns both the Array and its Field metadata, preserving the exact Arrow type
/// (e.g., Timestamp vs Date64) from the schema format string.
///
/// # Safety
/// Both boxes must contain valid Arrow C Data Interface structures.
/// After this call, ownership is transferred - do not access the boxes again.
pub unsafe fn import_from_c_owned(
    arr_box: Box<ArrowArray>,
    sch_box: Box<ArrowSchema>,
) -> (Arc<Array>, crate::Field) {
    // Get raw pointers for reading metadata.
    // This is safe because:
    // 1. For non-dict types: arr_box is passed to import functions which store it in ForeignBuffer
    // 2. The ForeignBuffer keeps the ArrowArray alive, so the pointer remains valid
    // 3. For dict types: we explicitly release after import_categorical completes
    let arr_ptr = &*arr_box as *const ArrowArray;
    let sch_ptr = &*sch_box as *const ArrowSchema;
    let arr = unsafe { &*arr_ptr };
    let sch = unsafe { &*sch_ptr };

    if arr.release.is_none() {
        panic!("FFI import_from_c_owned: ArrowArray has no release callback");
    }

    // Extract field name from schema
    let name = if sch.name.is_null() {
        String::new()
    } else {
        unsafe { std::ffi::CStr::from_ptr(sch.name) }
            .to_string_lossy()
            .into_owned()
    };

    // Check nullability from schema flags (ARROW_FLAG_NULLABLE = 2)
    let nullable = (sch.flags & 2) != 0;

    let fmt = unsafe { std::ffi::CStr::from_ptr(sch.format) }.to_bytes();
    let is_dict = !arr.dictionary.is_null() || !sch.dictionary.is_null();

    let dtype = match fmt {
        b"n" => ArrowType::Null,
        b"b" => ArrowType::Boolean,
        #[cfg(feature = "extended_numeric_types")]
        b"c" => ArrowType::Int8,
        #[cfg(feature = "extended_numeric_types")]
        b"C" => ArrowType::UInt8,
        #[cfg(feature = "extended_numeric_types")]
        b"s" => ArrowType::Int16,
        #[cfg(feature = "extended_numeric_types")]
        b"S" => ArrowType::UInt16,
        b"i" => ArrowType::Int32,
        b"I" => ArrowType::UInt32,
        b"l" => ArrowType::Int64,
        b"L" => ArrowType::UInt64,
        b"f" => ArrowType::Float32,
        b"g" => ArrowType::Float64,
        b"u" => ArrowType::String,
        #[cfg(feature = "large_string")]
        b"U" => ArrowType::LargeString,
        #[cfg(feature = "datetime")]
        b"tdD" => ArrowType::Date32,
        #[cfg(feature = "datetime")]
        b"tdm" => ArrowType::Date64,
        #[cfg(feature = "datetime")]
        b"tts" => ArrowType::Time32(crate::TimeUnit::Seconds),
        #[cfg(feature = "datetime")]
        b"ttm" => ArrowType::Time32(crate::TimeUnit::Milliseconds),
        #[cfg(feature = "datetime")]
        b"ttu" => ArrowType::Time64(crate::TimeUnit::Microseconds),
        #[cfg(feature = "datetime")]
        b"ttn" => ArrowType::Time64(crate::TimeUnit::Nanoseconds),
        #[cfg(feature = "datetime")]
        b"tDs" => ArrowType::Duration32(crate::TimeUnit::Seconds),
        #[cfg(feature = "datetime")]
        b"tDm" => ArrowType::Duration32(crate::TimeUnit::Milliseconds),
        #[cfg(feature = "datetime")]
        b"tDu" => ArrowType::Duration64(crate::TimeUnit::Microseconds),
        #[cfg(feature = "datetime")]
        b"tDn" => ArrowType::Duration64(crate::TimeUnit::Nanoseconds),
        #[cfg(feature = "datetime")]
        b"tiM" => ArrowType::Interval(IntervalUnit::YearMonth),
        #[cfg(feature = "datetime")]
        b"tiD" => ArrowType::Interval(IntervalUnit::DaysTime),
        #[cfg(feature = "datetime")]
        b"tin" => ArrowType::Interval(IntervalUnit::MonthDaysNs),
        #[cfg(feature = "datetime")]
        _ if fmt.starts_with(b"tss")
            || fmt.starts_with(b"tsm")
            || fmt.starts_with(b"tsu")
            || fmt.starts_with(b"tsn") =>
        {
            let unit = match &fmt[..3] {
                b"tss" => crate::TimeUnit::Seconds,
                b"tsm" => crate::TimeUnit::Milliseconds,
                b"tsu" => crate::TimeUnit::Microseconds,
                b"tsn" => crate::TimeUnit::Nanoseconds,
                _ => unreachable!(),
            };
            let tz = if fmt.len() > 4 {
                let tz_bytes = &fmt[4..];
                let tz_str = String::from_utf8_lossy(tz_bytes).into_owned();
                if tz_str.is_empty() { None } else { Some(tz_str) }
            } else {
                None
            };
            ArrowType::Timestamp(unit, tz)
        }
        o => panic!("unsupported format {:?}", o),
    };

    // For categorical (dictionary-encoded) types, the nested dictionary array is owned
    // by the parent's release callback. We can't take ownership of child arrays separately,
    // so the dictionary import copies its data. After copying, we call release on the parent.
    if is_dict {
        // Release the schema box (we don't need it for copying)
        drop(sch_box);
        let result = unsafe {
            import_categorical(arr, sch, match &dtype {
                ArrowType::Dictionary(i) => i.clone(),
                #[cfg(feature = "extended_numeric_types")]
                ArrowType::Int8 | ArrowType::UInt8 => {
                    #[cfg(feature = "extended_categorical")]
                    { CategoricalIndexType::UInt8 }
                    #[cfg(not(feature = "extended_categorical"))]
                    panic!("Extended categorical not enabled")
                }
                #[cfg(feature = "extended_numeric_types")]
                ArrowType::Int16 | ArrowType::UInt16 => {
                    #[cfg(feature = "extended_categorical")]
                    { CategoricalIndexType::UInt16 }
                    #[cfg(not(feature = "extended_categorical"))]
                    panic!("Extended categorical not enabled")
                }
                ArrowType::Int32 | ArrowType::UInt32 => CategoricalIndexType::UInt32,
                #[cfg(feature = "extended_numeric_types")]
                ArrowType::Int64 | ArrowType::UInt64 => {
                    #[cfg(feature = "extended_categorical")]
                    { CategoricalIndexType::UInt64 }
                    #[cfg(not(feature = "extended_categorical"))]
                    panic!("Extended categorical not enabled")
                }
                _ => panic!("FFI: unsupported dictionary index type {:?}", dtype),
            })
        };
        // Release the array after copying
        if let Some(release) = arr_box.release {
            let arr_ptr = Box::into_raw(arr_box);
            unsafe {
                release(arr_ptr);
                let _ = Box::from_raw(arr_ptr);
            }
        }
        let field = crate::Field::new(name, dtype, nullable, None);
        return (result, field);
    }

    // Drop schema box - we've extracted what we need
    drop(sch_box);

    // Pass ownership for zero-copy import
    // SAFETY: All import_* functions are unsafe and we're in an unsafe fn
    let array = unsafe {
        match dtype {
            ArrowType::Boolean => import_boolean(arr, Some(arr_box)),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int8 => import_integer::<i8>(arr, Some(arr_box), Array::from_int8),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt8 => import_integer::<u8>(arr, Some(arr_box), Array::from_uint8),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::Int16 => import_integer::<i16>(arr, Some(arr_box), Array::from_int16),
            #[cfg(feature = "extended_numeric_types")]
            ArrowType::UInt16 => import_integer::<u16>(arr, Some(arr_box), Array::from_uint16),
            ArrowType::Int32 => import_integer::<i32>(arr, Some(arr_box), Array::from_int32),
            ArrowType::UInt32 => import_integer::<u32>(arr, Some(arr_box), Array::from_uint32),
            ArrowType::Int64 => import_integer::<i64>(arr, Some(arr_box), Array::from_int64),
            ArrowType::UInt64 => import_integer::<u64>(arr, Some(arr_box), Array::from_uint64),
            ArrowType::Float32 => import_float::<f32>(arr, Some(arr_box), Array::from_float32),
            ArrowType::Float64 => import_float::<f64>(arr, Some(arr_box), Array::from_float64),
            ArrowType::String => import_utf8::<u32>(arr, Some(arr_box)),
            #[cfg(feature = "large_string")]
            ArrowType::LargeString => import_utf8::<u64>(arr, Some(arr_box)),
            #[cfg(feature = "datetime")]
            ArrowType::Date32 => import_datetime::<i32>(arr, Some(arr_box), crate::TimeUnit::Days),
            #[cfg(feature = "datetime")]
            ArrowType::Date64 => {
                import_datetime::<i64>(arr, Some(arr_box), crate::TimeUnit::Milliseconds)
            }
            #[cfg(feature = "datetime")]
            ArrowType::Time32(u) => import_datetime::<i32>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Time64(u) => import_datetime::<i64>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Timestamp(u, ref _tz) => import_datetime::<i64>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Duration32(u) => import_datetime::<i32>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Duration64(u) => import_datetime::<i64>(arr, Some(arr_box), u),
            #[cfg(feature = "datetime")]
            ArrowType::Interval(_u) => {
                panic!("FFI import_from_c_owned: Arrow Interval types are not yet supported");
            }
            ArrowType::Null => {
                panic!("FFI import_from_c_owned: Arrow Null arrays types are not yet supported")
            }
            ArrowType::Dictionary(_) => unreachable!("Dictionary handled above"),
        }
    };

    let field = crate::Field::new(name, dtype, nullable, None);
    (array, field)
}

/// Imports an integer array from Arrow C format using the given constructor.
///
/// # Arguments
/// * `arr` - Reference to the ArrowArray containing the data
/// * `ownership` - If `Some(box)`, takes ownership and does zero-copy via ForeignBuffer.
///                 If `None`, copies the data. This is needed for dictionary arrays inside
///                 categoricals, where the parent's release callback owns the child memory.
/// * `tag` - Constructor function to wrap the array
///
/// # Safety
/// `arr` must contain valid buffers of expected length and type.
unsafe fn import_integer<T: Integer>(
    arr: &ArrowArray,
    ownership: Option<Box<ArrowArray>>,
    tag: fn(IntegerArray<T>) -> Array,
) -> Arc<Array> {
    let len = arr.length as usize;
    let buffers = unsafe { slice::from_raw_parts(arr.buffers, 2) };
    let data_ptr = buffers[1] as *const T;
    let data_len_bytes = len * std::mem::size_of::<T>();

    // Null mask is always copied (small overhead)
    let null_mask = if !buffers[0].is_null() {
        Some(unsafe { Bitmask::from_raw_slice(buffers[0], len) })
    } else {
        None
    };

    // For empty arrays, create an empty buffer directly rather than using sentinel pointers
    let buffer: Buffer<T> = if len == 0 {
        Buffer::default()
    } else if let Some(arr_box) = ownership {
        // Zero-copy: wrap foreign buffer
        let foreign = ForeignBuffer {
            ptr: data_ptr as *const u8,
            len: data_len_bytes,
            array: Some(arr_box),
        };
        let shared = SharedBuffer::from_owner(foreign);
        Buffer::from_shared(shared)
    } else {
        // Copy: dictionary arrays inside categoricals have their memory owned by the
        // parent ArrowArray's release callback, so we can't take ownership of them
        let data = unsafe { slice::from_raw_parts(data_ptr, len) };
        Vec64::from(data).into()
    };

    let int_arr = IntegerArray::<T>::new(buffer, null_mask);
    Arc::new(tag(int_arr))
}

/// Imports a floating-point array from Arrow C format using the given constructor.
///
/// # Arguments
/// * `arr` - Reference to the ArrowArray containing the data
/// * `ownership` - If `Some(box)`, takes ownership and does zero-copy via ForeignBuffer.
///                 If `None`, copies the data. This is needed for dictionary arrays inside
///                 categoricals, where the parent's release callback owns the child memory.
/// * `tag` - Constructor function to wrap the array
///
/// # Safety
/// `arr` must contain valid buffers of expected length and type.
unsafe fn import_float<T>(
    arr: &ArrowArray,
    ownership: Option<Box<ArrowArray>>,
    tag: fn(FloatArray<T>) -> Array,
) -> Arc<Array>
where
    T: Float,
    FloatArray<T>: 'static,
{
    let len = arr.length as usize;
    let buffers = unsafe { slice::from_raw_parts(arr.buffers, 2) };
    let data_ptr = buffers[1] as *const T;
    let data_len_bytes = len * std::mem::size_of::<T>();

    // Null mask is always copied (small overhead)
    let null_mask = if !buffers[0].is_null() {
        Some(unsafe { Bitmask::from_raw_slice(buffers[0], len) })
    } else {
        None
    };

    // For empty arrays, create an empty buffer directly rather than using sentinel pointers
    let buffer: Buffer<T> = if len == 0 {
        Buffer::default()
    } else if let Some(arr_box) = ownership {
        // Zero-copy: wrap foreign buffer
        let foreign = ForeignBuffer {
            ptr: data_ptr as *const u8,
            len: data_len_bytes,
            array: Some(arr_box),
        };
        let shared = SharedBuffer::from_owner(foreign);
        Buffer::from_shared(shared)
    } else {
        // Copy: dictionary arrays inside categoricals have their memory owned by the
        // parent ArrowArray's release callback, so we can't take ownership of them
        let data = unsafe { slice::from_raw_parts(data_ptr, len) };
        Vec64::from(data).into()
    };

    let float_arr = FloatArray::<T>::new(buffer, null_mask);
    Arc::new(tag(float_arr))
}

/// Imports a boolean array from Arrow C format.
///
/// # Arguments
/// * `arr` - Reference to the ArrowArray containing the data
/// * `ownership` - If `Some(box)`, takes ownership and does zero-copy via ForeignBuffer.
///                 If `None`, copies the data. This is needed for dictionary arrays inside
///                 categoricals, where the parent's release callback owns the child memory.
///
/// # Safety
/// Buffers must be correctly aligned and sized for the declared length.
unsafe fn import_boolean(arr: &ArrowArray, ownership: Option<Box<ArrowArray>>) -> Arc<Array> {
    let len = arr.length as usize;
    let buffers = unsafe { slice::from_raw_parts(arr.buffers, 2) };
    let data_ptr = buffers[1];
    let data_len = (len + 7) / 8; // bytes needed for bit-packed data

    // Null mask is always copied (small overhead)
    let null_mask = if !buffers[0].is_null() {
        Some(unsafe { Bitmask::from_raw_slice(buffers[0], len) })
    } else {
        None
    };

    // For empty arrays, create an empty buffer directly rather than using sentinel pointers
    let buffer: Buffer<u8> = if len == 0 {
        Buffer::default()
    } else if let Some(arr_box) = ownership {
        // Zero-copy: wrap foreign buffer
        let foreign = ForeignBuffer {
            ptr: data_ptr as *const u8,
            len: data_len,
            array: Some(arr_box),
        };
        let shared = SharedBuffer::from_owner(foreign);
        Buffer::from_shared(shared)
    } else {
        // Copy: dictionary arrays inside categoricals have their memory owned by the
        // parent ArrowArray's release callback, so we can't take ownership of them
        let data = unsafe { slice::from_raw_parts(data_ptr, data_len) };
        Vec64::from(data).into()
    };

    let bool_mask = Bitmask::new(buffer, len);
    let bool_arr = BooleanArray::new(bool_mask, null_mask);
    Arc::new(Array::BooleanArray(bool_arr.into()))
}

/// Imports a Utf8 or LargeUtf8 string array from Arrow C format.
///
/// # Arguments
/// * `arr` - Reference to the ArrowArray containing the data
/// * `ownership` - If `Some(box)`, takes ownership and does zero-copy for values buffer.
///                 Offsets are always copied (small overhead).
///                 If `None`, copies all data. This is needed for dictionary arrays inside
///                 categoricals, where the parent's release callback owns the child memory.
///
/// # Safety
/// Expects three buffers: [nulls, offsets, values].
unsafe fn import_utf8<T: Integer>(
    arr: &ArrowArray,
    ownership: Option<Box<ArrowArray>>,
) -> Arc<Array> {
    let len = arr.length as usize;

    // For empty arrays, return an empty StringArray directly without touching sentinel pointers
    if len == 0 {
        return Arc::new(if std::mem::size_of::<T>() == 4 {
            Array::from_string32(StringArray::<u32>::default())
        } else {
            #[cfg(feature = "large_string")]
            {
                Array::from_string64(StringArray::<u64>::default())
            }
            #[cfg(not(feature = "large_string"))]
            {
                panic!("LargeUtf8 (u64 offsets) requires the 'large_string' feature")
            }
        });
    }

    let buffers = unsafe { std::slice::from_raw_parts(arr.buffers, 3) };
    let null_ptr = buffers[0];
    let offsets_ptr = buffers[1];
    let values_ptr = buffers[2];

    // Offsets - always read as slice for validation
    let offsets_slice = unsafe { std::slice::from_raw_parts(offsets_ptr as *const T, len + 1) };

    // --- BF-05: validate offsets monotonicity & bounds
    assert_eq!(
        offsets_slice.len(),
        len + 1,
        "UTF8: offsets length must be len+1"
    );
    assert_eq!(offsets_slice[0].to_usize(), 0, "UTF8: first offset must be 0");
    let mut prev = 0usize;
    for (i, off) in offsets_slice.iter().enumerate().take(len + 1) {
        let cur = off.to_usize().expect("Error: could not unwrap usize");
        assert!(
            cur >= prev,
            "UTF8: offsets not monotonically non-decreasing at {i}: {cur} < {prev}"
        );
        prev = cur;
    }
    let data_len = offsets_slice[len].to_usize();

    // Null mask - always copied (small overhead)
    let null_mask = if !null_ptr.is_null() {
        Some(unsafe { Bitmask::from_raw_slice(null_ptr, len) })
    } else {
        None
    };

    // Offsets - always copied (small: 4-8 bytes per string)
    let offsets = Vec64::from(offsets_slice);

    // Values - zero-copy if we own the ArrowArray, otherwise copy
    let values_buffer: Buffer<u8> = if data_len == 0 {
        // Empty values buffer - don't use sentinel pointer
        Buffer::default()
    } else if let Some(arr_box) = ownership {
        // Zero-copy: wrap foreign buffer for the values
        let foreign = ForeignBuffer {
            ptr: values_ptr as *const u8,
            len: data_len,
            array: Some(arr_box),
        };
        let shared = SharedBuffer::from_owner(foreign);
        Buffer::from_shared(shared)
    } else {
        // Copy: dictionary arrays inside categoricals have their memory owned by the
        // parent ArrowArray's release callback, so we can't take ownership of them
        let data = unsafe { std::slice::from_raw_parts(values_ptr, data_len) };
        Vec64::from(data).into()
    };

    let str_arr = StringArray::<T>::new(values_buffer, null_mask, offsets);

    #[cfg(feature = "large_string")]
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
        return Arc::new(Array::TextArray(TextArray::String64(Arc::new(unsafe {
            std::mem::transmute::<StringArray<T>, StringArray<u64>>(str_arr)
        }))));
    }
    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
        Arc::new(Array::TextArray(TextArray::String32(Arc::new(unsafe {
            std::mem::transmute::<StringArray<T>, StringArray<u32>>(str_arr)
        }))))
    } else {
        panic!("Unsupported offset type for StringArray (expected u32 or u64)");
    }
}

/// Imports a categorical array and dictionary from Arrow C format.
/// # Safety
/// Caller must ensure dictionary pointers are valid and formatted correctly.
unsafe fn import_categorical(
    arr: &ArrowArray,
    sch: &ArrowSchema,
    index_type: CategoricalIndexType,
) -> Arc<Array> {
    // buffers: [null, codes]
    let len = arr.length as usize;
    let buffers = unsafe { slice::from_raw_parts(arr.buffers, 2) };
    let null_ptr = buffers[0];
    let codes_ptr = buffers[1];

    // import dictionary recursively
    let dict = unsafe { import_from_c(arr.dictionary as *const _, sch.dictionary as *const _) };
    let dict_strings = match dict.as_ref() {
        Array::TextArray(TextArray::String32(s)) => (0..s.len())
            .map(|i| s.get(i).unwrap_or_default().to_string())
            .collect(),
        #[cfg(feature = "large_string")]
        Array::TextArray(TextArray::String64(s)) => (0..s.len())
            .map(|i| s.get(i).unwrap_or_default().to_string())
            .collect(),
        _ => panic!("Expected String32 dictionary"),
    };
    let null_mask = if !null_ptr.is_null() {
        Some(unsafe { Bitmask::from_raw_slice(null_ptr, len) })
    } else {
        None
    };

    // build codes & wrap
    let arc: Arc<Array> = match index_type {
        #[cfg(feature = "extended_numeric_types")]
        #[cfg(feature = "extended_categorical")]
        CategoricalIndexType::UInt8 => {
            let codes = unsafe { slice::from_raw_parts(codes_ptr as *const u8, len) };
            let arr = CategoricalArray::<u8>::new(Vec64::from(codes), dict_strings, null_mask);
            Arc::new(Array::TextArray(TextArray::Categorical8(Arc::new(arr))))
        }
        #[cfg(feature = "extended_numeric_types")]
        #[cfg(feature = "extended_categorical")]
        CategoricalIndexType::UInt16 => {
            let codes = unsafe { slice::from_raw_parts(codes_ptr as *const u16, len) };
            let arr = CategoricalArray::<u16>::new(Vec64::from(codes), dict_strings, null_mask);
            Arc::new(Array::TextArray(TextArray::Categorical16(Arc::new(arr))))
        }
        CategoricalIndexType::UInt32 => {
            let codes = unsafe { slice::from_raw_parts(codes_ptr as *const u32, len) };
            let arr = CategoricalArray::<u32>::new(Vec64::from(codes), dict_strings, null_mask);
            Arc::new(Array::TextArray(TextArray::Categorical32(Arc::new(arr))))
        }
        #[cfg(feature = "extended_numeric_types")]
        #[cfg(feature = "extended_categorical")]
        CategoricalIndexType::UInt64 => {
            let codes = unsafe { slice::from_raw_parts(codes_ptr as *const u64, len) };
            let arr = CategoricalArray::<u64>::new(Vec64::from(codes), dict_strings, null_mask);
            Arc::new(Array::TextArray(TextArray::Categorical64(Arc::new(arr))))
        }
    };
    arc
}

/// Imports a datetime array from Arrow C format.
///
/// # Arguments
/// * `arr` - Reference to the ArrowArray containing the data
/// * `ownership` - If `Some(box)`, takes ownership and does zero-copy via ForeignBuffer.
///                 If `None`, copies the data. This is needed for dictionary arrays inside
///                 categoricals, where the parent's release callback owns the child memory.
/// * `unit` - The time unit for the datetime values
///
/// # Safety
/// `arr` must contain valid time values and optional null mask.
#[cfg(feature = "datetime")]
unsafe fn import_datetime<T: Integer>(
    arr: &ArrowArray,
    ownership: Option<Box<ArrowArray>>,
    unit: crate::TimeUnit,
) -> Arc<Array> {
    let len = arr.length as usize;
    let buffers = unsafe { std::slice::from_raw_parts(arr.buffers, 2) };
    let data_ptr = buffers[1] as *const T;
    let data_len_bytes = len * std::mem::size_of::<T>();

    // Null mask is always copied (small overhead)
    let null_mask = if !buffers[0].is_null() {
        Some(unsafe { Bitmask::from_raw_slice(buffers[0], len) })
    } else {
        None
    };

    let buffer: Buffer<T> = if let Some(arr_box) = ownership {
        // Zero-copy: wrap foreign buffer
        let foreign = ForeignBuffer {
            ptr: data_ptr as *const u8,
            len: data_len_bytes,
            array: Some(arr_box),
        };
        let shared = SharedBuffer::from_owner(foreign);
        Buffer::from_shared(shared)
    } else {
        // Copy: dictionary arrays inside categoricals have their memory owned by the
        // parent ArrowArray's release callback, so we can't take ownership of them
        let data = unsafe { std::slice::from_raw_parts(data_ptr, len) };
        Vec64::from(data).into()
    };

    let dt_arr = DatetimeArray::<T> {
        data: buffer,
        null_mask,
        time_unit: unit,
    };

    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
        Arc::new(Array::TemporalArray(TemporalArray::Datetime64(Arc::new(
            unsafe { std::mem::transmute::<DatetimeArray<T>, DatetimeArray<i64>>(dt_arr) },
        ))))
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        Arc::new(Array::TemporalArray(TemporalArray::Datetime32(Arc::new(
            unsafe { std::mem::transmute::<DatetimeArray<T>, DatetimeArray<i32>>(dt_arr) },
        ))))
    } else {
        panic!("Unsupported DatetimeArray type (expected i32 or i64)");
    }
}

/// Verifies that all buffer pointers are 64-byte aligned.
/// This happens automatically when creating `Minarrow` buffers
/// so shouldn't be an issue.
///
/// Note: For empty arrays (length=0), we skip alignment checks because PyArrow
/// may return sentinel pointers (e.g., 0x4) for empty buffers that aren't
/// 64-byte aligned but also aren't actually dereferenced.
fn check_alignment(buf_ptrs: &mut Vec64<*const u8>, length: i64) {
    // Skip alignment check for empty arrays - buffer pointers may be invalid/sentinel values
    if length == 0 {
        return;
    }
    for &p in buf_ptrs.iter().take(3) {
        if !p.is_null() {
            assert_eq!(
                (p as usize) % 64,
                0,
                "FFI: Array buffer pointer {:p} is not 64-byte aligned",
                p
            );
        }
    }
}

/// Builds and returns ArrowArray and ArrowSchema for export.
/// # Safety
/// Caller must ensure buffer pointers remain valid for the lifetime of the consumer.
fn create_arrow_export(
    array: Arc<Array>,
    schema: Schema,
    mut buf_ptrs: Vec64<*const u8>,
    n_buffers: i64,
    length: i64,
    name_cstr: CString,
) -> (*mut ArrowArray, *mut ArrowSchema) {
    let null_count = if buf_ptrs[0].is_null() { 0 } else { -1 };

    let field = &schema.fields[0];

    // Format string as CString
    let format_cstr = fmt_c(field.dtype.clone());
    let format_ptr = format_cstr.as_ptr();

    // Metadata
    let metadata_cstr = if field.metadata.is_empty() {
        None
    } else {
        let flat = field
            .metadata
            .iter()
            .flat_map(|(k, v)| vec![k.as_str(), v.as_str()])
            .collect::<Vec64<_>>()
            .join("\u{0001}");
        Some(CString::new(flat).unwrap())
    };
    let metadata_ptr = metadata_cstr
        .as_ref()
        .map(|c| c.as_ptr())
        .unwrap_or(ptr::null());

    // ArrowArray
    let arr = Box::new(ArrowArray {
        length,
        null_count,
        offset: 0,
        n_buffers,
        n_children: 0,
        buffers: buf_ptrs.as_mut_ptr(),
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_arrow_array),
        private_data: ptr::null_mut(),
    });

    // ArrowSchema
    let flags = if field.nullable { 1 } else { 0 };
    let schema_box = Box::new(ArrowSchema {
        format: format_ptr,
        name: name_cstr.as_ptr(),
        metadata: metadata_ptr,
        flags,
        n_children: 0,
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_arrow_schema),
        private_data: ptr::null_mut(),
    });

    let holder = Box::new(Holder {
        array,
        _schema: schema_box.clone(),
        buf_ptrs,
        name_cstr,
        format_cstr,
        metadata_cstr,
    });

    let arr_ptr = Box::into_raw(arr);
    unsafe {
        (*arr_ptr).private_data = Box::into_raw(holder) as *mut c_void;
    }

    (arr_ptr, Box::into_raw(schema_box))
}

// Arrow C Data Interface basic tests
// E2E Tests with 'C' are under `tests/`.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    #[cfg(feature = "datetime")]
    use crate::DatetimeArray;
    use crate::ffi::arrow_c_ffi::export_to_c;
    #[cfg(feature = "datetime")]
    use crate::ffi::arrow_c_ffi::import_from_c;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::ffi::schema::Schema;
    use crate::{Array, BooleanArray, Field, FloatArray, IntegerArray, MaskedArray, StringArray};

    // Helper for constructing a one-field schema for the given type
    fn schema_for(name: &str, ty: ArrowType, nullable: bool) -> Schema {
        Schema {
            fields: vec![Field::new(name, ty, nullable, None)],
            metadata: Default::default(),
        }
    }

    #[test]
    fn test_arrow_c_export_int32() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(1);
        arr.push(2);
        arr.push(3);

        let array = Arc::new(Array::from_int32(arr));
        let schema = schema_for("ints", ArrowType::Int32, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            let vals = std::slice::from_raw_parts(bufs[1] as *const i32, 3);
            assert_eq!(vals, &[1, 2, 3]);
            // Idempotent release
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_int64() {
        let mut arr = IntegerArray::<i64>::default();
        arr.push(-42);
        arr.push(99);
        arr.push(1001);

        let array = Arc::new(Array::from_int64(arr));
        let schema = schema_for("big", ArrowType::Int64, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            let vals = std::slice::from_raw_parts(bufs[1] as *const i64, 3);
            assert_eq!(vals, &[-42, 99, 1001]);
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_u32() {
        let mut arr = IntegerArray::<u32>::default();
        arr.push(100);
        arr.push(200);
        arr.push(300);

        let array = Arc::new(Array::from_uint32(arr));
        let schema = schema_for("uints", ArrowType::UInt32, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            let vals = std::slice::from_raw_parts(bufs[1] as *const u32, 3);
            assert_eq!(vals, &[100, 200, 300]);
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_f32() {
        let mut arr = FloatArray::<f32>::default();
        arr.push(1.5);
        arr.push(-2.0);
        arr.push(3.25);

        let array = Arc::new(Array::from_float32(arr));
        let schema = schema_for("floats", ArrowType::Float32, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            let vals = std::slice::from_raw_parts(bufs[1] as *const f32, 3);
            assert_eq!(vals, &[1.5, -2.0, 3.25]);
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_f64() {
        let mut arr = FloatArray::<f64>::default();
        arr.push(0.1);
        arr.push(0.2);
        arr.push(0.3);

        let array = Arc::new(Array::from_float64(arr));
        let schema = schema_for("doubles", ArrowType::Float64, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            let vals = std::slice::from_raw_parts(bufs[1] as *const f64, 3);
            assert_eq!(vals, &[0.1, 0.2, 0.3]);
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_bool() {
        let mut arr = BooleanArray::default();
        arr.push(true);
        arr.push(false);
        arr.push(true);

        let array = Arc::new(Array::BooleanArray(arr.into()));
        let schema = schema_for("b", ArrowType::Boolean, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            assert!(!bufs[1].is_null());
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_str() {
        let mut utf = StringArray::default();
        utf.push_str("foo");
        utf.push_str("bar");

        let array = Arc::new(Array::from_string32(utf));
        let schema = schema_for("txt", ArrowType::String, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 2);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 3);
            // Arrow buffer order for UTF8: [nulls, offsets, values]
            assert!(!bufs[1].is_null(), "offsets buffer must be non-null");
            assert!(!bufs[2].is_null(), "values buffer must be non-null");
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_str_offsets() {
        let mut utf = StringArray::default();
        utf.push_str("foo");
        utf.push_str("bar");
        utf.push_str("baz");

        let array = Arc::new(Array::from_string32(utf));
        let schema = schema_for("txt", ArrowType::String, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 3);
            // Offsets are buffer[1], values are buffer[2]
            let offsets = std::slice::from_raw_parts(bufs[1] as *const u32, 4);
            assert_eq!(
                offsets,
                &[0, 3, 6, 9],
                "UTF8 offsets must be monotonically increasing starting at 0"
            );
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[test]
    fn test_arrow_c_export_with_null_mask() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(42);
        arr.push_null();
        arr.push(88);

        let array = Arc::new(Array::from_int32(arr));
        let schema = schema_for("ints", ArrowType::Int32, true);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 3);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            // null bitmap is buffer[0]
            assert!(!bufs[0].is_null());
            let bitmap = std::slice::from_raw_parts(bufs[0], 1);
            // Should be 0b00000101
            assert_eq!(bitmap[0] & 0b111, 0b101);
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_arrow_c_export_datetime() {
        use crate::TimeUnit;
        let mut dt = DatetimeArray::<i64>::default();
        dt.push(1);
        dt.push(2);
        dt.time_unit = TimeUnit::Milliseconds;

        let array = Arc::new(Array::from_datetime_i64(dt));
        let schema = schema_for("dt", ArrowType::Date64, false);

        let (arr_ptr, sch_ptr) = export_to_c(array, schema);

        unsafe {
            assert_eq!((*arr_ptr).length, 2);
            let bufs = std::slice::from_raw_parts((*arr_ptr).buffers, 2);
            assert!(!bufs[1].is_null());
            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_arrow_c_timezone_roundtrip_utc() {
        use crate::TimeUnit;
        let mut dt = DatetimeArray::<i64>::default();
        dt.push(1609459200);
        dt.push(1640995200);
        dt.time_unit = TimeUnit::Seconds;

        let array = Arc::new(Array::from_datetime_i64(dt));
        let schema = schema_for(
            "ts",
            ArrowType::Timestamp(TimeUnit::Seconds, Some("UTC".to_string())),
            false,
        );

        let (arr_ptr, sch_ptr) = export_to_c(array.clone(), schema);

        unsafe {
            let fmt = std::ffi::CStr::from_ptr((*sch_ptr).format).to_bytes();
            assert_eq!(fmt, b"tss:UTC", "Format string should include timezone");

            let imported = import_from_c(arr_ptr as *const _, sch_ptr as *const _);

            if let Array::TemporalArray(crate::TemporalArray::Datetime64(imported_dt)) =
                imported.as_ref()
            {
                assert_eq!(imported_dt.len(), 2);
                assert_eq!(imported_dt.time_unit, TimeUnit::Seconds);
            } else {
                panic!("Expected Datetime64 array");
            }

            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_arrow_c_timezone_roundtrip_iana() {
        use crate::TimeUnit;
        let mut dt = DatetimeArray::<i64>::default();
        dt.push(1609459200000);
        dt.push(1640995200000);
        dt.time_unit = TimeUnit::Milliseconds;

        let array = Arc::new(Array::from_datetime_i64(dt));
        let tz = "America/New_York".to_string();
        let schema = schema_for(
            "ts",
            ArrowType::Timestamp(TimeUnit::Milliseconds, Some(tz.clone())),
            false,
        );

        let (arr_ptr, sch_ptr) = export_to_c(array.clone(), schema);

        unsafe {
            let fmt = std::ffi::CStr::from_ptr((*sch_ptr).format).to_bytes();
            assert_eq!(
                fmt, b"tsm:America/New_York",
                "Format string should include IANA timezone"
            );

            let imported = import_from_c(arr_ptr as *const _, sch_ptr as *const _);

            if let Array::TemporalArray(crate::TemporalArray::Datetime64(imported_dt)) =
                imported.as_ref()
            {
                assert_eq!(imported_dt.len(), 2);
                assert_eq!(imported_dt.time_unit, TimeUnit::Milliseconds);
            } else {
                panic!("Expected Datetime64 array");
            }

            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_arrow_c_timezone_roundtrip_offset() {
        use crate::TimeUnit;
        let mut dt = DatetimeArray::<i64>::default();
        dt.push(1609459200000000);
        dt.time_unit = TimeUnit::Microseconds;

        let array = Arc::new(Array::from_datetime_i64(dt));
        let tz = "+05:30".to_string();
        let schema = schema_for(
            "ts",
            ArrowType::Timestamp(TimeUnit::Microseconds, Some(tz.clone())),
            false,
        );

        let (arr_ptr, sch_ptr) = export_to_c(array.clone(), schema);

        unsafe {
            let fmt = std::ffi::CStr::from_ptr((*sch_ptr).format).to_bytes();
            assert_eq!(
                fmt, b"tsu:+05:30",
                "Format string should include offset timezone"
            );

            let imported = import_from_c(arr_ptr as *const _, sch_ptr as *const _);

            if let Array::TemporalArray(crate::TemporalArray::Datetime64(imported_dt)) =
                imported.as_ref()
            {
                assert_eq!(imported_dt.len(), 1);
                assert_eq!(imported_dt.time_unit, TimeUnit::Microseconds);
            } else {
                panic!("Expected Datetime64 array");
            }

            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }

    #[cfg(feature = "datetime")]
    #[test]
    fn test_arrow_c_timezone_none() {
        use crate::TimeUnit;
        let mut dt = DatetimeArray::<i64>::default();
        dt.push(1609459200);
        dt.time_unit = TimeUnit::Seconds;

        let array = Arc::new(Array::from_datetime_i64(dt));
        let schema = schema_for("ts", ArrowType::Timestamp(TimeUnit::Seconds, None), false);

        let (arr_ptr, sch_ptr) = export_to_c(array.clone(), schema);

        unsafe {
            let fmt = std::ffi::CStr::from_ptr((*sch_ptr).format).to_bytes();
            assert_eq!(
                fmt, b"tss:",
                "Format string should have colon but no timezone"
            );

            ((*arr_ptr).release.unwrap())(arr_ptr);
            ((*sch_ptr).release.unwrap())(sch_ptr);
        }
    }
}
