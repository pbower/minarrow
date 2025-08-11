//! This module implements the *Apache Arrow* `C-Data-Interface`, which is used to send data
//! across language boundaries. It is not too difficult then, to send data across to *Python*,
//! *C++*, or any other run-time that contains an `Arrow` `C-Data-Interface` implementation.
//!  
//! ### Sending data over FFI
//! When doing so, it is important to keep the lifetime open so that Rust (`Minarrow`) doesn't
//! drop the value, when it's being used to back a type in the other language.
//! Frameworks such as `pyo3` streamline this process with `Python` via a simplified typing
//! interface, but it is not necessarily required. We also plan to add Pyo3 bindings to this
//! library in future. In the absence of that, because we already wrap inner array values in
//! `Arc`, we increment the reference count to avoid it being cleaned up. The difference is,
//! in this case, you will need to make sure it gets cleaned up afterwards.
//! 
//! To achieve the FFI, you will first need to make sure your `Array` has an associated `Field`,
//! or is bundled in a `FieldArray`. This requires you to ensure that the `ArrowType` backing
//! the field has the correct logical type metadata when you construct it. This is particularly
//! applicable for the `Datetime` array variants, as this is where compatibility for the transfer
//! into `TIME32`, `TIME64`, `INTERVAL`, `DATE32`, `DATE64` etc. happens, when on the `Minarrow`
//! side we use the one `DatetimeArray<T>` for common storage.
//!
//! ### Examples
//! There is a roundtrip example under `examples` of using this to send data to `Apache Arrow` in Rust.
//! It is runnable via `cargo run --example apache-ffi`. We plan to add additional examples to other languages in future.
//! 
//! ### Disclaimer
// The term `Apache Arrow` is a trademark of the `Apache Software Foundation`,
// and is used below under fair-use implementation of the public
// FFI-compatibility standard in accordance with https://www.apache.org/foundation/marks/

use std::ffi::{CString, c_void};
use std::sync::Arc;
use std::{ptr, slice};

use crate::ffi::arrow_dtype::ArrowType;
#[cfg(feature = "extended_categorical")]
use crate::ffi::arrow_dtype::CategoricalIndexType;
use crate::ffi::schema::Schema;
use crate::{
    Array, Bitmask, BooleanArray, CategoricalArray, Field, Float, FloatArray,
    Integer, IntegerArray, MaskedArray, StringArray, TextArray, Vec64, vec64
};
#[cfg(feature = "datetime")]
use crate::{IntervalUnit, TimeUnit, TemporalArray, DatetimeArray};

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
    pub private_data: *mut c_void
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
    pub private_data: *mut c_void
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
    metadata_cstr: Option<CString>
}

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
    let bytes: &'static [u8] = match dtype {
        ArrowType::Null      => b"n",
        ArrowType::Boolean   => b"b",

        #[cfg(feature = "extended_numeric_types")] ArrowType::Int8   => b"c",
        #[cfg(feature = "extended_numeric_types")] ArrowType::UInt8  => b"C",
        #[cfg(feature = "extended_numeric_types")] ArrowType::Int16  => b"s",
        #[cfg(feature = "extended_numeric_types")] ArrowType::UInt16 => b"S",

        ArrowType::Int32  => b"i",
        ArrowType::UInt32 => b"I",
        ArrowType::Int64  => b"l",
        ArrowType::UInt64 => b"L",
        ArrowType::Float32 => b"f",
        ArrowType::Float64 => b"g",

        ArrowType::String  => b"u",
        #[cfg(feature = "large_string")]
        ArrowType::LargeString => b"U",

        // ---- datetime ----
        #[cfg(feature = "datetime")]
        ArrowType::Date32 => b"tdD",
        #[cfg(feature = "datetime")]
        ArrowType::Date64 => b"tdm",

        #[cfg(feature = "datetime")]
        ArrowType::Time32(u) => match u {
            TimeUnit::Seconds      => b"tts",
            TimeUnit::Milliseconds => b"ttm",
            _ => panic!("Time32 supports Seconds or Milliseconds only"),
        },
        #[cfg(feature = "datetime")]
        ArrowType::Time64(u) => match u {
            TimeUnit::Microseconds => b"ttu",
            TimeUnit::Nanoseconds  => b"ttn",
            _ => panic!("Time64 supports Microseconds or Nanoseconds only"),
        },

        #[cfg(feature = "datetime")]
        ArrowType::Duration32(u) => match u {
            TimeUnit::Seconds      => b"tDs",
            TimeUnit::Milliseconds => b"tDm",
            _ => panic!("Duration32 supports Seconds or Milliseconds only"),
        },
        #[cfg(feature = "datetime")]
        ArrowType::Duration64(u) => match u {
            TimeUnit::Microseconds => b"tDu",
            TimeUnit::Nanoseconds  => b"tDn",
            _ => panic!("Duration64 supports Microseconds or Nanoseconds only"),
        },

        #[cfg(feature = "datetime")]
        ArrowType::Timestamp(u) => match u {
            TimeUnit::Seconds      => b"tss:",
            TimeUnit::Milliseconds => b"tsm:",
            TimeUnit::Microseconds => b"tsu:",
            TimeUnit::Nanoseconds  => b"tsn:",
            TimeUnit::Days => panic!("Timestamp(Days) is invalid in Arrow C format"),
        },

        #[cfg(feature = "datetime")]
        ArrowType::Interval(u) => match u {
            IntervalUnit::YearMonth   => b"tiM",
            IntervalUnit::DaysTime    => b"tiD",
            IntervalUnit::MonthDaysNs => b"tin",
        },

        // ---- dictionary (categorical) ----
        ArrowType::Dictionary(idx) => match idx {
            #[cfg(feature = "extended_categorical")] CategoricalIndexType::UInt8  => b"C",
            #[cfg(feature = "extended_categorical")] CategoricalIndexType::UInt16 => b"S",
            CategoricalIndexType::UInt32 => b"I",
            #[cfg(feature = "extended_categorical")] CategoricalIndexType::UInt64 => b"L",
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
        (Array::TemporalArray(TemporalArray::Datetime64(arr)), ArrowType::Timestamp(u)) => {
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
            32
        ),
        #[cfg(feature = "extended_categorical")]
        Array::TextArray(TextArray::Categorical8(cat)) => export_categorical_array_to_c(
            &array,
            schema,
            cat.data.len() as i64,
            &cat.unique_values,
            8
        ),
        #[cfg(feature = "extended_categorical")]
        Array::TextArray(TextArray::Categorical16(cat)) => export_categorical_array_to_c(
            &array,
            schema,
            cat.data.len() as i64,
            &cat.unique_values,
            16
        ),
        #[cfg(feature = "extended_categorical")]
        Array::TextArray(TextArray::Categorical64(cat)) => export_categorical_array_to_c(
            &array,
            schema,
            cat.data.len() as i64,
            &cat.unique_values,
            64
        ),
        
        _ => {
            let (data_ptr, len, _) = array.data_ptr_and_byte_len();
            let (mask_ptr, _) = array.null_mask_ptr_and_byte_len().unwrap_or((ptr::null(), 0));
            let mut buf_ptrs = vec64![mask_ptr, data_ptr];
            let name_cstr = CString::new(schema.fields[0].name.clone()).unwrap();
            check_alignment(&mut buf_ptrs);
            create_arrow_export(array, schema, buf_ptrs, 2, len as i64, name_cstr)
        }
    }
}

/// Exports a Utf8 or LargeUtf8 string array to Arrow C format.
fn export_string_array_to_c(
    array: &Arc<Array>,
    schema: Schema,
    len: i64
) -> (*mut ArrowArray, *mut ArrowSchema) {
    let (offsets_ptr, _) = array.offsets_ptr_and_len().unwrap();
    let (values_ptr, _, _) = array.data_ptr_and_byte_len();
    let (null_ptr, _) = array.null_mask_ptr_and_byte_len().unwrap_or((ptr::null(), 0));
    // Arrow expects: [null, offsets, values]
    let mut buf_ptrs = vec64![null_ptr, offsets_ptr, values_ptr];
    let name_cstr = CString::new(schema.fields[0].name.clone()).unwrap();
    check_alignment(&mut buf_ptrs);
    create_arrow_export(array.clone(), schema, buf_ptrs, 3, len, name_cstr)
}

/// Exports a categorical array and its dictionary in Arrow C format.
fn export_categorical_array_to_c(
    array: &Arc<Array>,
    schema: Schema,
    codes_len: i64,
    unique_values: &Vec64<String>,
    index_bits: usize
) -> (*mut ArrowArray, *mut ArrowSchema) {
    let codes_ptr = array.data_ptr_and_byte_len().0;
    let null_ptr = array.null_mask_ptr_and_byte_len().map_or(ptr::null(), |(p, _)| p);

    let mut buf_ptrs = vec64![null_ptr, codes_ptr];
    check_alignment(&mut buf_ptrs);

    // Export dictionary as string array with correct buffer order
    let dict_offsets: Vec64<u32> = {
        let mut offsets = Vec64::with_capacity(unique_values.len() + 1);
        let mut total = 0u32;
        offsets.push(0);
        for s in unique_values {
            total =
                total.checked_add(s.len() as u32).expect("String data too large for u32 offset");
            offsets.push(total);
        }
        offsets
    };
    let dict_data: Vec64<u8> = unique_values.iter().flat_map(|s| s.as_bytes()).copied().collect();

    let dict_array = StringArray {
        offsets: dict_offsets.into(),
        data: dict_data.into(),
        null_mask: None
    };

    let dict_schema = Schema::from(vec![Field::new("dictionary", ArrowType::String, false, None)]);
    let dict_array_arc = Arc::new(Array::TextArray(TextArray::String32(Arc::new(dict_array))));
    let (dict_arr_ptr, dict_schema_ptr) = export_to_c(dict_array_arc, dict_schema);

    let name_cstr = CString::new(schema.fields[0].name.clone()).unwrap();

    let mut field = schema.fields[0].clone();
    field.dtype = match index_bits {
        #[cfg(feature = "extended_numeric_types")]
        8 => ArrowType::Dictionary(crate::ffi::arrow_dtype::CategoricalIndexType::UInt8),
        #[cfg(feature = "extended_numeric_types")]
        16 => ArrowType::Dictionary(crate::ffi::arrow_dtype::CategoricalIndexType::UInt16),
        32 => ArrowType::Dictionary(crate::ffi::arrow_dtype::CategoricalIndexType::UInt32),
        #[cfg(feature = "extended_numeric_types")]
        64 => ArrowType::Dictionary(crate::ffi::arrow_dtype::CategoricalIndexType::UInt64),
        _ => panic!("Invalid index bits for categorical array")
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
    let metadata_ptr = metadata_cstr.as_ref().map(|c| c.as_ptr()).unwrap_or(ptr::null());

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
        private_data: ptr::null_mut()
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
        private_data: ptr::null_mut()
    });

    let holder = Box::new(Holder {
        array: array.clone(),
        _schema: schema_box.clone(),
        buf_ptrs,
        name_cstr,
        format_cstr,
        metadata_cstr
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
        _ if fmt.starts_with(b"tss") => ArrowType::Timestamp(crate::TimeUnit::Seconds),
        #[cfg(feature = "datetime")]
        _ if fmt.starts_with(b"tsm") => ArrowType::Timestamp(crate::TimeUnit::Milliseconds),
        #[cfg(feature = "datetime")]
        _ if fmt.starts_with(b"tsu") => ArrowType::Timestamp(crate::TimeUnit::Microseconds),
        #[cfg(feature = "datetime")]
        _ if fmt.starts_with(b"tsn") => ArrowType::Timestamp(crate::TimeUnit::Nanoseconds),
        o => panic!("unsupported format {:?}", o)
    };

    // if the array owns a dictionary, map the physical index dtype ➜ CategoricalIndexType
    let maybe_cat_index = if is_dict {
        Some(match dtype {
            #[cfg(feature = "extended_numeric_types")]
            #[cfg(feature = "extended_categorical")]
            ArrowType::Int8 | ArrowType::UInt8   => CategoricalIndexType::UInt8,
            #[cfg(feature = "extended_numeric_types")]
            #[cfg(feature = "extended_categorical")]
            ArrowType::Int16 | ArrowType::UInt16 => CategoricalIndexType::UInt16,
            ArrowType::Int32 | ArrowType::UInt32 => CategoricalIndexType::UInt32,
            #[cfg(feature = "extended_numeric_types")]
            #[cfg(feature = "extended_categorical")]
            ArrowType::Int64 | ArrowType::UInt64 => CategoricalIndexType::UInt64,
            _ => panic!("FFI import_from_c: unsupported dictionary index type {:?}", dtype),
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
                    _ => panic!("Expected Dictionary type")
                }
            )
        }
    } else {
        match dtype {
            ArrowType::Boolean => unsafe { import_boolean(arr) },
            #[cfg(feature = "extended_numeric_types")]
                    ArrowType::Int8 => unsafe { import_integer::<i8>(arr, Array::from_int8) },
            #[cfg(feature = "extended_numeric_types")]
                    ArrowType::UInt8 => unsafe { import_integer::<u8>(arr, Array::from_uint8) },
            #[cfg(feature = "extended_numeric_types")]
                    ArrowType::Int16 => unsafe { import_integer::<i16>(arr, Array::from_int16) },
            #[cfg(feature = "extended_numeric_types")]
                    ArrowType::UInt16 => unsafe { import_integer::<u16>(arr, Array::from_uint16) },
            ArrowType::Int32 => unsafe { import_integer::<i32>(arr, Array::from_int32) },
            ArrowType::UInt32 => unsafe { import_integer::<u32>(arr, Array::from_uint32) },
            ArrowType::Int64 => unsafe { import_integer::<i64>(arr, Array::from_int64) },
            ArrowType::UInt64 => unsafe { import_integer::<u64>(arr, Array::from_uint64) },
            ArrowType::Float32 => unsafe { import_float::<f32>(arr, Array::from_float32) },
            ArrowType::Float64 => unsafe { import_float::<f64>(arr, Array::from_float64) },
            ArrowType::String => unsafe { import_utf8::<u32>(arr) },
            #[cfg(feature = "large_string")]
                    ArrowType::LargeString => unsafe { import_utf8::<u64>(arr) },
            #[cfg(feature = "datetime")]
                    ArrowType::Date32 => unsafe { import_datetime::<i32>(arr, crate::TimeUnit::Days) },
            #[cfg(feature = "datetime")]
                    ArrowType::Date64 => unsafe {
                        import_datetime::<i64>(arr, crate::TimeUnit::Milliseconds)
                    },
            #[cfg(feature = "datetime")]
                    ArrowType::Time32(u) => unsafe { import_datetime::<i32>(arr, u) },
            #[cfg(feature = "datetime")]
                    ArrowType::Time64(u) => unsafe { import_datetime::<i64>(arr, u) },
            #[cfg(feature = "datetime")]
                    ArrowType::Timestamp(u) => unsafe { import_datetime::<i64>(arr, u) },
            #[cfg(feature = "datetime")]
                    ArrowType::Duration32(u) => unsafe { import_datetime::<i32>(arr, u) },
            #[cfg(feature = "datetime")]
                    ArrowType::Duration64(u) => unsafe { import_datetime::<i64>(arr, u) },
            #[cfg(feature = "datetime")]
                    ArrowType::Interval(_u) => {
                        panic!("FFI import_from_c: Arrow Interval types are not yet supported");
                    }
            ArrowType::Null => panic!("FFI import_from_c: Arrow Null arrays types are not yet supported"),
            ArrowType::Dictionary(idx) => {
                if arr.dictionary.is_null() || sch.dictionary.is_null() {
                    panic!("FFI import_from_c: dictionary pointers missing for dictionary-encoded array");
                }
                unsafe { import_categorical(arr, sch, idx) }
            }
        }
    }
}

/// Imports an integer array from Arrow C format using the given constructor.
/// # Safety
/// `arr` must contain valid buffers of expected length and type.
unsafe fn import_integer<T: Integer>(
    arr: &ArrowArray,
    tag: fn(IntegerArray<T>) -> Array
) -> Arc<Array> {
    let len = arr.length as usize;
    let buffers = unsafe { slice::from_raw_parts(arr.buffers, 2) };
    let data = unsafe { slice::from_raw_parts(buffers[1] as *const T, len) };
    let null_mask = if !buffers[0].is_null() {
        Some(unsafe { Bitmask::from_slice(buffers[0], len) })
    } else {
        None
    };
    let arr = IntegerArray::<T>::new(Vec64::from(data), null_mask);
    Arc::new(tag(arr))
}

/// Imports a floating-point array from Arrow C format using the given constructor.
/// # Safety
/// `arr` must contain valid buffers of expected length and type.
unsafe fn import_float<T>(arr: &ArrowArray, tag: fn(FloatArray<T>) -> Array) -> Arc<Array>
where
    T: Float,
    FloatArray<T>: 'static
{
    let len = arr.length as usize;
    let buffers = unsafe { slice::from_raw_parts(arr.buffers, 2) };
    let data = unsafe { slice::from_raw_parts(buffers[1] as *const T, len) };
    let null_mask = if !buffers[0].is_null() {
        Some(unsafe { Bitmask::from_slice(buffers[0], len) })
    } else {
        None
    };
    let arr = FloatArray::<T>::new(Vec64::from(data), null_mask);
    Arc::new(tag(arr))
}

/// Imports a boolean array from Arrow C format.
/// # Safety
/// Buffers must be correctly aligned and sized for the declared length.
unsafe fn import_boolean(arr: &ArrowArray) -> Arc<Array> {
    let len = arr.length as usize;
    let buffers = unsafe { slice::from_raw_parts(arr.buffers, 2) };
    let data_ptr = buffers[1];
    let data_len = (len + 7) / 8;
    let data_vec = unsafe { Vec64::from_raw_parts(data_ptr as *mut u8, data_len, data_len) };
    let bool_mask = Bitmask::new(data_vec, len);
    let null_mask = if !buffers[0].is_null() {
        Some(unsafe { Bitmask::from_slice(buffers[0], len) })
    } else {
        None
    };
    let arr = BooleanArray::new(bool_mask, null_mask);
    Arc::new(Array::BooleanArray(arr.into()))
}

/// Imports a Utf8 or LargeUtf8 string array from Arrow C format.
/// # Safety
/// Expects three buffers: [nulls, offsets, values].
unsafe fn import_utf8<T: Integer>(arr: &ArrowArray) -> Arc<Array> {
    let len = arr.length as usize;
    let buffers = unsafe { std::slice::from_raw_parts(arr.buffers, 3) };
    let null_ptr   = buffers[0];
    let offsets_ptr= buffers[1];
    let values_ptr = buffers[2];

    // Offsets
    let offsets = unsafe { std::slice::from_raw_parts(offsets_ptr as *const T, len + 1) };

    // --- BF-05: validate offsets monotonicity & bounds
    assert_eq!(offsets.len(), len + 1, "UTF8: offsets length must be len+1");
    assert_eq!(offsets[0].to_usize(), 0, "UTF8: first offset must be 0");
    let mut prev = 0usize;
    for (i, off) in offsets.iter().enumerate().take(len + 1) {
        let cur = off.to_usize().expect("Error: could not unwrap usize");
        assert!(cur >= prev, "UTF8: offsets not monotonically non-decreasing at {i}: {cur} < {prev}");
        prev = cur;
    }
    let data_len = if len == 0 { 0 } else { offsets[len].to_usize() };

    // Values
    let data = unsafe { std::slice::from_raw_parts(values_ptr, data_len) };

    // Null mask
    let null_mask = if !null_ptr.is_null() {
        Some(unsafe { Bitmask::from_slice(null_ptr, len) })
    } else {
        None
    };

    let arr = StringArray::<T>::new(Vec64::from(data), null_mask, Vec64::from(offsets));

    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u64>() {
        Arc::new(Array::TextArray(TextArray::String64(Arc::new(unsafe { std::mem::transmute::<
            StringArray<T>, StringArray<u64>>(arr) }
        ))))
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
        Arc::new(Array::TextArray(TextArray::String32(Arc::new(unsafe { std::mem::transmute::<
            StringArray<T>, StringArray<u32>>(arr) }
        ))))
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
    index_type: CategoricalIndexType
) -> Arc<Array> {
    // buffers: [null, codes]
    let len = arr.length as usize;
    let buffers = unsafe { slice::from_raw_parts(arr.buffers, 2) };
    let null_ptr = buffers[0];
    let codes_ptr = buffers[1];

    // import dictionary recursively
    let dict = unsafe { import_from_c(arr.dictionary as *const _, sch.dictionary as *const _) };
    let dict_strings = match dict.as_ref() {
        Array::TextArray(TextArray::String32(s)) => {
            (0..s.len()).map(|i| s.get(i).unwrap_or_default().to_string()).collect()
        }
        Array::TextArray(TextArray::String64(s)) => {
            (0..s.len()).map(|i| s.get(i).unwrap_or_default().to_string()).collect()
        }
        _ => panic!("Expected String32 dictionary")
    };
    let null_mask = if !null_ptr.is_null() {
        Some(unsafe { Bitmask::from_slice(null_ptr, len) })
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
/// # Safety
/// `arr` must contain valid time values and optional null mask.
#[cfg(feature = "datetime")]
unsafe fn import_datetime<T: Integer>(arr: &ArrowArray, unit: crate::TimeUnit) -> Arc<Array> {
    let len = arr.length as usize;
    let buffers = unsafe { std::slice::from_raw_parts(arr.buffers, 2) };
    let data = unsafe { std::slice::from_raw_parts(buffers[1] as *const T, len) };
    let null_mask = if !buffers[0].is_null() {
        Some(unsafe { Bitmask::from_slice(buffers[0], len) })
    } else {
        None
    };
    let arr = DatetimeArray::<T> {
        data: Vec64::from(data).into(),
        null_mask,
        time_unit: unit
    };

    if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
        Arc::new(Array::TemporalArray(TemporalArray::Datetime64(Arc::new(unsafe {
            std::mem::transmute::<DatetimeArray<T>, DatetimeArray<i64>>(arr)
        }))))
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
        Arc::new(Array::TemporalArray(TemporalArray::Datetime32(Arc::new(unsafe {
            std::mem::transmute::<DatetimeArray<T>, DatetimeArray<i32>>(arr)
        }))))
    } else {
        panic!("Unsupported DatetimeArray type (expected i32 or i64)");
    }
}

/// Verifies that all buffer pointers are 64-byte aligned.
/// This happens automatically when creating `Minarrow` buffers
/// so shouldn't be an issue.
fn check_alignment(buf_ptrs: &mut Vec64<*const u8>) {
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
    name_cstr: CString
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
    let metadata_ptr = metadata_cstr.as_ref().map(|c| c.as_ptr()).unwrap_or(ptr::null());

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
        private_data: ptr::null_mut()
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
        private_data: ptr::null_mut()
    });

    let holder = Box::new(Holder {
        array,
        _schema: schema_box.clone(),
        buf_ptrs,
        name_cstr,
        format_cstr,
        metadata_cstr
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
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::ffi::schema::Schema;
    use crate::{Array, BooleanArray, Field, FloatArray, IntegerArray, MaskedArray, StringArray};

    // Helper for constructing a one-field schema for the given type
    fn schema_for(name: &str, ty: ArrowType, nullable: bool) -> Schema {
        Schema {
            fields: vec![Field::new(name, ty, nullable, None)],
            metadata: Default::default()
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
            assert_eq!(offsets, &[0, 3, 6, 9], "UTF8 offsets must be monotonically increasing starting at 0");
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
    
}
