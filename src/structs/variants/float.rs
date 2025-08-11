use std::fmt::{Display, Formatter};

use crate::structs::vec64::Vec64;
use crate::traits::print::{MAX_PREVIEW, format_float};
use crate::traits::type_unions::Float;
use crate::{
    Bitmask, Buffer, Length, MaskedArray, Offset, impl_arc_masked_array, impl_array_ref_deref,
    impl_from_vec_primitive, impl_masked_array, impl_numeric_array_constructors
};

/// Arrow-compatible Float array with 64-byte SIMD alignment.
///
/// This can be used as a standalone array type, or as part of the unified `NumericArray`
/// and/or `Array` abstractions.
///
/// ### Features
/// - A null mask for flagging missing values.
/// - A backing data buffer, typically wrapping `Vec64`, which uses a custom allocator
///   (`Alloc64`) to store standard Rust values. `Buffer` implements `Deref`, so in most
///   cases it behaves like a standard `Vec`.
///
/// ### Fields
/// - `data`: Backing buffer containing float values.
/// - `null_mask`: Optional bit-packed validity bitmap (1 = valid, 0 = null).
///
/// ### Null Mask Handling
/// The null mask can be omitted in favour of using standard `NaN` values,
/// similar to conventions in libraries like *Pandas* and *NumPy*.
/// However, the `Apache Arrow` framework defines nullability using an explicit bitmask.
///
/// ### Usage Tips
/// When working with inner types such as `FloatArray<T>` rather than the
/// unified `Array` type or mid-level variants, it is often preferable to
/// define functions that accept slices (e.g. `&[f64]` or `&[T]`, where `T`
/// implements traits like `Float`, `Numeric`, or `Primitive` from
/// [`crate::traits::type_unions`]). This approach promotes compatibility with:
/// - This library’s array types
/// - Enum representations
/// - Standard `Vec` and slice types (zero-copy compatible)
///
/// ### Performance Benchmarks
/// Benchmarks under `examples/hotloop_benchmark_std` show the performance of various
/// array representations. The tests were run 1000 times on a 2024 Intel(R) Core(TM)
/// Ultra 7 155H (22 logical CPUs, x86_64, 2 threads per core, 32 GB RAM).
///
/// Each test measured the time taken from buffer construction to summing 1000 float values.
///
/// **Important**: Run benchmarks in `--release` mode for meaningful results. Debug mode
/// may inaccurately portray performance (especially for `arrow-rs`).
///
/// #### Averaged Results (1000 runs, size = 1000)
/// - `Vec<f64>` (raw):                            avg = 0.632 µs
/// - `FloatArray` (minarrow, direct):            avg = 0.636 µs
/// - `Float64Array` (arrow-rs, struct):          avg = 0.798 µs
/// - `FloatArray` (minarrow, enum):              avg = 1.047 µs
/// - `Float64Array` (arrow-rs, dynamic trait):   avg = 1.255 µs
///
/// ### Key Takeaways
/// - `FloatArray` incurs negligible overhead relative to raw `Vec<f64>`, while enabling
///   SIMD-aligned memory compatible with AVX-512.
/// - In most practical scenarios (excluding live trading, defence, etc.), performance
///   differences are minimal.
/// - `minarrow` is marginally faster than `arrow-rs` at the time of testing, due to
///   avoiding indirection via `ArrayData` and downcasting.
/// - Enum wrapping introduces slightly less overhead than dynamic trait dispatch in `arrow-rs`.
/// - For latency-critical applications, prefer inner types like `FloatArray` directly.
/// - In non-critical contexts, performance differences are typically insignificant
///   compared to the actual computations being performed.
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub struct FloatArray<T> {
    /// Backing buffer for values.
    pub data: Buffer<T>,
    /// Optional null mask (bit-packed; 1=valid, 0=null).
    pub null_mask: Option<Bitmask>
}

impl_numeric_array_constructors!(FloatArray, Float);
impl_masked_array!(FloatArray, Float, Buffer<T>, T);
impl_from_vec_primitive!(FloatArray);
impl_array_ref_deref!(FloatArray<T>);
impl_arc_masked_array!(
    Inner = FloatArray<T>,
    T = T,
    Container = Buffer<T>,
    LogicalType = T,
    CopyType = T,
    BufferT = T,
    Variant = NumericArray,
    Bound = Float,
);

impl<T> Display for FloatArray<T>
where
    T: Float + Display
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let len = self.len();
        let nulls = self.null_count();

        writeln!(f, "FloatArray [{} values] (dtype: float, nulls: {})", len, nulls)?;

        write!(f, "[")?;

        for i in 0..usize::min(len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, ", ")?;
            }

            match self.get(i) {
                Some(v) => write!(f, "{}", format_float(v))?,
                None => write!(f, "null")?
            }
        }

        if len > MAX_PREVIEW {
            write!(f, ", … ({} total)", len)?;
        }

        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::masked_array::MaskedArray;
    use crate::vec64;

    #[test]
    fn test_new_and_with_capacity() {
        let arr = FloatArray::<f64>::default();
        assert_eq!(arr.data.len(), 0);
        assert!(arr.null_mask.is_none());

        let arr = FloatArray::<f32>::with_capacity(16, true);
        assert_eq!(arr.data.len(), 0);
        assert!(arr.data.capacity() >= 16);
        assert!(arr.null_mask.is_some());
        assert!(arr.null_mask.as_ref().unwrap().capacity() >= 2);
    }

    #[test]
    fn test_push_and_get_no_null_mask() {
        let mut arr = FloatArray::<f64>::with_capacity(2, false);
        arr.push(3.14);
        arr.push(2.71);
        assert_eq!(arr.data, vec64![3.14, 2.71],);
        assert_eq!(arr.get(0), Some(3.14));
        assert_eq!(arr.get(1), Some(2.71));
        assert!(!arr.is_null(0));
        assert!(!arr.is_null(1));
    }

    #[test]
    fn test_push_and_get_with_null_mask() {
        let mut arr = FloatArray::<f32>::with_capacity(3, true);
        arr.push(1.23);
        arr.push_null();
        arr.push(9.87);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some(1.23));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(2), Some(9.87));
        assert!(!arr.is_null(0));
        assert!(arr.is_null(1));
        assert!(!arr.is_null(2));
    }

    #[test]
    fn test_push_null_auto_mask() {
        let mut arr = FloatArray::<f32>::default();
        arr.push_null();
        assert_eq!(arr.data, vec64![0.0]);
        assert!(arr.is_null(0));
        assert!(arr.null_mask.is_some());
    }

    #[test]
    fn test_set_and_set_null() {
        let mut arr = FloatArray::<f32>::with_capacity(4, true);
        arr.push(0.1);
        arr.push(0.2);
        arr.push(0.3);
        arr.set(1, 7.7);
        assert_eq!(arr.get(1), Some(7.7));
        arr.set_null(2);
        assert_eq!(arr.get(2), None);
        assert!(arr.is_null(2));
    }

    #[test]
    fn test_trait_masked_array() {
        let mut arr = FloatArray::<f64>::with_capacity(2, true);
        arr.push(11.1);
        arr.push_null();
        assert_eq!(arr.data, vec64![11.1, 0.0]);
        assert_eq!(arr.get(0), Some(11.1));
        assert_eq!(arr.get(1), None);
        assert!(arr.is_null(1));
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_trait_mutable_array() {
        let mut arr = FloatArray::<f32>::with_capacity(2, true);
        arr.push(1.0);
        arr.push(2.0);
        arr.set(0, 10.0);
        arr.set_null(1);
        assert_eq!(arr.get(0), Some(10.0));
        assert_eq!(arr.get(1), None);
        let data = arr.data_mut();
        data[1] = 123.0;
        assert_eq!(arr.data[1], 123.0);
    }

    #[test]
    fn test_bulk_push_nulls() {
        let mut arr = FloatArray::<f64>::with_capacity(8, true);
        arr.push(9.9);
        arr.push_nulls(3);
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some(9.9));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(3), None);
    }

    #[test]
    fn test_is_empty_and_len() {
        let mut arr = FloatArray::<f64>::default();
        assert!(arr.is_empty());
        arr.push(7.0);
        assert!(!arr.is_empty());
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn test_out_of_bounds() {
        let arr = FloatArray::<f32>::default();
        assert_eq!(arr.get(0), None);
        assert_eq!(arr.get(10), None);
    }

    #[test]
    fn test_null_mask_replace() {
        let mut arr = FloatArray::<f64>::default();
        arr.push(1.0);
        arr.set_null_mask(Some(Bitmask::from_bytes([0b0000_0001], 1)));
        assert!(!arr.is_null(0));
    }

    #[test]
    fn test_float_array_slice() {
        let mut arr = FloatArray::<f64>::default();
        arr.push(1.5);
        arr.push(2.5);
        arr.push_null();
        arr.push(4.5);
        arr.push(5.5);

        let sliced = arr.slice_clone(1, 3);
        assert_eq!(sliced.len(), 3);
        assert_eq!(sliced.get(0), Some(2.5));
        assert_eq!(sliced.get(1), None); // was null
        assert_eq!(sliced.get(2), Some(4.5));
        assert_eq!(sliced.null_count(), 1);
    }
}

#[cfg(test)]
#[cfg(feature = "parallel_proc")]
mod parallel_tests {
    use rayon::prelude::*;

    use crate::{Bitmask, FloatArray};

    #[test]
    fn test_floatarray_par_iter() {
        let mut arr = FloatArray::<f32>::from_slice(&[1.0, 2.5, 3.5]);
        let mut mask = Bitmask::new_set_all(3, false);
        mask.set_bits_chunk(0, 0b0000_0001, 3);
        arr.null_mask = Some(mask);
        let vals: Vec<f32> = arr.par_iter().map(|v| *v).collect();
        assert_eq!(vals, vec![1.0, 2.5, 3.5]);
    }

    #[test]
    fn test_floatarray_par_iter_opt() {
        let mut arr = FloatArray::<f32>::from_slice(&[1.0, 2.5, 3.5]);
        let mut mask = Bitmask::new_set_all(3, false);
        mask.set_bits_chunk(0, 0b0000_0001, 3);
        arr.null_mask = Some(mask);
        let opt: Vec<Option<f32>> = arr.par_iter_opt().map(|v| v.copied()).collect();
        assert_eq!(opt, vec![Some(1.0), None, None]);
    }

    #[test]
    fn test_floatarray_par_iter_mut() {
        let mut arr = FloatArray::<f32>::from_slice(&[1.0, 2.0, 3.0]);
        arr.par_iter_mut().for_each(|v| *v *= 10.0);
        let expected = vec![10.0, 20.0, 30.0];
        let actual: Vec<f32> = arr.par_iter().map(|v| *v).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_floatarray_par_iter_range_unchecked() {
        let arr = FloatArray::<f32>::from_slice(&[1.0, 2.5, 3.5, 4.5]);
        let out: Vec<&f32> = unsafe { arr.par_iter_range_unchecked(1, 4).collect() };
        assert_eq!(*out[0], 2.5);
        assert_eq!(*out[1], 3.5);
        assert_eq!(*out[2], 4.5);
    }

    #[test]
    fn test_floatarray_par_iter_range_opt_unchecked() {
        let mut arr = FloatArray::<f32>::from_slice(&[1.1, 2.2, 3.3, 4.4]);
        let mut mask = Bitmask::new_set_all(4, false);
        mask.set_bits_chunk(0, 0b0000_0110, 4); // indices 1,2 valid
        arr.null_mask = Some(mask);
        let out: Vec<Option<&f32>> = unsafe { arr.par_iter_range_opt_unchecked(0, 4).collect() };
        assert_eq!(
            out.iter().map(|x| x.copied()).collect::<Vec<_>>(),
            vec![None, Some(2.2), Some(3.3), None]
        );
    }
}
