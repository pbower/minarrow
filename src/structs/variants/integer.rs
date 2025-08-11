use std::fmt::{Display, Formatter};

use crate::structs::vec64::Vec64;
use crate::traits::print::MAX_PREVIEW;
use crate::traits::type_unions::Integer;
use crate::{
    Bitmask, Buffer, Length, MaskedArray, Offset, impl_arc_masked_array, impl_array_ref_deref,
    impl_from_vec_primitive, impl_masked_array, impl_numeric_array_constructors
};

/// Arrow-compatible Integer array with 64-byte SIMD alignment.
///
/// This array can be used as a standalone numeric buffer or as part of the broader
/// `NumericArray` and/or `Array` abstractions.
///
/// ### Features
/// - A null mask for marking missing values.
/// - A backing data buffer, typically wrapping `Vec64` with the `Alloc64` allocator,
///   storing native Rust integer types. `Buffer` implements `Deref`, allowing
///   seamless use similar to standard `Vec`.
///
/// ### Fields
/// - `data`: Backing buffer containing integer values.
/// - `null_mask`: Optional bit-packed validity bitmap (1 = valid, 0 = null).
///
/// ### Null Mask Handling
/// Nullability can be represented using a bit-packed mask. Alternatively, for integer
/// types where there is no `NaN`, the mask should be used when nullable semantics are required.
/// This follows the `Apache Arrow` framework for representing nulls.
///
/// ### Usage Tips
/// For performance and compatibility, prefer defining functions over generic slices,
/// such as `&[i64]` or `&[T]`, where `T` implements integer-related traits from
/// [`crate::traits::type_unions`]. This enables:
/// - Compatibility with this library’s internal types
/// - Seamless interop with enum representations
/// - Zero-copy access from standard `Vec` and slice types
///
/// While this approach may reduce ergonomics when exposing types to external consumers,
/// it enables fast, type-safe access patterns suited for numeric computing pipelines.
///
/// ### Performance Benchmarks
/// Benchmarks under `examples/hotloop_benchmark_std` compare construction and summation
/// performance across different array representations. Results are from 1000 independent
/// runs on a 2024 Intel(R) Core(TM) Ultra 7 155H (22 logical CPUs, x86_64, 2 threads per core, 32 GB RAM).
///
/// Each benchmark measures time from buffer construction to summing 1000 integer values.
///
/// **Note**: Always run with `--release` mode enabled for representative performance. Debug
/// builds distort performance characteristics, particularly for trait-based systems like `arrow-rs`.
///
/// #### Averaged Results (1000 runs, size = 1000)
/// - `Vec<i64>` (raw):                          avg = 0.131 µs
/// - `IntegerArray` (minarrow, direct):        avg = 0.130 µs
/// - `Int64Array` (arrow-rs, struct):          avg = 0.327 µs
/// - `IntegerArray` (minarrow, enum):          avg = 2.494 µs
/// - `Int64Array` (arrow-rs, dynamic trait):   avg = 3.283 µs
///
/// ### Key Takeaways
/// - `IntegerArray` (direct usage) matches the raw `Vec<i64>` performance, while providing
///   SIMD-aligned memory and Arrow-compatible metadata.
/// - The `minarrow` representation outperforms `arrow-rs`, largely due to avoiding
///   trait-based dispatch and internal indirection via `ArrayData`.
/// - Enum-based dispatch incurs moderate overhead but still outperforms dynamic dispatch
///   used by `arrow-rs`.
/// - For real-time or latency-sensitive systems, prefer direct usage of `IntegerArray`.
/// - For general-purpose applications, performance differences are negligible compared
///   to the cost of actual computations and system-level overhead.
#[repr(C, align(64))]
#[derive(PartialEq, Clone, Debug, Default)]
pub struct IntegerArray<T> {
    /// Backing buffer for values (Arrow-compatible).
    pub data: Buffer<T>,
    /// Optional null mask (bit-packed; 1=valid, 0=null).
    pub null_mask: Option<Bitmask>
}

impl_numeric_array_constructors!(IntegerArray, Integer);
impl_masked_array!(IntegerArray, Integer, Buffer<T>, T);
impl_from_vec_primitive!(IntegerArray);
impl_array_ref_deref!(IntegerArray<T>);
impl_arc_masked_array!(
    Inner = IntegerArray<T>,
    T = T,
    Container = Buffer<T>,
    LogicalType = T,
    CopyType = T,
    BufferT = T,
    Variant = NumericArray,
    Bound = Integer,
);

impl<T> Display for IntegerArray<T>
where
    T: Integer + Display
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let len = self.len();
        let nulls = self.null_count();

        writeln!(f, "IntegerArray [{} values] (dtype: int, nulls: {})", len, nulls)?;

        write!(f, "[")?;

        for i in 0..usize::min(len, MAX_PREVIEW) {
            if i > 0 {
                write!(f, ", ")?;
            }

            match self.get(i) {
                Some(val) => write!(f, "{}", val)?,
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
    use crate::structs::bitmask::Bitmask;
    use crate::traits::masked_array::MaskedArray;
    use crate::vec64;

    #[test]
    fn test_new_and_with_capacity() {
        let arr = IntegerArray::<i32>::default();
        assert_eq!(arr.data.len(), 0);
        assert!(arr.null_mask.is_none());

        let arr = IntegerArray::<u16>::with_capacity(8, true);
        assert_eq!(arr.data.len(), 0);
        assert!(arr.data.capacity() >= 8);
        assert!(arr.null_mask.is_some());
        assert!(arr.null_mask.as_ref().unwrap().capacity() >= 8);
    }

    #[test]
    fn test_push_and_get_no_null_mask() {
        let mut arr = IntegerArray::<i64>::with_capacity(4, false);
        arr.push(123);
        arr.push(-456);
        assert_eq!(arr.data, vec64![123, -456]);
        assert_eq!(arr.get(0), Some(123));
        assert_eq!(arr.get(1), Some(-456));
        assert!(!arr.is_null(0));
        assert!(!arr.is_null(1));
    }

    #[test]
    fn test_push_and_get_with_null_mask() {
        let mut arr = IntegerArray::<u8>::with_capacity(3, true);
        arr.push(42);
        arr.push_null();
        arr.push(7);
        assert_eq!(arr.len(), 3);
        assert_eq!(arr.get(0), Some(42));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(2), Some(7));
        assert!(!arr.is_null(0));
        assert!(arr.is_null(1));
        assert!(!arr.is_null(2));
    }

    #[test]
    fn test_push_null_auto_mask() {
        let mut arr = IntegerArray::<i16>::default();
        arr.push_null();
        assert_eq!(arr.data, vec64![0]);
        assert!(arr.is_null(0));
        assert!(arr.null_mask.is_some());
    }

    #[test]
    fn test_set_and_set_null() {
        let mut arr = IntegerArray::<u32>::with_capacity(3, true);
        arr.push(100);
        arr.push(200);
        arr.push(300);
        arr.set(1, 222);
        assert_eq!(arr.get(1), Some(222));
        arr.set_null(2);
        assert_eq!(arr.get(2), None);
        assert!(arr.is_null(2));
    }

    #[test]
    fn test_trait_masked_array() {
        let mut arr = IntegerArray::<u64>::with_capacity(2, true);
        arr.push(111);
        arr.push_null();
        assert_eq!(arr.data, vec64![111, 0]);
        assert_eq!(arr.get(0), Some(111));
        assert_eq!(arr.get(1), None);
        assert!(arr.is_null(1));
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_trait_mutable_array() {
        let mut arr = IntegerArray::<i8>::with_capacity(2, true);
        arr.push(7);
        arr.push(8);
        arr.set(0, 77);
        arr.set_null(1);
        assert_eq!(arr.get(0), Some(77));
        assert_eq!(arr.get(1), None);
        let data = arr.data_mut();
        data[1] = -9;
        assert_eq!(arr.data[1], -9);
    }

    #[test]
    fn test_bulk_push_nulls() {
        let mut arr = IntegerArray::<u16>::with_capacity(8, true);
        arr.push(19);
        arr.push_nulls(3);
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.get(0), Some(19));
        assert_eq!(arr.get(1), None);
        assert_eq!(arr.get(3), None);
        assert!(arr.is_null(2));
    }

    #[test]
    fn test_is_empty_and_len() {
        let mut arr = IntegerArray::<i32>::default();
        assert!(arr.is_empty());
        arr.push(1);
        assert!(!arr.is_empty());
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn test_out_of_bounds() {
        let arr = IntegerArray::<i64>::default();
        assert_eq!(arr.get(0), None);
        assert_eq!(arr.get(100), None);
    }

    #[test]
    fn test_null_mask_replace() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(9);
        let mut mask = Bitmask::new_set_all(1, false);
        mask.set(0, true);
        arr.set_null_mask(Some(mask));
        assert!(!arr.is_null(0));
    }

    #[test]
    fn test_integer_array_slice() {
        let mut arr = IntegerArray::<i32>::default();
        arr.push(10);
        arr.push(20);
        arr.push(30);
        arr.push_null();
        arr.push(50);

        let sliced = arr.slice_clone(1, 3);
        assert_eq!(*sliced.data(), vec64![20, 30, 0]); // default for null = 0
        assert_eq!(sliced.len(), 3);
        assert_eq!(sliced.get(0), Some(20));
        assert_eq!(sliced.get(1), Some(30));
        assert_eq!(sliced.get(2), None); // was null
        assert_eq!(sliced.null_count(), 1);
    }
}

#[cfg(test)]
#[cfg(feature = "parallel_proc")]
mod parallel_tests {
    use rayon::prelude::*;

    use super::*;
    use crate::structs::bitmask::Bitmask;
    use crate::vec64;

    #[test]
    fn test_integerarray_par_iter() {
        let mut arr = IntegerArray::<i32>::from_slice(&[1, 2, 3, 4]);
        // Only idx 1 valid
        let mut mask = Bitmask::new_set_all(4, false);
        mask.set(1, true);
        arr.null_mask = Some(mask);
        let vals: Vec<i32> = arr.par_iter().map(|v| *v).collect();
        assert_eq!(vals, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_integerarray_par_iter_opt() {
        let mut arr = IntegerArray::<i32>::from_slice(&[1, 2, 3, 4]);
        let mut mask = Bitmask::new_set_all(4, false);
        mask.set(1, true);
        arr.null_mask = Some(mask);
        let opt: Vec<Option<i32>> = arr.par_iter_opt().map(|opt| opt.copied()).collect();
        assert_eq!(opt, vec![None, Some(2), None, None]);
    }

    #[test]
    fn test_integerarray_par_iter_mut() {
        let mut arr = IntegerArray::<i32>::from_slice(&[10, 20, 30, 40]);
        arr.par_iter_mut().for_each(|v| *v += 1);
        let expected = vec![11, 21, 31, 41];
        let actual: Vec<i32> = arr.par_iter().map(|v| *v).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_integerarray_par_iter_range_unchecked() {
        let arr = IntegerArray::<i32>::from_slice(&[1, 2, 3, 4, 5, 6]);
        let out: Vec<&i32> = unsafe { arr.par_iter_range_unchecked(2, 5).collect() };
        assert_eq!(*out[0], 3);
        assert_eq!(*out[1], 4);
        assert_eq!(*out[2], 5);
    }

    #[test]
    fn test_integerarray_par_iter_range_opt_unchecked() {
        let mut arr = IntegerArray::<i32>::from_slice(&[1, 2, 3, 4, 5]);
        let mut mask = Bitmask::new_set_all(5, false);
        mask.set(0, true);
        mask.set(2, true);
        mask.set(4, true);
        arr.null_mask = Some(mask);
        let out: Vec<Option<&i32>> = unsafe { arr.par_iter_range_opt_unchecked(0, 5).collect() };
        assert_eq!(
            out.iter().map(|x| x.copied()).collect::<Vec<_>>(),
            vec![Some(1), None, Some(3), None, Some(5)]
        );
    }

    #[test]
    fn test_integer_array_append_array() {
        use crate::traits::masked_array::MaskedArray;

        // Base: [10, 20, 30] (no nulls)
        let mut arr1 = IntegerArray::<i16>::from_slice(&[10, 20, 30]);
        assert_eq!(arr1.null_mask(), None);

        // Append: [40, 50] (no nulls)
        let arr2 = IntegerArray::<i16>::from_slice(&[40, 50]);
        arr1.append_array(&arr2);
        assert_eq!(*arr1.data(), vec64![10, 20, 30, 40, 50]);
        assert_eq!(arr1.null_mask(), None);
        assert_eq!(arr1.len(), 5);
        assert_eq!(arr1.get(0), Some(10));
        assert_eq!(arr1.get(4), Some(50));

        // Now, add nulls to both and append
        let mut arr3 = IntegerArray::<i16>::with_capacity(3, true);
        arr3.push(60); // valid
        arr3.push_null(); // null
        arr3.push(70); // valid

        let mut arr4 = IntegerArray::<i16>::with_capacity(2, true);
        arr4.push_null(); // null
        arr4.push(80); // valid

        arr3.append_array(&arr4);
        // arr3: [60, None, 70, None, 80]
        assert_eq!(arr3.len(), 5);
        let vals: Vec<Option<i16>> = (0..arr3.len()).map(|i| arr3.get(i)).collect();
        assert_eq!(vals, vec![Some(60), None, Some(70), None, Some(80)]);
        // Null mask correct
        let mask = arr3.null_mask().unwrap();
        assert!(mask.get(0));
        assert!(!mask.get(1));
        assert!(mask.get(2));
        assert!(!mask.get(3));
        assert!(mask.get(4));
        assert_eq!(arr3.null_count(), 2);

        // Append from maskless into masked
        let mut arr5 = IntegerArray::<i16>::with_capacity(2, true);
        arr5.push(90);
        arr5.push(91);
        let arr6 = IntegerArray::<i16>::from_slice(&[92, 93]);
        arr5.append_array(&arr6);
        assert_eq!(*arr5.data(), vec64![90, 91, 92, 93]);
        assert!(arr5.null_mask().unwrap().all_set());

        // Append from masked into maskless
        let mut arr7 = IntegerArray::<i16>::from_slice(&[100, 101]);
        let mut arr8 = IntegerArray::<i16>::with_capacity(2, true);
        arr8.push_null();
        arr8.push(103);
        arr7.append_array(&arr8);
        // arr7: [100, 101, None, 103]
        assert_eq!(arr7.get(2), None);
        assert_eq!(arr7.get(3), Some(103));
        assert_eq!(arr7.null_count(), 1);
    }
}
