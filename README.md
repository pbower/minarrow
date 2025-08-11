# Minarrow

_Benefits without Baggage_.

## Introduction

_Welcome to Minarrow_.

Minarrow is a from-scratch columnar library built for real-time and systems workloads in Rust.  
It keeps the surface small, makes types explicit, compiles fast, and aligns data for predictable SIMD performance.  
It speaks Arrow when you need to interchange — but the core stays lean.  

## Design Priorities

- **Typed, direct access** – No downcasting chains  
- **Predictable performance** – 64-byte alignment by default  
- **Fast iteration** – Minimal dependencies, sub-1.5 s clean builds, <0.15 s rebuilds  
- **Interoperability on demand** – Convert to and from Arrow at the boundary  

## Why This Exists

The Arrow format is a powerful standard for columnar data.  
Apache Arrow has driven an entire ecosystem forward, with zero-copy interchange, multi-language support, and extensive integration.  
Minarrow complements that ecosystem by focusing on Rust-first ergonomics, predictable SIMD behaviour, and extremely low build-time friction.

## Key Features

### Fast Compilation

Minarrow compiles in under 1.5 seconds with default features, minimising development iteration time with < 0.15s rebuilds. This is achieved through minimal dependencies - primarily `num-traits` with optional `rayon` for parallelism.

### Data access

Minarrow provides direct, always-typed access to array values. Unlike other Rust implementations that unify all array types as untyped byte buffers *(requiring downcasting and dynamic checks)*, Minarrow retains concrete types throughout the API. This enables developers to inspect and manipulate data without downcasting or additional indirection, ensuring safe and ergonomic access at all times.

## Type System

Six concrete array types cover common workloads:  

- BooleanArray<u8>  
- CategoricalArray<T>  
- DatetimeArray<T>  
- FloatArray<T>  
- IntegerArray<T>  
- StringArray<T>  

Unified views:  

- NumericArray<T>  
- TextArray<T>  
- TemporalArray<T>  

And a single top-level Array for mixed tables. 


## Example: Create Arrays

```rust
use std::sync::Arc;  
use minarrow::{Array, IntegerArray, NumericArray, arr_bool, arr_cat32, arr_f64, arr_i32, arr_i64, arr_str32, vec64};  

let int_arr = arr_i32![1, 2, 3, 4];  
let float_arr = arr_f64![0.5, 1.5, 2.5];  
let bool_arr = arr_bool![true, false, true];  
let str_arr = arr_str32!["a", "b", "c"];  
let cat_arr = arr_cat32!["x", "y", "x", "z"];  

assert_eq!(int_arr.len(), 4);  
assert_eq!(str_arr.len(), 3);  

let int = IntegerArray::<i64>::from_slice(&[100, 200]);  
let wrapped: NumericArray = NumericArray::Int64(Arc::new(int));  
let array = Array::NumericArray(wrapped);  
```

## Example: Build Table

```rust
use minarrow::{FieldArray, Print, Table, arr_i32, arr_str32, vec64};  

let col1 = FieldArray::from_inner("numbers", arr_i32![1, 2, 3]);  
let col2 = FieldArray::from_inner("letters", arr_str32!["x", "y", "z"]);  

let mut tbl = Table::new("Demo".into(), vec![col1, col2].into());  
tbl.print();  

See _examples/_ for more.
```

When working with arrays, remember to import the `MaskedArray` trait,
which ensures all required methods are available.

## SIMD by Default

- All buffers use 64-byte-aligned allocation from ingestion through processing. No reallocation step to fix alignment.  
- Stable vectorised behaviour on modern CPUs via `Vec64` with a custom allocator.
- The companion `Lightstream-IO` crate provides IPC readers and writers that maintain this alignment, avoiding reallocation overhead during data ingestion.

### Enum-Based Architecture

Minarrow uses enums for type dispatch instead of trait object downcasting, providing:

- **Performance** – Enables aggressive compiler inlining and optimisation  
- **Maintainability** – Centralised, predictable dispatch logic  
- **Type Safety** – All types are statically known; no `Any` or runtime downcasts  
- **Ergonomics** – Direct, typed accessors such as `myarray.num().i64()`  

The structure is layered:

1. **Top-level `Array` enum** – Arc-wrapped for zero-copy sharing  
2. **Semantic groupings**:  
   - `NumericArray` – All numeric types in one variant set  
   - `TextArray` – String and categorical data  
   - `TemporalArray` – All date/time variants  
   - `BooleanArray` – Boolean data  

This design supports flexible function signatures like `impl Into<NumericArray>` while preserving static typing.  
Because dispatch is static, the compiler retains full knowledge of types across calls, enabling inlining and eliminating virtual call overhead.

### Flexible Integration
- **Apache Arrow** compatibility via `.to_apache_arrow()`.
- **Polars** compatibility via `.to_polars()`.
- **Async support** through the companion Lightstream crate
- **Zero-copy** views for windowed operations *(see Views and Windowing below)*
- **Memory-mapped** file support with maintained SIMD alignment
- **FFI Support** for cross-language compatibility.

Lightstream *(planned Aug ’25)* enables IPC streaming in Tokio async contexts with composable encoder/decoder traits, both sync and async, without losing SIMD alignment.

## Views and Windowing

- Optional view variants provide zero-copy windowed access to arrays, and encode offset and length for efficient subset operations without copying.  
- For extreme performance needs, minimal tuple aliases `(&InnerArrayVariant, offset, len)` are available.

## Benchmarks

***Sum of 1 billion sequential integers starting at 0.***

Intel(R) Core(TM) Ultra 7 155H | x86_64 | 22 CPUs  
Averaged over 1,000 runs (release).  

### No SIMD 

***(n=1000, lanes=4, iters=1000)***

| Case                               | Avg time  |
|------------------------------------|-----------|
| **Integer (i64)**                  |           |
| raw vec: `Vec<i64>`                | 85 ns     |
| minarrow direct: `IntegerArray`    | 88 ns     |
| arrow-rs struct: `Int64Array`      | 147 ns    |
| minarrow enum: `IntegerArray`      | 124 ns    |
| arrow-rs dyn: `Int64Array`         | 181 ns    |
| **Float (f64)**                    |           |
| raw vec: `Vec<f64>`                | 475 ns    |
| minarrow direct: `FloatArray`      | 476 ns    |
| arrow-rs struct: `Float64Array`    | 527 ns    |
| minarrow enum: `FloatArray`        | 507 ns    |
| arrow-rs dyn: `Float64Array`       | 1.952 µs  |

### SIMD 

***(n=1000, lanes=4, iters=1000)***

| Case                                  | Avg (ns) |
|---------------------------------------|---------:|
| raw vec: `Vec<i64>`                   | 64       |
| raw vec64: `Vec64<i64>`               | 55       |
| minarrow direct: `IntegerArray`       | 88       |
| arrow-rs struct: `Int64Array`         | 162      |
| minarrow enum: `IntegerArray`         | 170      |
| arrow-rs dyn: `Int64Array`            | 173      |
| raw vec: `Vec<f64>`                   | 57       |
| raw vec64: `Vec64<f64>`               | 58       |
| minarrow direct: `FloatArray`         | 91       |
| arrow-rs struct: `Float64Array`       | 181      |
| minarrow enum: `FloatArray`           | 180      |
| arrow-rs dyn: `Float64Array`          | 196      |

### SIMD + Rayon

***(n=1000, lanes=4, iters=1000)***

| Case                                    | Time (ms)   |
|-----------------------------------------|-------------|
| SIMD + Rayon `IntegerArray<i64>`        | 113.874     |
| SIMD + Rayon `FloatArray<f64>`          | 114.095     |

## Ideal Use Cases

| Use Case                              | Description |
|---------------------------------------|-------------|
| Real-time Data Pipelines              | Zero-copy interchange for streaming and event-driven systems |
| Embedded and Edge Computing           | Minimal deps, predictable memory layout, fast compile |
| Systems-Level Integration             | 64-byte alignment and FFI-friendly representation |
| High-Performance Analytics            | SIMD kernels and direct buffer access |
| Rapid Prototyping and Development     | Simple type system, intuitive APIs |
| Data-Intensive Rust Applications      | Rust-native data structures with no runtime abstraction penalty |
| Extreme Latency Scenarios             | Inner types for trading, defence, and other nanosecond-sensitive systems |

## Design Philosophy

- **Direct data access** over abstraction layers
- **Developer velocity** through simple, predictable APIs
- **Performance** with SIMD as a first-class citizen
- **Interoperability** while maintaining its own identity

This approach trades some features (like deeply nested types) for a more streamlined experience in common data processing scenarios.

## Contributing

We welcome contributions! Areas of focus include:

1. **Connectors** - Connectors to data sources and sinks
2. **Optimisations** - Performance improvements that maintain API simplicity
3. **Bug Fixes** - Bug fixes and improvements
4. **PyO3 Integration** - Python bindings and interoperability
5. **List and Struct Types** - Support for nested types (PRs welcome)
6. **Datetime** - Improving datetime ergonomics.

Additionally, if you are interested in working on a SIMD kernels crate that's in development,
and have relevant experience, please feel free to reach out.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for contributing guidelines.

## License

This project is licensed under the `MIT License`. See _LICENSE_ for details.

## Acknowledgments

Special thanks to the Apache Arrow community and all contributors to the Arrow ecosystem. Minarrow is inspired by the excellent work and standards established by these projects.

## Feedback

We value community input and would appreciate your thoughts and feedback. Please don't hesitate to reach out with questions, suggestions, or contributions.
