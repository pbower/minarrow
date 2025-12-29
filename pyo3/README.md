# minarrow-pyo3

PyO3 bindings for MinArrow - zero-copy Arrow interop with Python via PyArrow.

## Overview

This crate provides transparent wrapper types that enable seamless conversion between MinArrow's Rust types and PyArrow's Python types using the Arrow C Data Interface.

## Type Mappings

| MinArrow | PyArrow | Wrapper Type |
|----------|---------|--------------|
| `Array` | `pa.Array` | `PyArray` |
| `Table` | `pa.RecordBatch` | `PyRecordBatch` |

## Installation

### Prerequisites

- Python 3.9+
- PyArrow 14+
- Rust nightly (for MinArrow)
- maturin

```bash
pip install maturin pyarrow
```

### Building

```bash
cd minarrow-pyo3
maturin develop
```

For a release build:
```bash
maturin build --release
```

## Usage

### Rust Side

Create PyO3 functions that accept and return PyArrow types:

```rust
use minarrow_pyo3::{PyArray, PyRecordBatch};
use minarrow::{Array, Table, IntegerArray, MaskedArray};
use pyo3::prelude::*;

#[pyfunction]
fn double_values(input: PyArray) -> PyResult<PyArray> {
    // Access the MinArrow Array
    let array = input.inner();

    // Process... (example: clone and return)
    Ok(PyArray::from(array.clone()))
}

#[pyfunction]
fn process_batch(input: PyRecordBatch) -> PyResult<PyRecordBatch> {
    let table: Table = input.into();
    // Process the table...
    Ok(PyRecordBatch::from(table))
}

#[pymodule]
fn my_extension(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double_values, m)?)?;
    m.add_function(wrap_pyfunction!(process_batch, m)?)?;
    Ok(())
}
```

### Python Side

```python
import pyarrow as pa
import my_extension

# Array roundtrip
arr = pa.array([1, 2, 3, 4, 5], type=pa.int32())
result = my_extension.double_values(arr)
print(result)  # PyArrow array

# RecordBatch roundtrip
batch = pa.RecordBatch.from_pydict({
    "id": [1, 2, 3],
    "name": ["alpha", "beta", "gamma"]
})
result = my_extension.process_batch(batch)
print(result)  # PyArrow RecordBatch
```

## Features

- `datetime` - Enable datetime/temporal type support
- `extended_numeric_types` - Enable i8, i16, u8, u16 types
- `extended_categorical` - Enable Categorical8, Categorical16, Categorical64

## Testing

### Python Tests

Run the comprehensive Python test suite:

```bash
cd pyo3
python3 -m venv .venv
source .venv/bin/activate
pip install pyarrow maturin
maturin develop
python test_roundtrip.py
```

### Rust Tests

Run the Rust roundtrip tests. These require special setup because PyO3's `extension-module`
feature (default) doesn't link against libpython.

```bash
cd pyo3

# 1. Find which Python library the binary links against:
cargo build --example run_tests --no-default-features --features "datetime,extended_numeric_types,extended_categorical"
ldd target/debug/examples/run_tests | grep python

# 2. Set PYTHONHOME to that Python's prefix:
#    e.g., if it links to /usr/local/lib/libpython3.12.so, use PYTHONHOME=/usr/local
#    You can verify with: /usr/local/bin/python3.12 -c "import sys; print(sys.prefix)"

# 3. Run the tests:
PYTHONHOME=/usr/local cargo run --example run_tests \
    --no-default-features \
    --features "datetime,extended_numeric_types,extended_categorical"
```

The `--no-default-features` disables `extension-module`, allowing the binary to link
against libpython for standalone execution.

## Architecture

The bindings use the Arrow C Data Interface for zero-copy data transfer:

1. **Rust -> Python**: MinArrow's `export_to_c()` exports to Arrow C format, PyArrow's `_import_from_c()` imports it
2. **Python -> Rust**: PyArrow's `_export_to_c()` exports, MinArrow's `import_from_c()` imports

Memory is managed through reference counting - the Arrow release callbacks ensure proper cleanup when either side releases the data.

## License

MIT
