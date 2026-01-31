#!/usr/bin/env python3
"""PyCapsule Exchange Demo

Shows how the Arrow PyCapsule Interface lets Python consume data produced
 in Rust, without any PyArrow-specific glue code.

    PYCAPSULE - is the standard protocol - for us with Python Arrow libraries:
        # Rust returns an object implementing __arrow_c_stream__
        stream = minarrow_pyo3.generate_sample_batch()
        # Any Arrow-compatible library can consume it:
        reader = pa.RecordBatchReader.from_stream(stream)   # PyArrow
        # or: nanoarrow.ArrayStream(stream)                  # nanoarrow
        # or: pl.from_arrow(reader.read_all())               # Polars

    ARROW C API (this is largely superceded by PyCapsule in current Python versions):
        # Rust exports raw memory addresses as integers
        array_ptr, schema_ptr = some_ffi_call()
        arr = pa.Array._import_from_c(array_ptr, schema_ptr)
        # Only works with PyArrow. Polars, DuckDB, nanoarrow prefer PyCapsule.

Prerequisites:
    pip install maturin pyarrow
    cd pyo3 && maturin develop

Run:
    python examples/pycapsule_demo.py
"""

import minarrow_pyo3 as ma


def demo_stream_capsule():
    """Rust generates a table and returns it via __arrow_c_stream__.

    The Python side never touches minarrow types - it receives an object
    implementing the Arrow PyCapsule protocol and reads it with PyArrow's
    public API.
    """
    print("1. Stream PyCapsule: Rust table -> Python")
    print("   " + "-" * 45)

    # Rust builds the data and returns a protocol-conforming object
    stream = ma.generate_sample_batch()
    print(f"   Received: {type(stream).__name__}")

    # PyArrow consumes it via the standard protocol
    import pyarrow as pa

    reader = pa.RecordBatchReader.from_stream(stream)
    table = reader.read_all()

    print(f"   Schema: {table.schema}")
    print(f"   Rows:   {table.num_rows}")
    print(f"   Data:")
    for name in table.column_names:
        col = table.column(name)
        print(f"     {name}: {col.to_pylist()}")
    print()


def demo_array_capsule():
    """Rust generates a nullable array and returns it via __arrow_c_array__.

    PyArrow's pa.array() automatically calls __arrow_c_array__ on the
    object - no raw pointer extraction needed.
    """
    print("2. Array PyCapsule: Rust nullable array -> Python")
    print("   " + "-" * 45)

    # Rust builds a nullable int64 array and returns a protocol wrapper
    wrapper = ma.generate_nullable_array()
    print(f"   Received: {type(wrapper).__name__}")

    # PyArrow imports via the standard protocol - just pass the object
    import pyarrow as pa

    arr = pa.array(wrapper)
    print(f"   Type:   {arr.type}")
    print(f"   Values: {arr.to_pylist()}")
    print(f"   Nulls:  {arr.null_count}")
    print()


def demo_roundtrip_via_capsule():
    """Full roundtrip: Python -> Rust -> PyCapsule -> Python.

    Shows how export_batch_stream_capsule returns a stream capsule
    that any Arrow consumer can read.
    """
    print("3. Roundtrip: PyArrow -> Rust -> PyCapsule -> PyArrow")
    print("   " + "-" * 45)

    import pyarrow as pa

    # Start with a PyArrow batch
    original = pa.record_batch(
        [
            pa.array(["Alice", "Bob", "Charlie"]),
            pa.array([25, 30, 35], type=pa.int32()),
        ],
        names=["name", "age"],
    )
    print(f"   Original: {original.num_rows} rows, schema={original.schema}")

    # Send through Rust and get back as a PyCapsule stream
    capsule = ma.export_batch_stream_capsule(original)

    # Read it back - this could be any Arrow-compatible library
    reader = pa.RecordBatchReader.from_stream(capsule)
    result = reader.read_all()

    print(f"   Result:   {result.num_rows} rows, schema={result.schema}")
    for name in result.column_names:
        print(f"     {name}: {result.column(name).to_pylist()}")
    print()


def demo_nanoarrow():
    """Optional: consume a PyCapsule with nanoarrow if installed."""
    print("4. nanoarrow consumption (optional)")
    print("   " + "-" * 45)
    try:
        import nanoarrow as na
    except ImportError:
        print("   Skipped (nanoarrow not installed: pip install nanoarrow)")
        print()
        return

    stream = ma.generate_sample_batch()
    na_stream = na.ArrayStream(stream)
    print(f"   Schema: {na_stream.schema}")

    result = na_stream.read_all()
    print(f"   Chunks: {result.n_chunks}, Children: {result.n_children}")
    print(f"   Data:   {result.to_pylist()}")
    print()


def demo_polars():
    """Optional: consume a PyCapsule with Polars if installed."""
    print("5. Polars consumption (optional)")
    print("   " + "-" * 45)
    try:
        import polars as pl
    except ImportError:
        print("   Skipped (polars not installed: pip install polars)")
        print()
        return

    import pyarrow as pa

    # Rust generates data as a stream capsule
    stream = ma.generate_sample_batch()

    # Read into PyArrow first, then hand to Polars
    reader = pa.RecordBatchReader.from_stream(stream)
    arrow_table = reader.read_all()
    df = pl.from_arrow(arrow_table)

    print(f"   DataFrame: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df)
    print()


if __name__ == "__main__":
    print("=" * 55)
    print("  Arrow PyCapsule Interface Demo")
    print("  Data produced in Rust, consumed in Python")
    print("=" * 55)
    print()

    demo_stream_capsule()
    demo_array_capsule()
    demo_roundtrip_via_capsule()
    demo_nanoarrow()
    demo_polars()

    print("Done.")
