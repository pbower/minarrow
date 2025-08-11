use crate::{Array, ArrayV, Length, MaskedArray, Offset};

// Zero-copy sliced views into arrays with multiple abstraction levels.
///
/// Provides three view types:
/// - **Native Rust Slice**: `&[T]` for fixed-width types, `&[u8]` for variable-width types
/// - **ArrayView**: Zero-copy `Arc`-cloned reference with simplified windowed indexing
/// - **TupleView**: Minimal `(&Array, offset, length)` tuple for maximum performance
///
/// `ArrayView` provides semantic type variants (`.num()`, `.text()`, `.dt()`, `.bool()`) 
/// enabling `impl Into<NumericalArrayView>` function signatures that accept both full arrays 
/// and views with zero-copy conversion between compatible types.
pub trait View: MaskedArray + Into<Array> + Clone
where
    <Self as MaskedArray>::Container: AsRef<[Self::BufferT]>,
{

    /// The fixed-width buffer type (e.g. `u8`, `f32`, `bool`, etc.)
    type BufferT: Default + PartialEq + Clone + Copy + Sized;

    /// Returns the whole array buffer as a &[T] slice.
    fn as_slice(&self) -> &[Self::BufferT] {
        self.data().as_ref()
    }

    /// Slices the data values from offset to offset + length.
    ///
    /// # Parameters
    /// - `offset`: Starting index.
    /// - `len`: Number of elements.
    ///
    /// # Returns
    /// A native slice of the data values.
    fn slice(&self, offset: usize, len: usize) -> (&[Self::BufferT], Offset, Length) {
        (&self.data().as_ref()[offset..offset + len], offset, len)
    }
    
    /// Returns a zero-copy, windowed view (`ArrayView`) into this array.
    ///
    /// ## Ownership Semantics
    /// - For `Arc`-wrapped arrays (e.g., `Array`), this method consumes the `Arc`. 
    ///   If you need to retain access to the original array after calling `view`, 
    ///   clone the `Arc` at the call site (cheap pointer clone).
    /// - For direct array variants (e.g., `IntegerArray<u64>`), calling `view` consumes ownership.
    ///   If continued access to the original is required, promote the variant into an `Arc` (or `Array`) first.
    ///
    /// ## Behaviour
    /// - The returned `ArrayView` enforces the window (`offset`, `length`) and provides safe, zero-copy access.
    /// - The backing array remains accessible via `&Array` within the view.
    /// - Type-specific accessors (`.num()`, `.str()`, `.bool()`, `.dt()`) are available on `ArrayView` for ergonomic, typed access.
    ///
    /// # Implementation Notes
    /// - Unlike `Apache Arrow`, where arrays are trait-based views over reference-counted
    ///   buffers, `MinArrow` uses a concrete `Array` struct which already holds a *zero-copy*
    ///   `Arc` reference to its inner array data.
    /// - `ArrayView` therefore provides a *windowed zero-copy view* into a specific region of
    ///   this array.
    ///
    /// # Parameters
    /// - `offset`: The logical start index for the window.
    /// - `len`:    The logical length of the window.
    ///
    /// # Returns
    /// An `ArrayView` representing the specified logical window.
    fn view(self, offset: usize, len: usize) -> ArrayV {
        ArrayV::new(self.into(), offset, len)
    }

    /// Returns a tuple view of (&Array, offset, length).
    ///
    /// # Parameters
    /// - `offset`: Starting index.
    /// - `length`: Number of elements.
    ///
    /// # Returns
    /// Tuple containing array reference, offset, and length.
    fn tuple_view(self, offset: usize, len: usize) -> (Array, usize, usize) {
        self.view(offset, len).as_tuple()
    }
}
