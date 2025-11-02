//! # **Selection Traits** - *Extensible selection across dimensions*
//!
//! Traits for field and data selection that enable polymorphic methods
//! across Table, TableV, Cube, and future types.
//!
//! ## Architecture
//! - **FieldSelector**: Input types that can specify field selection (e.g., `&str`, `usize`)
//! - **DataSelector**: Input types that can specify data selection (e.g., `usize`, ranges)
//! - **FieldSelection**: Capability trait for types that support field selection
//! - **DataSelection**: Capability trait for types that support data selection
//! - **Selection2D**: Combined 2D selection (FieldSelection + DataSelection)
//! - **Selection3D**: Extension for 3D selection
//! - **Selection4D**: Extension for 4D selection

use std::ops::{Range, RangeFrom, RangeFull, RangeTo, RangeInclusive};
use std::sync::Arc;
use crate::Field;

// Input types that can be passed to selection methods
// ===================================================
// These traits are implemented on user-facing input types like `&str`, `usize`, and ranges.
// They convert user input (e.g., `table.f("name")` or `table.d(0..10)`) into index vectors.
// These are "what the user writes" when selecting.

/// Trait for types that can specify a field selection (named/schema dimension)
pub trait FieldSelector {
    /// Resolve this selection to field indices for the given fields
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize>;
}

/// Trait for types that can specify a data selection (index-based dimension)
pub trait DataSelector {
    /// Resolve this selection to indices (within the given count)
    fn resolve_indices(&self, count: usize) -> Vec<usize>;
}

// Data structures that provide selection methods
// ===============================================
// These traits are implemented on structures like Table, ArrayV, etc.
// They define what selection methods are available on each structure.
// These are "what the structure can do" for selection operations.

/// Trait for types that support field selection
pub trait FieldSelection {
    /// The view type returned by selection operations
    type View;

    /// Select fields (columns) by name or index
    fn f<S: FieldSelector>(&self, selection: S) -> Self::View;

    /// Explicit alias for `.f()` - select fields
    fn fields<S: FieldSelector>(&self, selection: S) -> Self::View {
        self.f(selection)
    }

    /// Spatial alias for `.f()` - Y-axis (vertical, schema dimension)
    fn y<S: FieldSelector>(&self, selection: S) -> Self::View {
        self.f(selection)
    }

    /// Get the fields for field resolution
    fn get_fields(&self) -> Vec<Arc<Field>>;
}

/// Trait for types that support data selection
pub trait DataSelection {
    /// The view type returned by selection operations
    type View;

    /// Select data (rows) by index or range
    fn d<S: DataSelector>(&self, selection: S) -> Self::View;

    /// Explicit alias for `.d()` - select data
    fn data<S: DataSelector>(&self, selection: S) -> Self::View {
        self.d(selection)
    }

    /// Spatial alias for `.d()` - X-axis (horizontal, data dimension)
    fn x<S: DataSelector>(&self, selection: S) -> Self::View {
        self.d(selection)
    }

    /// Get the count for data resolution
    fn get_data_count(&self) -> usize;
}

/// Combined trait for 2D selection (field + data dimensions)
///
/// This trait is automatically implemented for any type that implements
/// both `FieldSelection` and `DataSelection` with the same `View` type.
pub trait Selection2D: FieldSelection + DataSelection {}

/// Blanket implementation for any type that implements both traits
impl<T> Selection2D for T
where
    T: FieldSelection + DataSelection<View = <T as FieldSelection>::View>,
{}

/// Extension trait for 3D selection (time/batch dimension)
///
/// TODO: Implement for Cube and CubeV when those types are added
pub trait Selection3D: Selection2D {
    /// Select along the time/batch dimension
    fn t<S: DataSelector>(&self, selection: S) -> <Self as FieldSelection>::View;

    /// Explicit alias for `.t()` - select time
    fn time<S: DataSelector>(&self, selection: S) -> <Self as FieldSelection>::View {
        self.t(selection)
    }

    /// Spatial alias for `.t()` - Z-axis (depth, time dimension)
    fn z<S: DataSelector>(&self, selection: S) -> <Self as FieldSelection>::View {
        self.t(selection)
    }

    /// Get the count for time resolution
    fn get_time_count(&self) -> usize;
}

/// Extension trait for 4D selection (space/semantic dimension)
///
/// TODO: Implement for 4D structures when those types are added
pub trait Selection4D: Selection3D {
    /// Select along the space/semantic dimension
    fn s<S: DataSelector>(&self, selection: S) -> <Self as FieldSelection>::View;

    /// Explicit alias for `.s()` - select space
    fn space<S: DataSelector>(&self, selection: S) -> <Self as FieldSelection>::View {
        self.s(selection)
    }

    /// Get the count for space resolution
    fn get_space_count(&self) -> usize;
}

// FieldSelector implementations for common input types
// ====================================================
// These allow users to pass names, indices, and ranges when selecting fields.
// For example: table.f("age"), table.f(&["name", "age"]), table.f(0..3)

/// Single field by name
impl FieldSelector for &str {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        fields.iter()
            .position(|f| f.name == *self)
            .into_iter()
            .collect()
    }
}

/// Multiple fields by names
impl FieldSelector for &[&str] {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        self.iter()
            .filter_map(|name| fields.iter().position(|f| f.name == *name))
            .collect()
    }
}

/// Multiple fields by names (array reference)
impl<const N: usize> FieldSelector for &[&str; N] {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        self.iter()
            .filter_map(|name| fields.iter().position(|f| f.name == *name))
            .collect()
    }
}

/// Multiple fields by names (Vec)
impl FieldSelector for Vec<&str> {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        self.iter()
            .filter_map(|name| fields.iter().position(|f| f.name == *name))
            .collect()
    }
}

/// Single field by index
impl FieldSelector for usize {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        if *self < fields.len() {
            vec![*self]
        } else {
            Vec::new()
        }
    }
}

/// Multiple fields by indices
impl FieldSelector for &[usize] {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        self.iter()
            .copied()
            .filter(|&idx| idx < fields.len())
            .collect()
    }
}

/// Multiple fields by indices (array reference)
impl<const N: usize> FieldSelector for &[usize; N] {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        self.iter()
            .copied()
            .filter(|&idx| idx < fields.len())
            .collect()
    }
}

/// Multiple fields by indices (Vec)
impl FieldSelector for Vec<usize> {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        self.iter()
            .copied()
            .filter(|&idx| idx < fields.len())
            .collect()
    }
}

/// Field range selection
impl FieldSelector for Range<usize> {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        let end = self.end.min(fields.len());
        (self.start..end).collect()
    }
}

/// Field range from selection
impl FieldSelector for RangeFrom<usize> {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        (self.start..fields.len()).collect()
    }
}

/// Field range to selection
impl FieldSelector for RangeTo<usize> {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        let end = self.end.min(fields.len());
        (0..end).collect()
    }
}

/// Field full range selection
impl FieldSelector for RangeFull {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        (0..fields.len()).collect()
    }
}

/// Field inclusive range selection
impl FieldSelector for RangeInclusive<usize> {
    fn resolve_fields(&self, fields: &[Arc<Field>]) -> Vec<usize> {
        let start = *self.start();
        let end = (*self.end() + 1).min(fields.len());
        (start..end).collect()
    }
}

// DataSelector implementations for common input types
// ===================================================
// These allow users to pass indices and ranges when selecting data (rows, time, etc.).
// For example: table.d(5), table.d(&[1, 3, 5]), table.d(0..10)

/// Single data index
impl DataSelector for usize {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        if *self < count {
            vec![*self]
        } else {
            Vec::new()
        }
    }
}

/// Multiple data indices
impl DataSelector for &[usize] {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        self.iter()
            .copied()
            .filter(|&idx| idx < count)
            .collect()
    }
}

/// Multiple data indices (array reference)
impl<const N: usize> DataSelector for &[usize; N] {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        self.iter()
            .copied()
            .filter(|&idx| idx < count)
            .collect()
    }
}

/// Multiple data indices (Vec)
impl DataSelector for Vec<usize> {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        self.iter()
            .copied()
            .filter(|&idx| idx < count)
            .collect()
    }
}

/// Data range selection
impl DataSelector for Range<usize> {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        let end = self.end.min(count);
        (self.start..end).collect()
    }
}

/// Data range from selection
impl DataSelector for RangeFrom<usize> {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        (self.start..count).collect()
    }
}

/// Data range to selection
impl DataSelector for RangeTo<usize> {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        let end = self.end.min(count);
        (0..end).collect()
    }
}

/// Data full range selection
impl DataSelector for RangeFull {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        (0..count).collect()
    }
}

/// Data inclusive range selection
impl DataSelector for RangeInclusive<usize> {
    fn resolve_indices(&self, count: usize) -> Vec<usize> {
        let start = *self.start();
        let end = (*self.end() + 1).min(count);
        (start..end).collect()
    }
}
