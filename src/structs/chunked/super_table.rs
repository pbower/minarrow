//! # **SuperTable** - *Holds multiple Tables for chunked data partitioning, streaming + fast memIO*
//!
//! SuperTable groups multiple `Table` batches under a shared schema.
//!
//! ## Purpose
//! - Treats an ordered sequence of `Table` batches as one dataset.
//! - Preserves per-batch independence while allowing unified export over Arrow FFI.
//! - Enables both bounded and unbounded (Live / streaming) workflows.
//!
//! ## Behaviour
//! - All batches must have identical column definitions (`Field` metadata).
//! - Row counts may differ between batches, but not between inner arrays.
//! - When sent via Arrow FFI, exposed as a single logical table.
//! - Supports concatenation into a materialised `Table` on demand.
//!
//! ## Typical Uses
//! - Partitioned storage readers (e.g., multiple Arrow IPC files).
//! - Streaming ingestion into append-only datasets.
//! - Windowed or mini-batch analytics.
//! - Incremental build-up of tables for later unification.

use std::fmt::{Display, Formatter};

use std::iter::FromIterator;
use std::sync::Arc;

use crate::enums::{error::MinarrowError, shape_dim::ShapeDim};
use crate::structs::chunked::super_array::RechunkStrategy;
use crate::structs::field::Field;
use crate::structs::field_array::FieldArray;
use crate::structs::table::Table;
#[cfg(feature = "size")]
use crate::traits::byte_size::ByteSize;
use crate::traits::concatenate::Concatenate;
use crate::traits::shape::Shape;
#[cfg(feature = "views")]
use crate::{SuperTableV, TableV};

/// # SuperTable
///
/// Higher-order container representing a sequence of `Table` batches with consistent schema.
///
/// ## Overview
/// - Each batch is a `Table` (record batch) with identical column metadata.
/// - Stored as `Vec<Arc<Table>>`, preserving order and schema consistency.
/// - Row counts per batch may vary, but are consistent across all Table columns.
/// - When exported via Arrow FFI, the batches are viewed as a single logical table.
/// - Useful for open-ended streams, partitioned datasets, or
///   other scenarios where batches are processed independently.
///
/// ## Fields
/// - `batches`: ordered collection of `Table` batches.
/// - `schema`: cached schema from the first batch for fast access.
/// - `n_rows`: total row count across all batches.
/// - `name`: super table name.
///
/// ## Use cases
/// - Streaming and mini-batch processing.
/// - Reading multiple Arrow IPC/memory-mapped files as one dataset.
/// - Parallel or windowed in-memory analytics.
/// - Incremental table construction where batches arrive over time.
#[derive(Clone, Debug, PartialEq)]
pub struct SuperTable {
    pub batches: Vec<Arc<Table>>,
    pub schema: Vec<Arc<Field>>,
    pub n_rows: usize,
    pub name: String,
}

impl SuperTable {
    /// Creates a new empty BatchedTable with a specified name.
    pub fn new(name: String) -> Self {
        Self {
            batches: Vec::new(),
            schema: Vec::new(),
            n_rows: 0,
            name,
        }
    }

    /// Builds from a collection of Table batches.
    ///
    /// Panics if column count or field metadata are inconsistent.
    pub fn from_batches(batches: Vec<Arc<Table>>, name_override: Option<String>) -> Self {
        if batches.is_empty() {
            return Self::new("Unnamed".into());
        }

        let name = name_override.unwrap_or_else(|| batches[0].name.clone());
        let schema: Vec<Arc<Field>> = batches[0].cols.iter().map(|fa| fa.field.clone()).collect();
        let n_cols = schema.len();
        let mut total_rows = 0usize;

        // Validate all batches.
        for (b_idx, batch) in batches.iter().enumerate() {
            assert_eq!(
                batch.n_cols(),
                n_cols,
                "Batch {b_idx} column-count mismatch"
            );
            for col_idx in 0..n_cols {
                let field = &schema[col_idx];
                let fa = &batch.cols[col_idx];
                assert_eq!(
                    &fa.field, field,
                    "Batch {b_idx} col {col_idx} schema mismatch"
                );
            }
            total_rows += batch.n_rows;
        }

        Self {
            batches,
            schema,
            n_rows: total_rows,
            name,
        }
    }

    /// Append a new Table batch.
    ///
    /// Panics on schema mismatch.
    pub fn push(&mut self, batch: Arc<Table>) {
        if self.batches.is_empty() {
            self.schema = batch.cols.iter().map(|fa| fa.field.clone()).collect();
        }
        let n_cols = self.schema.len();
        assert_eq!(batch.n_cols(), n_cols, "Pushed batch column-count mismatch");
        for col_idx in 0..n_cols {
            let field = &self.schema[col_idx];
            let fa = &batch.cols[col_idx];
            assert_eq!(
                &fa.field, field,
                "Pushed batch col {col_idx} schema mismatch"
            );
        }
        self.n_rows += batch.n_rows;
        self.batches.push(batch);
    }

    /// Inserts rows from another SuperTable (or Table) at the specified index.
    ///
    /// This is an **O(n)** operation where n is the number of rows in the batch
    /// containing the insertion point.
    ///
    /// # Arguments
    /// * `index` - Global row position before which to insert (0 = prepend, n_rows = append)
    /// * `other` - SuperTable or Table to insert (via `Into<SuperTable>`)
    ///
    /// # Requirements
    /// - Schema (column names, types, nullability) must match
    /// - `index` must be <= `self.n_rows`
    ///
    /// # Strategy
    /// Finds the batch containing the insertion point, splits it at that position, then
    /// inserts other's batches in between the split halves. This redistributes rows across
    /// batches while preserving chunked structure.
    ///
    /// # Errors
    /// - `IndexError` if index > n_rows
    /// - Schema mismatch if field metadata doesn't match
    pub fn insert_rows(
        &mut self,
        index: usize,
        other: impl Into<SuperTable>,
    ) -> Result<(), MinarrowError> {
        let other = other.into();

        // Validate index
        if index > self.n_rows {
            return Err(MinarrowError::IndexError(format!(
                "Index {} out of bounds for SuperTable with {} rows",
                index, self.n_rows
            )));
        }

        // If other is empty, nothing to do
        if other.n_rows == 0 {
            return Ok(());
        }

        // Validate schema match
        if !self.batches.is_empty() {
            if self.schema.len() != other.schema.len() {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "SuperTable",
                    to: "SuperTable",
                    message: Some(format!(
                        "Column count mismatch: {} vs {}",
                        self.schema.len(),
                        other.schema.len()
                    )),
                });
            }

            for (col_idx, (self_field, other_field)) in
                self.schema.iter().zip(other.schema.iter()).enumerate()
            {
                if self_field.name != other_field.name {
                    return Err(MinarrowError::IncompatibleTypeError {
                        from: "SuperTable",
                        to: "SuperTable",
                        message: Some(format!(
                            "Column {} name mismatch: '{}' vs '{}'",
                            col_idx, self_field.name, other_field.name
                        )),
                    });
                }

                if self_field.dtype != other_field.dtype {
                    return Err(MinarrowError::IncompatibleTypeError {
                        from: "SuperTable",
                        to: "SuperTable",
                        message: Some(format!(
                            "Column '{}' type mismatch: {:?} vs {:?}",
                            self_field.name, self_field.dtype, other_field.dtype
                        )),
                    });
                }

                if self_field.nullable != other_field.nullable {
                    return Err(MinarrowError::IncompatibleTypeError {
                        from: "SuperTable",
                        to: "SuperTable",
                        message: Some(format!(
                            "Column '{}' nullable mismatch: {} vs {}",
                            self_field.name, self_field.nullable, other_field.nullable
                        )),
                    });
                }
            }
        }

        // Handle empty self - just append other's batches
        if self.batches.is_empty() {
            self.batches = other.batches;
            self.schema = other.schema;
            self.n_rows = other.n_rows;
            return Ok(());
        }

        // Find which batch contains the insertion index
        let mut cumulative = 0;
        let mut target_idx = 0;
        let mut local_index = index;

        for (idx, batch) in self.batches.iter().enumerate() {
            let batch_rows = batch.n_rows;

            if index <= cumulative + batch_rows {
                target_idx = idx;
                local_index = index - cumulative;
                break;
            }

            cumulative += batch_rows;
        }

        let target_batch_rows = self.batches[target_idx].n_rows;

        // Handle edge cases: prepend or append to a batch without splitting
        if local_index == 0 {
            // Insert before target batch
            let mut new_batches = Vec::with_capacity(self.batches.len() + other.batches.len());
            new_batches.extend(self.batches.drain(0..target_idx));
            new_batches.extend(other.batches.into_iter());
            new_batches.extend(self.batches.drain(..));
            self.batches = new_batches;
            self.n_rows += other.n_rows;
        } else if local_index == target_batch_rows {
            // Insert after target batch
            let mut new_batches = Vec::with_capacity(self.batches.len() + other.batches.len());
            new_batches.extend(self.batches.drain(0..=target_idx));
            new_batches.extend(other.batches.into_iter());
            new_batches.extend(self.batches.drain(..));
            self.batches = new_batches;
            self.n_rows += other.n_rows;
        } else {
            // Split the target batch at the insertion point
            let target_batch = self.batches.remove(target_idx);
            let target_table = Arc::try_unwrap(target_batch).unwrap_or_else(|arc| (*arc).clone());
            let mut split_batches = target_table.split(local_index)?;

            // Build new batch list: batches before target + left batch + other's batches + right batch + remaining batches
            let mut new_batches = Vec::with_capacity(self.batches.len() + other.batches.len() + 2);
            new_batches.extend(self.batches.drain(0..target_idx));
            new_batches.extend(split_batches.batches.drain(0..1));
            new_batches.extend(other.batches.into_iter());
            new_batches.extend(split_batches.batches.drain(..));
            new_batches.extend(self.batches.drain(..));

            self.batches = new_batches;
            self.n_rows += other.n_rows;
        }

        Ok(())
    }

    /// Materialise a single `Table` containing all rows (concatenated).
    ///
    /// Uses arc data clones
    pub fn to_table(&self, name: Option<&str>) -> Table {
        assert!(!self.batches.is_empty(), "to_table() on empty BatchedTable");
        let n_cols = self.schema.len();
        let mut unified_cols = Vec::with_capacity(n_cols);

        for col_idx in 0..n_cols {
            let field = self.schema[col_idx].clone();
            // Use first array as base
            let mut arr = self.batches[0].cols[col_idx].array.clone();
            for batch in self.batches.iter().skip(1) {
                arr.concat_array(&batch.cols[col_idx].array);
            }
            let null_count = arr.null_count();
            unified_cols.push(FieldArray {
                field,
                array: arr.clone(),
                null_count,
            });
        }

        Table {
            cols: unified_cols,
            n_rows: self.n_rows,
            name: name
                .map(str::to_owned)
                .unwrap_or_else(|| "unified_table".to_string()),
        }
    }

    // API

    #[inline]
    pub fn n_cols(&self) -> usize {
        self.schema.len()
    }

    // TODO: Add test, confirm null case

    /// Returns the columns of the Super Table
    ///
    /// Holds an assumption that all inner tables have the same fields
    #[inline]
    pub fn cols(&self) -> Vec<Arc<Field>> {
        self.batches[0]
            .cols()
            .iter()
            .map(|x| x.field.clone())
            .collect()
    }

    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    #[inline]
    pub fn n_batches(&self) -> usize {
        self.batches.len()
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.n_rows
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_rows == 0
    }
    #[inline]
    pub fn schema(&self) -> &[Arc<Field>] {
        &self.schema
    }
    #[inline]
    pub fn batches(&self) -> &[Arc<Table>] {
        &self.batches
    }
    #[inline]
    pub fn batch(&self, idx: usize) -> Option<&Arc<Table>> {
        self.batches.get(idx)
    }

    // Return a new BatchedTable over a sub-range of rows.
    #[cfg(feature = "views")]
    pub fn view(&self, offset: usize, len: usize) -> SuperTableV {
        assert!(offset + len <= self.n_rows, "slice out of bounds");
        let mut slices = Vec::<TableV>::new();
        let mut remaining = len;
        let mut global_row = offset;

        for batch in &self.batches {
            if global_row >= batch.n_rows {
                global_row -= batch.n_rows;
                continue;
            }
            let take = (batch.n_rows - global_row).min(remaining);
            slices.push(TableV::from_arc_table(batch.clone(), global_row, take));
            global_row = 0;
            remaining -= take;
            if remaining == 0 {
                break;
            }
        }
        SuperTableV { slices, len }
    }

    #[cfg(feature = "views")]
    pub fn from_views(slices: &[TableV], name: String) -> Self {
        assert!(!slices.is_empty(), "from_slices: no slices provided");
        let n_cols = slices[0].n_cols();
        let mut batches = Vec::with_capacity(slices.len());
        let mut total_rows = 0usize;
        for slice in slices {
            let table = slice.to_table();
            assert_eq!(table.n_cols(), n_cols, "Batch column-count mismatch");
            total_rows += table.n_rows;
            batches.push(table.into());
        }
        let schema = slices[0].fields.iter().cloned().collect();
        Self {
            batches,
            schema,
            n_rows: total_rows,
            name,
        }
    }

    /// Rechunks the table according to the specified strategy.
    ///
    /// Redistributes rows across batches using an efficient incremental approach
    /// that avoids full materialization:
    /// - `Count(n)`: Creates batches of `n` rows (last batch may be smaller)
    /// - `Auto`: Uses a default size of 8192 rows
    /// - `Memory(bytes)`: Targets a specific memory size per batch
    ///
    /// # Arguments
    /// * `strategy` - The rechunking strategy to use
    ///
    /// # Errors
    /// - Returns `IndexError` if `Count(0)` is specified
    /// - Returns `IndexError` if memory-based calculation results in 0 chunk size
    ///
    /// # Example
    /// ```ignore
    /// // Rechunk into 1024-row batches
    /// table.rechunk(RechunkStrategy::Count(1024))?;
    ///
    /// // Rechunk with default size
    /// table.rechunk(RechunkStrategy::Auto)?;
    ///
    /// // Target 64KB per batch
    /// table.rechunk(RechunkStrategy::Memory(65536))?;
    /// ```
    pub fn rechunk(&mut self, strategy: RechunkStrategy) -> Result<(), MinarrowError> {
        if self.batches.is_empty() || self.n_rows == 0 {
            return Ok(());
        }

        // Determine chunk size based on strategy
        let chunk_size = match strategy {
            RechunkStrategy::Count(size) => {
                if size == 0 {
                    return Err(MinarrowError::IndexError(
                        "Count chunk size must be greater than 0".to_string(),
                    ));
                }
                size
            }
            RechunkStrategy::Auto => 8192,
            #[cfg(feature = "size")]
            RechunkStrategy::Memory(bytes_per_chunk) => {
                let total_bytes = self.est_bytes();
                let total_rows = self.n_rows;

                if total_bytes == 0 {
                    return Err(MinarrowError::IndexError(
                        "Cannot rechunk: table has 0 estimated bytes".to_string(),
                    ));
                }

                ((bytes_per_chunk * total_rows) / total_bytes).max(1)
            }
        };

        // Fast path: single batch already at target size
        if self.batches.len() == 1 && self.batches[0].n_rows == chunk_size {
            return Ok(());
        }

        let mut new_batches = Vec::new();
        let mut accumulator: Option<Table> = None;

        // Process each existing batch
        for batch_arc in self.batches.drain(..) {
            let batch = Arc::try_unwrap(batch_arc).unwrap_or_else(|arc| (*arc).clone());
            let mut remaining = batch;

            while remaining.n_rows > 0 {
                if let Some(ref mut acc) = accumulator {
                    let acc_rows = acc.n_rows;
                    let needed = chunk_size - acc_rows;

                    if remaining.n_rows <= needed {
                        // Entire remaining batch fits in accumulator
                        *acc = acc.clone().concat(remaining)?;

                        // If accumulator is now full, emit it
                        if acc.n_rows == chunk_size {
                            new_batches.push(Arc::new(accumulator.take().unwrap()));
                        }
                        break; // consumed remaining
                    } else {
                        // Split remaining batch to complete accumulator
                        let split_result = remaining.split(needed)?;
                        let mut parts = split_result.batches;
                        let to_add =
                            Arc::try_unwrap(parts.remove(0)).unwrap_or_else(|arc| (*arc).clone());
                        remaining =
                            Arc::try_unwrap(parts.remove(0)).unwrap_or_else(|arc| (*arc).clone());

                        // Complete and emit the accumulator
                        *acc = acc.clone().concat(to_add)?;
                        new_batches.push(Arc::new(accumulator.take().unwrap()));
                    }
                } else {
                    // No accumulator - start processing remaining
                    if remaining.n_rows == chunk_size {
                        // Exact fit - use remaining as-is
                        new_batches.push(Arc::new(remaining));
                        break;
                    } else if remaining.n_rows > chunk_size {
                        // Split off one chunk_size portion
                        let split_result = remaining.split(chunk_size)?;
                        let mut parts = split_result.batches;
                        new_batches.push(parts.remove(0));
                        remaining =
                            Arc::try_unwrap(parts.remove(0)).unwrap_or_else(|arc| (*arc).clone());
                    } else {
                        // Remaining becomes new accumulator
                        accumulator = Some(remaining);
                        break;
                    }
                }
            }
        }

        // Emit any remaining accumulator as final batch
        if let Some(final_batch) = accumulator {
            new_batches.push(Arc::new(final_batch));
        }

        self.batches = new_batches;
        Ok(())
    }

    /// Rechunks only the first `up_to_row` rows, leaving the rest untouched.
    ///
    /// This is useful for streaming scenarios where new data is being appended
    /// and you want to rechunk stable data while leaving recent additions alone.
    ///
    /// # Arguments
    /// * `up_to_row` - Rechunk only rows before this index
    /// * `strategy` - The rechunking strategy to use
    ///
    /// # Errors
    /// - Returns `IndexError` if `up_to_row` is greater than total row count
    /// - Returns same errors as `rechunk()` for invalid strategies
    ///
    /// # Example
    /// ```ignore
    /// // Rechunk first 1000 rows, leave the rest untouched
    /// table.rechunk_to(1000, RechunkStrategy::Count(512))?;
    /// ```
    pub fn rechunk_to(
        &mut self,
        up_to_row: usize,
        strategy: RechunkStrategy,
    ) -> Result<(), MinarrowError> {
        let total_rows = self.n_rows;

        if up_to_row > total_rows {
            return Err(MinarrowError::IndexError(format!(
                "rechunk_to row {} out of bounds for table with {} rows",
                up_to_row, total_rows
            )));
        }

        if up_to_row == 0 || self.batches.is_empty() {
            return Ok(());
        }

        if up_to_row == total_rows {
            // Rechunk everything
            return self.rechunk(strategy);
        }

        // Find which batches contain the data up to up_to_row
        let mut current_offset = 0;
        let mut split_point = 0;

        for (i, batch) in self.batches.iter().enumerate() {
            let batch_end = current_offset + batch.n_rows;
            if batch_end > up_to_row {
                split_point = i;
                break;
            }
            current_offset = batch_end;
        }

        // Extract batches to rechunk and batches to keep
        let mut to_rechunk = self.batches.drain(..=split_point).collect::<Vec<_>>();
        let keep_batches = self.batches.drain(..).collect::<Vec<_>>();

        // If the split batch needs to be divided
        if current_offset < up_to_row {
            let split_batch_arc = to_rechunk.pop().unwrap();
            let split_batch = Arc::try_unwrap(split_batch_arc).unwrap_or_else(|arc| (*arc).clone());
            let split_at = up_to_row - current_offset;

            let split_result = split_batch.split(split_at)?;
            let mut parts = split_result.batches;
            to_rechunk.push(parts.remove(0));
            self.batches.push(parts.remove(0));
        }

        // Rechunk the selected portion
        self.batches.extend(keep_batches);
        // from_batches infers schema from the batches, second param is name
        let mut temp = SuperTable::from_batches(to_rechunk.into(), Some(self.name.clone()));
        temp.rechunk(strategy)?;

        // Reconstruct rechunked portion + untouched portion
        let mut result = temp.batches;
        result.extend(self.batches.drain(..));
        self.batches = result;

        // Recalculate n_rows
        self.n_rows = self.batches.iter().map(|b| b.n_rows).sum();

        Ok(())
    }
}

impl Default for SuperTable {
    fn default() -> Self {
        Self::new("Unnamed".into())
    }
}

impl FromIterator<Table> for SuperTable {
    fn from_iter<T: IntoIterator<Item = Table>>(iter: T) -> Self {
        let batches: Vec<Arc<Table>> = iter.into_iter().map(|x| x.into()).collect();
        SuperTable::from_batches(batches, None)
    }
}

impl Shape for SuperTable {
    fn shape(&self) -> ShapeDim {
        ShapeDim::Rank2 {
            rows: self.n_rows(),
            cols: self.n_cols(),
        }
    }
}

impl Concatenate for SuperTable {
    /// Concatenates two SuperTables by appending all batches from `other` to `self`.
    ///
    /// # Requirements
    /// - Both SuperTables must have the same schema (column names and types)
    ///
    /// # Returns
    /// A new SuperTable containing all batches from `self` followed by all batches from `other`
    ///
    /// # Errors
    /// - `IncompatibleTypeError` if schemas don't match
    fn concat(self, other: Self) -> Result<Self, MinarrowError> {
        // If both are empty, return empty
        if self.batches.is_empty() && other.batches.is_empty() {
            return Ok(SuperTable::new(format!("{}+{}", self.name, other.name)));
        }

        // If one is empty, return the other
        if self.batches.is_empty() {
            let mut result = other;
            result.name = format!("{}+{}", self.name, result.name);
            return Ok(result);
        }
        if other.batches.is_empty() {
            let mut result = self;
            result.name = format!("{}+{}", result.name, other.name);
            return Ok(result);
        }

        // Validate schemas match
        if self.schema.len() != other.schema.len() {
            return Err(MinarrowError::IncompatibleTypeError {
                from: "SuperTable",
                to: "SuperTable",
                message: Some(format!(
                    "Cannot concatenate SuperTables with different column counts: {} vs {}",
                    self.schema.len(),
                    other.schema.len()
                )),
            });
        }

        // Check schema compatibility field by field
        for (col_idx, (self_field, other_field)) in
            self.schema.iter().zip(other.schema.iter()).enumerate()
        {
            if self_field.name != other_field.name {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "SuperTable",
                    to: "SuperTable",
                    message: Some(format!(
                        "Column {} name mismatch: '{}' vs '{}'",
                        col_idx, self_field.name, other_field.name
                    )),
                });
            }

            if self_field.dtype != other_field.dtype {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "SuperTable",
                    to: "SuperTable",
                    message: Some(format!(
                        "Column '{}' type mismatch: {:?} vs {:?}",
                        self_field.name, self_field.dtype, other_field.dtype
                    )),
                });
            }

            if self_field.nullable != other_field.nullable {
                return Err(MinarrowError::IncompatibleTypeError {
                    from: "SuperTable",
                    to: "SuperTable",
                    message: Some(format!(
                        "Column '{}' nullable mismatch: {} vs {}",
                        self_field.name, self_field.nullable, other_field.nullable
                    )),
                });
            }
        }

        // Concatenate batches
        let mut result_batches = self.batches;
        result_batches.extend(other.batches);
        let total_rows = self.n_rows + other.n_rows;

        Ok(SuperTable {
            batches: result_batches,
            schema: self.schema,
            n_rows: total_rows,
            name: format!("{}+{}", self.name, other.name),
        })
    }
}

impl Display for SuperTable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "SuperTable \"{}\" [{} rows, {} columns, {} batches]",
            self.name,
            self.n_rows,
            self.schema.len(),
            self.batches.len()
        )?;

        for (batch_idx, batch) in self.batches.iter().enumerate() {
            writeln!(
                f,
                "  ├─ Batch {batch_idx}: {} rows, {} columns",
                batch.n_rows,
                batch.n_cols()
            )?;
            for (col_idx, col) in batch.cols.iter().enumerate() {
                let indent = "    │ ";
                writeln!(
                    f,
                    "{indent}Col {col_idx}: \"{}\" (dtype: {}, nulls: {})",
                    col.field.name, col.field.dtype, col.null_count
                )?;
                for line in format!("{}", col.array).lines() {
                    writeln!(f, "{indent}  {line}")?;
                }
            }
        }

        Ok(())
    }
}

#[cfg(feature = "views")]
impl From<SuperTableV> for SuperTable {
    fn from(super_table_v: SuperTableV) -> Self {
        if super_table_v.is_empty() {
            return SuperTable::new("".to_string());
        }
        SuperTable::from_views(&super_table_v.slices, "SuperTable".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;
    use crate::{Array, Field, FieldArray, NumericArray, Table};

    fn fa(name: &str, vals: &[i32]) -> FieldArray {
        let arr = Array::from_int32(crate::IntegerArray::<i32>::from_slice(vals));
        let field = Field::new(name.to_string(), ArrowType::Int32, false, None);
        FieldArray::new(field, arr)
    }

    fn table(cols: Vec<FieldArray>) -> Table {
        let n_rows = cols[0].len();
        for c in &cols {
            assert_eq!(c.len(), n_rows, "all columns must have same len for Table");
        }
        Table {
            cols,
            n_rows,
            name: "batch".to_string(),
        }
    }

    #[test]
    fn test_empty_and_default() {
        let t = SuperTable::default();
        assert!(t.is_empty());
        assert_eq!(t.n_cols(), 0);
        assert_eq!(t.n_batches(), 0);
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn test_from_batches_basic() {
        let col1 = fa("a", &[1, 2, 3]);
        let col2 = fa("b", &[10, 11, 12]);
        let col3 = fa("a", &[4, 5]);
        let col4 = fa("b", &[13, 14]);
        let batch1 = Arc::new(table(vec![col1.clone(), col2.clone()]));
        let batch2 = Arc::new(table(vec![col3.clone(), col4.clone()]));
        let batches = vec![batch1, batch2].into();

        let t = SuperTable::from_batches(batches, None);
        assert_eq!(t.n_cols(), 2);
        assert_eq!(t.n_batches(), 2);
        assert_eq!(t.len(), 5);
        assert_eq!(t.schema()[0].name, "a");
        assert_eq!(t.schema()[1].name, "b");
        assert_eq!(t.batches()[0].cols[0], col1);
        assert_eq!(t.batches()[1].cols[1], col4);
    }

    #[test]
    #[should_panic(expected = "column-count mismatch")]
    fn test_from_batches_col_count_mismatch() {
        let batch1 = Arc::new(table(vec![fa("a", &[1, 2])]));
        let batch2 = Arc::new(table(vec![fa("a", &[3, 4]), fa("b", &[5, 6])]));
        SuperTable::from_batches(vec![batch1, batch2].into(), None);
    }

    #[test]
    #[should_panic(expected = "schema mismatch")]
    fn test_from_batches_schema_mismatch() {
        let batch1 = Arc::new(table(vec![fa("a", &[1, 2])]));
        let mut wrong = fa("z", &[3, 4]);
        let mut mismatched_field = (*wrong.field).clone();
        mismatched_field.dtype = ArrowType::Int32;
        wrong.field = Arc::new(mismatched_field);
        let batch2 = Arc::new(table(vec![wrong]));
        SuperTable::from_batches(vec![batch1, batch2].into(), None);
    }

    #[test]
    fn test_push_and_to_table() {
        let mut t = SuperTable::default();
        t.push(Arc::new(table(vec![fa("x", &[1, 2]), fa("y", &[3, 4])])));
        t.push(Arc::new(table(vec![fa("x", &[5]), fa("y", &[6])])));
        assert_eq!(t.n_cols(), 2);
        assert_eq!(t.n_batches(), 2);
        assert_eq!(t.len(), 3);
        let tab = t.to_table(Some("joined"));
        assert_eq!(tab.name, "joined");
        assert_eq!(tab.n_rows, 3);
        assert_eq!(tab.cols[0].field.name, "x");
        assert_eq!(tab.cols[1].field.name, "y");
    }

    #[test]
    #[should_panic(expected = "column-count mismatch")]
    fn test_push_col_count_mismatch() {
        let mut t = SuperTable::default();
        t.push(Arc::new(table(vec![fa("a", &[1, 2])])));
        t.push(Arc::new(table(vec![fa("a", &[3, 4]), fa("b", &[5, 6])])));
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_slice_and_owned_table() {
        let batch1 = Arc::new(table(vec![fa("q", &[1, 2, 3]), fa("w", &[4, 5, 6])]));
        let batch2 = Arc::new(table(vec![fa("q", &[7, 8]), fa("w", &[9, 10])]));
        let t = SuperTable::from_batches(vec![batch1, batch2].into(), None);

        // Slice rows 2..5 (3 rows), crossing the batch boundary
        let slice = t.view(2, 3);
        assert_eq!(slice.len, 3);
        assert_eq!(slice.slices.len(), 2);

        let owned = slice.to_table(Some("part"));
        assert_eq!(owned.name, "part");
        assert_eq!(owned.n_rows, 3);
        assert_eq!(owned.cols[0].field.name, "q");
        assert_eq!(owned.cols[1].field.name, "w");

        let arr = &owned.cols[0].array;
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
            assert_eq!(ints.data.as_slice(), &[3, 7, 8]);
        } else {
            panic!("expected Int32 array");
        }

        let arr = &owned.cols[1].array;
        if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
            assert_eq!(ints.data.as_slice(), &[6, 9, 10]);
        } else {
            panic!("expected Int32 array");
        }
    }

    #[test]
    fn test_schema_and_batch_access() {
        let t = SuperTable::from_batches(vec![Arc::new(table(vec![fa("alpha", &[1, 2])]))], None);
        assert_eq!(t.n_cols(), 1);
        assert_eq!(t.schema()[0].name, "alpha");
        assert!(t.batch(0).is_some());
        assert!(t.batch(5).is_none());
        assert_eq!(t.batches().len(), 1);
    }

    #[cfg(feature = "views")]
    #[test]
    fn test_from_slices() {
        let batch1 = Arc::new(table(vec![fa("x", &[1, 2]), fa("y", &[3, 4])]));
        let batch2 = Arc::new(table(vec![fa("x", &[5, 6]), fa("y", &[7, 8])]));
        let t = SuperTable::from_batches(vec![batch1.clone(), batch2.clone()], None);

        // Break into 4 slices of 1 row each
        let mut table_slices = Vec::new();
        for i in 0..t.len() {
            let bts = t.view(i, 1);
            for ts in bts.slices.clone() {
                table_slices.push(ts);
            }
        }

        // Reconstruct a new batched table
        let rebuilt = SuperTable::from_views(&table_slices, "rebuilt".to_string());

        assert_eq!(rebuilt.n_cols(), t.n_cols());
        assert_eq!(rebuilt.len(), t.len());

        // Schema should match
        for (left, right) in rebuilt.schema().iter().zip(t.schema()) {
            assert_eq!(left.name, right.name);
            assert_eq!(left.dtype, right.dtype);
        }

        // Validate data for each column
        let expected_x = [1, 2, 5, 6];
        let expected_y = [3, 4, 7, 8];
        for (col_idx, expected) in [expected_x.as_slice(), expected_y.as_slice()]
            .iter()
            .enumerate()
        {
            let arr = rebuilt.to_table(None).cols[col_idx].array.clone();
            if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
                assert_eq!(ints.data.as_slice(), *expected);
            } else {
                panic!("unexpected array type at col {col_idx}");
            }
        }
    }

    #[test]
    fn test_insert_rows_into_first_batch() {
        let batch1 = Arc::new(table(vec![fa("a", &[1, 2, 3]), fa("b", &[10, 20, 30])]));
        let batch2 = Arc::new(table(vec![fa("a", &[4, 5]), fa("b", &[40, 50])]));
        let mut st = SuperTable::from_batches(vec![batch1, batch2], None);

        let insert_batch = Arc::new(table(vec![fa("a", &[99]), fa("b", &[88])]));
        let insert_st = SuperTable::from_batches(vec![insert_batch], None);

        st.insert_rows(1, insert_st).unwrap();

        assert_eq!(st.n_rows(), 6);
        assert_eq!(st.n_batches(), 4);

        let materialised = st.to_table(None);
        match &materialised.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 99, 2, 3, 4, 5]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_insert_rows_into_second_batch() {
        let batch1 = Arc::new(table(vec![fa("a", &[1, 2])]));
        let batch2 = Arc::new(table(vec![fa("a", &[3, 4, 5])]));
        let mut st = SuperTable::from_batches(vec![batch1, batch2], None);

        let insert_batch = Arc::new(table(vec![fa("a", &[99, 88])]));
        let insert_st = SuperTable::from_batches(vec![insert_batch], None);

        st.insert_rows(3, insert_st).unwrap();

        assert_eq!(st.n_rows(), 7);

        let materialised = st.to_table(None);
        match &materialised.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 2, 3, 99, 88, 4, 5]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_insert_rows_prepend() {
        let batch1 = Arc::new(table(vec![fa("a", &[1, 2, 3])]));
        let mut st = SuperTable::from_batches(vec![batch1], None);

        let insert_batch = Arc::new(table(vec![fa("a", &[99])]));
        let insert_st = SuperTable::from_batches(vec![insert_batch], None);

        st.insert_rows(0, insert_st).unwrap();

        assert_eq!(st.n_rows(), 4);

        let materialised = st.to_table(None);
        match &materialised.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[99, 1, 2, 3]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_insert_rows_append() {
        let batch1 = Arc::new(table(vec![fa("a", &[1, 2])]));
        let batch2 = Arc::new(table(vec![fa("a", &[3, 4])]));
        let mut st = SuperTable::from_batches(vec![batch1, batch2], None);

        let insert_batch = Arc::new(table(vec![fa("a", &[99])]));
        let insert_st = SuperTable::from_batches(vec![insert_batch], None);

        st.insert_rows(4, insert_st).unwrap();

        assert_eq!(st.n_rows(), 5);

        let materialised = st.to_table(None);
        match &materialised.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 2, 3, 4, 99]);
            }
            _ => panic!("wrong type"),
        }
    }

    #[test]
    fn test_insert_rows_schema_mismatch() {
        let batch1 = Arc::new(table(vec![fa("a", &[1, 2])]));
        let mut st = SuperTable::from_batches(vec![batch1], None);

        let wrong_batch = Arc::new(table(vec![fa("b", &[99])]));
        let wrong_st = SuperTable::from_batches(vec![wrong_batch], None);

        let result = st.insert_rows(0, wrong_st);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_rows_out_of_bounds() {
        let batch1 = Arc::new(table(vec![fa("a", &[1, 2])]));
        let mut st = SuperTable::from_batches(vec![batch1], None);

        let insert_batch = Arc::new(table(vec![fa("a", &[99])]));
        let insert_st = SuperTable::from_batches(vec![insert_batch], None);

        let result = st.insert_rows(10, insert_st);
        assert!(result.is_err());
    }

    #[test]
    fn test_rechunk_uniform() {
        // Create a SuperTable with 3 batches of varying sizes
        let batch1 = Arc::new(table(vec![fa("x", &[1, 2, 3]), fa("y", &[10, 20, 30])]));
        let batch2 = Arc::new(table(vec![
            fa("x", &[4, 5, 6, 7]),
            fa("y", &[40, 50, 60, 70]),
        ]));
        let batch3 = Arc::new(table(vec![fa("x", &[8, 9]), fa("y", &[80, 90])]));
        let mut st = SuperTable::from_batches(vec![batch1, batch2, batch3], None);

        // Total rows: 3 + 4 + 2 = 9 rows
        assert_eq!(st.n_rows(), 9);
        assert_eq!(st.n_batches(), 3);

        // Rechunk to batches of 4 rows each
        st.rechunk(RechunkStrategy::Count(4)).unwrap();

        // Should have 3 batches: [4 rows, 4 rows, 1 row]
        assert_eq!(st.n_batches(), 3);
        assert_eq!(st.batch(0).unwrap().n_rows, 4);
        assert_eq!(st.batch(1).unwrap().n_rows, 4);
        assert_eq!(st.batch(2).unwrap().n_rows, 1);
        assert_eq!(st.n_rows(), 9);
    }

    #[test]
    fn test_rechunk_auto() {
        // Create a SuperTable with many rows spread across small batches
        let mut batches = Vec::new();
        for i in 0..100 {
            let vals: Vec<i32> = vec![i * 10, i * 10 + 1];
            batches.push(Arc::new(table(vec![fa("col", &vals)])));
        }
        let mut st = SuperTable::from_batches(batches.into(), None);

        // Total rows: 100 batches * 2 rows = 200 rows
        assert_eq!(st.n_rows(), 200);
        assert_eq!(st.n_batches(), 100);

        // Rechunk with Auto strategy (default 8192 rows per batch)
        st.rechunk(RechunkStrategy::Auto).unwrap();

        // Should consolidate to 1 batch since 200 < 8192
        assert_eq!(st.n_batches(), 1);
        assert_eq!(st.batch(0).unwrap().n_rows, 200);
        assert_eq!(st.n_rows(), 200);
    }

    #[test]
    #[cfg(feature = "size")]
    fn test_rechunk_by_memory() {
        // Create a SuperTable with i32 data
        let batch1 = Arc::new(table(vec![fa("a", &[1, 2, 3, 4]), fa("b", &[5, 6, 7, 8])]));
        let batch2 = Arc::new(table(vec![
            fa("a", &[9, 10, 11, 12]),
            fa("b", &[13, 14, 15, 16]),
        ]));
        let mut st = SuperTable::from_batches(vec![batch1, batch2], None);

        assert_eq!(st.n_rows(), 8);
        assert_eq!(st.n_batches(), 2);

        // Use a larger memory target to get predictable chunking
        // The actual byte size includes overhead beyond raw data
        st.rechunk(RechunkStrategy::Memory(64)).unwrap();

        // Should rechunk into batches
        assert!(st.n_batches() >= 1);
        assert_eq!(st.n_rows(), 8);

        // Verify data integrity after rechunking
        let materialized = st.to_table(None);
        assert_eq!(materialized.n_rows, 8);
    }

    #[test]
    fn test_rechunk_uniform_zero_error() {
        let batch1 = Arc::new(table(vec![fa("x", &[1, 2, 3])]));
        let mut st = SuperTable::from_batches(vec![batch1], None);

        let result = st.rechunk(RechunkStrategy::Count(0));
        assert!(result.is_err());
        if let Err(MinarrowError::IndexError(msg)) = result {
            assert!(msg.contains("Count chunk size must be greater than 0"));
        } else {
            panic!("Expected IndexError for zero chunk size");
        }
    }

    #[test]
    fn test_rechunk_empty_table() {
        let mut st = SuperTable::default();
        assert!(st.is_empty());

        // Rechunking an empty table should succeed and remain empty
        st.rechunk(RechunkStrategy::Auto).unwrap();
        assert!(st.is_empty());
        assert_eq!(st.n_batches(), 0);

        st.rechunk(RechunkStrategy::Count(10)).unwrap();
        assert!(st.is_empty());
        assert_eq!(st.n_batches(), 0);
    }

    #[test]
    fn test_rechunk_preserves_data_order() {
        // Create batches with sequential data
        let batch1 = Arc::new(table(vec![fa("num", &[1, 2, 3])]));
        let batch2 = Arc::new(table(vec![fa("num", &[4, 5, 6, 7])]));
        let batch3 = Arc::new(table(vec![fa("num", &[8, 9])]));
        let mut st = SuperTable::from_batches(vec![batch1, batch2, batch3], None);

        assert_eq!(st.n_rows(), 9);

        // Rechunk with different size
        st.rechunk(RechunkStrategy::Count(5)).unwrap();

        // Materialize to verify order is preserved
        let materialized = st.to_table(None);
        match &materialized.cols[0].array {
            Array::NumericArray(NumericArray::Int32(arr)) => {
                assert_eq!(arr.data.as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
            }
            _ => panic!("Expected Int32 array"),
        }
    }
}
