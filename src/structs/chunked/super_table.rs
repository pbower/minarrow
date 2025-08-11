use std::fmt::{Display, Formatter};
use std::iter::FromIterator;
use std::sync::Arc;

use crate::structs::field::Field;
use crate::structs::field_array::FieldArray;
use crate::structs::table::Table;
#[cfg(feature = "views")]
use crate::{SuperTableV, TableV};

/// Batched (windowed/chunked) table - collection of `Tables`.
///
/// ### Data structure
/// Each Table represents a record batch with schema consistency enforced.
/// Windows/row-chunks are tracked as Vec<Table>.
///
/// - `batches`: Ordered record batches.
/// - `schema`: schema of the first batch, cached for fast access.
/// - `n_rows`: total number of rows across all batches.
/// - `name`: logical group name.
///
/// ### Use cases
/// Useful for cases such as :
///     1. Streaming *(and mini-batch processing)*
///     2. Reading from multiple memory mapped arrow files from disk
///     3. In-memory analytics, where chunks can be used as a source for
///     parallelism, or windowing, etc., depending on use case semantics.
///
/// # Naming Rationale
/// *Apache Arrow*'s "Table" represents a single logical dataset with chunked columns internally.
/// `SuperTable` explicitly indicates this is a collection of separate record batches (Tables),
/// each maintaining independent column buffers whilst sharing schema consistency.
/// The "Super" prefix distinguishes it as a higher-order container rather than a single flat table,
/// and the fact that it's epic. A `ChunkedTable` alias is available for alternative preference.
#[derive(Clone, Debug, PartialEq)]
pub struct SuperTable {
    pub batches: Vec<Arc<Table>>,
    pub schema: Vec<Arc<Field>>,
    pub n_rows: usize,
    pub name: String
}

impl SuperTable {
    /// Creates a new empty BatchedTable with a specified name.
    pub fn new(name: String) -> Self {
        Self {
            batches: Vec::new(),
            schema: Vec::new(),
            n_rows: 0,
            name
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
            assert_eq!(batch.n_cols(), n_cols, "Batch {b_idx} column-count mismatch");
            for col_idx in 0..n_cols {
                let field = &schema[col_idx];
                let fa = &batch.cols[col_idx];
                assert_eq!(&fa.field, field, "Batch {b_idx} col {col_idx} schema mismatch");
            }
            total_rows += batch.n_rows;
        }

        Self {
            batches,
            schema,
            n_rows: total_rows,
            name
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
            assert_eq!(&fa.field, field, "Pushed batch col {col_idx} schema mismatch");
        }
        self.n_rows += batch.n_rows;
        self.batches.push(batch);
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
            unified_cols.push(FieldArray { field, array: arr.clone(), null_count });
        }

        Table {
            cols: unified_cols,
            n_rows: self.n_rows,
            name: name.map(str::to_owned).unwrap_or_else(|| "unified_table".to_string())
        }
    }

    // API

    #[inline]
    pub fn n_cols(&self) -> usize {
        self.schema.len()
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
            name
        }
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
        Table { cols, n_rows, name: "batch".to_string() }
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
        for (col_idx, expected) in [expected_x.as_slice(), expected_y.as_slice()].iter().enumerate()
        {
            let arr = rebuilt.to_table(None).cols[col_idx].array.clone();
            if let Array::NumericArray(NumericArray::Int32(ints)) = arr {
                assert_eq!(ints.data.as_slice(), *expected);
            } else {
                panic!("unexpected array type at col {col_idx}");
            }
        }
    }
}
