//! Integration tests for pandas-style selection on Table and TableV

#![cfg(feature = "select")]

use minarrow::{Array, FieldArray, IntegerArray, MaskedArray, StringArray, Table};
use minarrow::traits::selection::{FieldSelection, DataSelection};

#[test]
fn test_table_column_selection_by_names() {
    let table = create_test_table();

    // Select columns by names
    let view = table.c(&["name", "value"]);

    assert_eq!(view.n_cols(), 2);
    assert_eq!(view.col_names(), vec!["name", "value"]);
    assert_eq!(view.n_rows(), 5);
}

#[test]
fn test_table_column_selection_by_indices() {
    let table = create_test_table();

    // Select columns by indices
    let view = table.c(&[0, 2]);

    assert_eq!(view.n_cols(), 2);
    assert_eq!(view.n_rows(), 5);
}

#[test]
fn test_table_column_selection_by_range() {
    let table = create_test_table();

    // Select columns by range
    let view = table.c(0..2);

    assert_eq!(view.n_cols(), 2);
    assert_eq!(view.n_rows(), 5);
}

#[test]
fn test_table_row_selection_by_range() {
    let table = create_test_table();

    // Select rows by range
    let view = table.r(1..4);

    assert_eq!(view.n_rows(), 3);
    assert_eq!(view.n_cols(), 3);
}

#[test]
fn test_table_row_selection_by_indices() {
    let table = create_test_table();

    // Select specific rows
    let view = table.r(&[0, 2, 4]);

    assert_eq!(view.n_rows(), 3);
    assert_eq!(view.n_cols(), 3);
}

#[test]
fn test_chained_column_then_row_selection() {
    let table = create_test_table();

    // Select columns first, then rows
    let view = table.c(&["id", "value"]).r(1..4);

    assert_eq!(view.n_cols(), 2);
    assert_eq!(view.n_rows(), 3);

    // Verify column names are correct
    let names = view.col_names();
    assert_eq!(names, vec!["id", "value"]);
}

#[test]
fn test_chained_row_then_column_selection() {
    let table = create_test_table();

    // Select rows first, then columns
    let view = table.r(1..4).c(&["name"]);

    assert_eq!(view.n_rows(), 3);
    assert_eq!(view.n_cols(), 1);
    assert_eq!(view.col_names(), vec!["name"]);
}

#[test]
fn test_tablev_column_refinement() {
    let table = create_test_table();

    // Create initial view with all columns
    let view1 = table.c(&["id", "name", "value"]);
    assert_eq!(view1.n_cols(), 3);

    // Refine to fewer columns
    let view2 = view1.c(&["name", "value"]);
    assert_eq!(view2.n_cols(), 2);
    assert_eq!(view2.col_names(), vec!["name", "value"]);
}

#[test]
fn test_tablev_row_refinement() {
    let table = create_test_table();

    // Create initial view with rows 0..5
    let view1 = table.r(0..5);
    assert_eq!(view1.n_rows(), 5);

    // Refine to rows 1..3 (which maps to physical rows 1..3)
    let view2 = view1.r(&[1, 2]);
    assert_eq!(view2.n_rows(), 2);
}

#[test]
fn test_tablev_methods_respect_column_selection() {
    let table = create_test_table();
    let view = table.c(&["id", "value"]);

    // Test n_cols
    assert_eq!(view.n_cols(), 2);

    // Test col_names
    assert_eq!(view.col_names(), vec!["id", "value"]);

    // Test col by index (logical index within selection)
    let col0 = view.col(0);
    assert!(col0.is_some());

    let col1 = view.col(1);
    assert!(col1.is_some());

    // Logical index 2 doesn't exist (only 2 columns selected)
    let col2 = view.col(2);
    assert!(col2.is_none());

    // Test col_by_name
    assert!(view.col_by_name("id").is_some());
    assert!(view.col_by_name("value").is_some());
    assert!(view.col_by_name("name").is_none()); // Not in selection
}

#[test]
fn test_tablev_methods_respect_row_selection() {
    let table = create_test_table();
    let view = table.r(&[1, 3]);

    // Test n_rows
    assert_eq!(view.n_rows(), 2);
}

#[test]
fn test_to_table_respects_selections() {
    let table = create_test_table();

    // Create view with column and row selections
    let view = table.c(&["id", "value"]).r(1..4);

    // Convert back to table
    let materialized = view.to_table();

    assert_eq!(materialized.n_cols(), 2);
    assert_eq!(materialized.n_rows, 3);
}

// ===== Tests for new extensible API (.f(), .d(), etc.) =====

#[test]
fn test_field_selection_methods() {
    let table = create_test_table();

    // Test .f() method
    let view1 = table.f(&["name", "value"]);
    assert_eq!(view1.n_cols(), 2);
    assert_eq!(view1.col_names(), vec!["name", "value"]);

    // Test .fields() alias
    let view2 = table.fields(&["name", "value"]);
    assert_eq!(view2.n_cols(), 2);
    assert_eq!(view2.col_names(), vec!["name", "value"]);

    // Test .y() alias (spatial thinking)
    let view3 = table.y(&["id", "value"]);
    assert_eq!(view3.n_cols(), 2);
    assert_eq!(view3.col_names(), vec!["id", "value"]);
}

#[test]
fn test_data_selection_methods() {
    let table = create_test_table();

    // Test .d() method
    let view1 = table.d(1..4);
    assert_eq!(view1.n_rows(), 3);

    // Test .data() alias
    let view2 = table.data(1..4);
    assert_eq!(view2.n_rows(), 3);

    // Test .x() alias (spatial thinking)
    let view3 = table.x(&[0, 2, 4]);
    assert_eq!(view3.n_rows(), 3);
}

#[test]
fn test_mixed_api_chaining() {
    let table = create_test_table();

    // Chain .f() and .d()
    let view1 = table.f(&["id", "value"]).d(1..4);
    assert_eq!(view1.n_cols(), 2);
    assert_eq!(view1.n_rows(), 3);

    // Chain .c() (old) with .d() (new)
    let view2 = table.c(&["id", "value"]).d(1..4);
    assert_eq!(view2.n_cols(), 2);
    assert_eq!(view2.n_rows(), 3);

    // Chain .f() (new) with .r() (old)
    let view3 = table.f(&["id", "value"]).r(1..4);
    assert_eq!(view3.n_cols(), 2);
    assert_eq!(view3.n_rows(), 3);

    // Spatial aliases
    let view4 = table.y(&["name"]).x(0..3);
    assert_eq!(view4.n_cols(), 1);
    assert_eq!(view4.n_rows(), 3);
}

#[test]
fn test_tablev_new_api_refinement() {
    let table = create_test_table();

    // Create initial view with .f() and .d()
    let view1 = table.f(&["id", "name", "value"]).d(0..5);
    assert_eq!(view1.n_cols(), 3);
    assert_eq!(view1.n_rows(), 5);

    // Refine using new API
    let view2 = view1.f(&["name", "value"]).d(&[1, 2, 3]);
    assert_eq!(view2.n_cols(), 2);
    assert_eq!(view2.n_rows(), 3);
    assert_eq!(view2.col_names(), vec!["name", "value"]);
}

#[test]
fn test_api_compatibility() {
    let table = create_test_table();

    // Old API
    let view_old = table.c(&["id", "value"]).r(1..4);

    // New API (should produce identical results)
    let view_new = table.f(&["id", "value"]).d(1..4);

    assert_eq!(view_old.n_cols(), view_new.n_cols());
    assert_eq!(view_old.n_rows(), view_new.n_rows());
    assert_eq!(view_old.col_names(), view_new.col_names());
}

// Helper function to create a test table
fn create_test_table() -> Table {
    // Create id column (0, 1, 2, 3, 4)
    let mut id_arr = IntegerArray::<i32>::default();
    for i in 0..5 {
        id_arr.push(i);
    }

    // Create name column
    let mut name_arr = StringArray::<u32>::default();
    for i in 0..5 {
        name_arr.push(format!("name{}", i));
    }

    // Create value column
    let mut value_arr = IntegerArray::<i64>::default();
    for i in 0..5 {
        value_arr.push((i * 10) as i64);
    }

    let col_id = FieldArray::from_arr("id", Array::from_int32(id_arr));
    let col_name = FieldArray::from_arr("name", Array::from_string32(name_arr));
    let col_value = FieldArray::from_arr("value", Array::from_int64(value_arr));

    Table::new("TestTable".to_string(), Some(vec![col_id, col_name, col_value]))
}
