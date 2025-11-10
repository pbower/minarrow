use minarrow::traits::selection::{ColumnSelection, RowSelection};
use minarrow::*;

fn main() {
    // Create a sample table
    let table = create_sample_table();

    println!("Original table:");
    println!("{}\n", table);

    // Table-specific API
    println!("=== Table-specific API ===\n");

    println!("table.c(&[\"name\", \"age\"]).r(1..4)");
    let view1 = table.c(&["name", "age"]).r(1..4);
    println!("{}\n", view1);

    println!("table.c(&[0, 2]).r(&[0, 2, 4])");
    let view2 = table.c(&[0, 2]).r(&[0, 2, 4]);
    println!("{}\n", view2);

    println!("table.c(0..2).r(0..5)");
    let view3 = table.c(0..2).r(0..5);
    println!("{}\n", view3);

    println!("table.c(1).r(&[2, 4, 6])");
    let view4 = table.c(1).r(&[2, 4, 6]);
    println!("{}\n", view4);

    // Aliased API
    println!("=== Aliases ===\n");

    println!("table.f(&[\"id\", \"age\"]).d(0..3)");
    let view5 = table.c(&["id", "age"]).r(0..3);
    println!("{}\n", view5);

    println!("table.fields(&[\"name\"]).data(1..5)");
    let view6 = table.c(&["name"]).r(1..5);
    println!("{}\n", view6);

    println!("table.f(0..2).d(&[0, 3, 6, 9])");
    let view7 = table.c(0..2).r(&[0, 3, 6, 9]);
    println!("{}\n", view7);

    println!("table.y(2).x(..5)");
    let view8 = table.y(2).y(..5);
    println!("{}\n", view8);

    println!("table.fields(1..).data(5..)");
    let view9 = table.c(1..).r(5..);
    println!("{}\n", view9);

    // Materialise selections
    println!("=== Materialisation ===\n");

    println!("table.f(&[\"name\", \"age\"]).d(0..3).to_table()");
    let view = table.c(&["name", "age"]).r(0..3);
    let materialised = view.to_table();
    println!("{}\n", materialised);

    // Array and FieldArray selection
    println!("=== Array & FieldArray Selection ===\n");

    // Get a single column as FieldArray
    let age_col = table.col(2).unwrap().clone();
    println!("Age column (FieldArray):");
    println!("  Field: {} ({})", age_col.field.name, age_col.arrow_type());
    println!("  Length: {}", age_col.len());
    println!(
        "  Values: {:?}\n",
        (0..age_col.len())
            .map(|i| age_col.array.inner::<IntegerArray<i32>>().get(i).unwrap())
            .collect::<Vec<_>>()
    );

    // Select specific rows from FieldArray using .d()
    println!("age_col.d(&[1, 3, 5, 7])");
    let age_view = age_col.r(&[1, 3, 5, 7]);
    println!("  View length: {}", age_view.len());
    println!(
        "  Selected indices: {:?}\n",
        (0..age_view.len())
            .map(|i| age_view.get::<IntegerArray<i32>>(i).unwrap())
            .collect::<Vec<_>>()
    );

    // ArrayV selection (direct array view)
    let id_array = Array::from_int32({
        let mut arr = IntegerArray::<i32>::default();
        for i in 0..10 {
            arr.push(i * 10);
        }
        arr
    });
    let id_view = ArrayV::from(id_array);
    println!("ArrayV:");
    println!("  Length: {}", id_view.len());
    println!(
        "  Values: {:?}\n",
        (0..id_view.len())
            .map(|i| id_view.get::<IntegerArray<i32>>(i).unwrap())
            .collect::<Vec<_>>()
    );

    // Select from ArrayV using .data() and .x() aliases
    println!("id_view.data(0..5)");
    let id_selected1 = id_view.select_rows(0..5);
    println!("  Length: {}", id_selected1.len());
    println!(
        "  Values: {:?}\n",
        (0..id_selected1.len())
            .map(|i| id_selected1.get::<IntegerArray<i32>>(i).unwrap())
            .collect::<Vec<_>>()
    );

    println!("id_view.x(&[2, 4, 6, 8])");
    let id_selected2 = id_view.y(&[2, 4, 6, 8]);
    println!("  Length: {}", id_selected2.len());
    println!(
        "  Values: {:?}",
        (0..id_selected2.len())
            .map(|i| id_selected2.get::<IntegerArray<i32>>(i).unwrap())
            .collect::<Vec<_>>()
    );
}

fn create_sample_table() -> Table {
    // Create id column
    let mut id_arr = IntegerArray::<i32>::default();
    for i in 0..10 {
        id_arr.push(i);
    }

    // Create name column
    let mut name_arr = StringArray::<u32>::default();
    let names = [
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack",
    ];
    for name in names {
        name_arr.push(name.to_string());
    }

    // Create age column
    let mut age_arr = IntegerArray::<i32>::default();
    for i in 0..10 {
        age_arr.push(20 + (i * 3) as i32);
    }

    let col_id = FieldArray::from_arr("id", Array::from_int32(id_arr));
    let col_name = FieldArray::from_arr("name", Array::from_string32(name_arr));
    let col_age = FieldArray::from_arr("age", Array::from_int32(age_arr));

    Table::new("People".to_string(), Some(vec![col_id, col_name, col_age]))
}
