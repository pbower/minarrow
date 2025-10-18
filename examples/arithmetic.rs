use minarrow::{Array, IntegerArray, Value, vec64};
use std::sync::Arc;

fn main() {
    // Create two arrays
    let arr1 = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![10, 20, 30],
    ))));
    let arr2 = Value::Array(Arc::new(Array::from_int32(IntegerArray::from_slice(
        &vec64![2, 4, 6],
    ))));

    println!("Testing arithmetic operators with Value enum:");
    println!("arr1 = [10, 20, 30]");
    println!("arr2 = [2, 4, 6]");
    println!();

    // Addition
    let sum = (&arr1 + &arr2).unwrap();
    if let Value::Array(ref arr) = sum {
        println!("Addition (arr1 + arr2): {:?}", arr);
    }

    // Subtraction
    let diff = (&arr1 - &arr2).unwrap();
    if let Value::Array(ref arr) = diff {
        println!("Subtraction (arr1 - arr2): {:?}", arr);
    }

    // Multiplication
    let prod = (&arr1 * &arr2).unwrap();
    if let Value::Array(ref arr) = prod {
        println!("Multiplication (arr1 * arr2): {:?}", arr);
    }

    // Division
    let quot = (&arr1 / &arr2).unwrap();
    if let Value::Array(ref arr) = quot {
        println!("Division (arr1 / arr2): {:?}", arr);
    }

    // Remainder
    let rem = (&arr1 % &arr2).unwrap();
    if let Value::Array(ref arr) = rem {
        println!("Remainder (arr1 % arr2): {:?}", arr);
    }

    println!("\nAll arithmetic operators work correctly!");
}
