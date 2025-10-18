// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under MIT License.

use std::sync::Arc;

use crate::Bitmask;
#[cfg(feature = "cube")]
use crate::Cube;
use crate::enums::error::KernelError;
use crate::kernels::broadcast::table::broadcast_table_add;

#[cfg(all(feature = "cube", feature = "scalar_type"))]
use crate::Scalar;

#[cfg(feature = "cube")]
use crate::{Array, FieldArray, Table};

#[cfg(all(feature = "cube", feature = "views"))]
use crate::ArrayV;

use crate::enums::{error::MinarrowError, operators::ArithmeticOperator};

#[cfg(feature = "cube")]
use crate::kernels::broadcast::{
    array::broadcast_array_to_table,
    table::{broadcast_table_to_array, broadcast_table_to_scalar, broadcast_table_with_operator},
};

/// Broadcasts addition over Cube tables (3D structure)
///
/// Cubes are 3D structures where the Vec<Table> acts as the 'z' axis.
/// Both cubes must have the same number of tables and compatible table shapes.
/// Addition is applied table-wise between corresponding tables.
pub fn broadcast_cube_add(
    lhs: Cube,
    rhs: Cube,
    null_mask: Option<Arc<Bitmask>>,
) -> Result<Cube, KernelError> {
    // Check table count compatibility
    if lhs.tables.len() != rhs.tables.len() {
        return Err(KernelError::BroadcastingError(format!(
            "Cube table count mismatch: LHS {} tables, RHS {} tables",
            lhs.tables.len(),
            rhs.tables.len()
        )));
    }

    // Apply addition table-wise
    let mut result_tables = Vec::with_capacity(lhs.tables.len());

    for (i, (lhs_table, rhs_table)) in lhs.tables.iter().zip(rhs.tables.iter()).enumerate() {
        let result_table =
            broadcast_table_add(lhs_table.clone(), rhs_table.clone(), null_mask.clone()).map_err(
                |e| KernelError::BroadcastingError(format!("Table {} addition failed: {}", i, e)),
            )?;

        result_tables.push(result_table);
    }

    // Create result Cube with same metadata as left cube
    let result_n_rows: Vec<usize> = result_tables.iter().map(|t| t.n_rows()).collect();
    Ok(Cube {
        tables: result_tables,
        n_rows: result_n_rows,
        name: lhs.name.clone(),
        third_dim_index: lhs.third_dim_index.clone(),
    })
}

/// Broadcast Cube to Scalar - apply scalar to each table in the cube
#[cfg(all(feature = "cube", feature = "scalar_type"))]
pub fn broadcast_cube_to_scalar(
    op: ArithmeticOperator,
    cube: &Cube,
    scalar: &Scalar,
) -> Result<Cube, MinarrowError> {
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_table_to_scalar(op, table, scalar)?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast Cube to Array - apply array to each table in the cube
#[cfg(feature = "cube")]
pub fn broadcast_cube_to_array(
    op: ArithmeticOperator,
    cube: &Cube,
    array: &Array,
) -> Result<Cube, MinarrowError> {
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_table_to_array(op, table, array)?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast FieldArray to Cube - apply field array's inner array to each table in the cube
#[cfg(feature = "cube")]
pub fn broadcast_fieldarray_to_cube(
    op: ArithmeticOperator,
    field_array: &FieldArray,
    cube: &Cube,
) -> Result<Cube, MinarrowError> {
    let array = &field_array.array;
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_array_to_table(op, array, table)?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast Cube to FieldArray - apply field array's inner array to each table in the cube
#[cfg(feature = "cube")]
pub fn broadcast_cube_to_fieldarray(
    op: ArithmeticOperator,
    cube: &Cube,
    field_array: &FieldArray,
) -> Result<Cube, MinarrowError> {
    let array = &field_array.array;
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_table_to_array(op, table, array)?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast Table to Cube - apply table to each table in the cube
#[cfg(feature = "cube")]
pub fn broadcast_table_to_cube(
    op: ArithmeticOperator,
    table: &Table,
    cube: &Cube,
) -> Result<Cube, MinarrowError> {
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for cube_table in &cube.tables {
        let broadcasted = broadcast_table_with_operator(op, table.clone(), cube_table.clone())?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast Cube to Table - apply table to each table in the cube
#[cfg(feature = "cube")]
pub fn broadcast_cube_to_table(
    op: ArithmeticOperator,
    cube: &Cube,
    table: &Table,
) -> Result<Cube, MinarrowError> {
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for cube_table in &cube.tables {
        let broadcasted = broadcast_table_with_operator(op, cube_table.clone(), table.clone())?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast ArrayView to Cube - apply array view to each table in the cube
#[cfg(all(feature = "cube", feature = "views"))]
pub fn broadcast_arrayview_to_cube(
    op: ArithmeticOperator,
    array_view: &crate::ArrayV,
    cube: &Cube,
) -> Result<Cube, MinarrowError> {
    use crate::kernels::broadcast::array_view::broadcast_arrayview_to_table;
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_arrayview_to_table(op, array_view, table)?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast Cube to ArrayView - apply array view to each table in the cube
#[cfg(all(feature = "cube", feature = "views"))]
pub fn broadcast_cube_to_arrayview(
    op: ArithmeticOperator,
    cube: &Cube,
    array_view: &crate::ArrayV,
) -> Result<Cube, MinarrowError> {
    use crate::kernels::broadcast::table::broadcast_table_to_arrayview;
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_table_to_arrayview(op, table, array_view)?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast NumericArrayView to Cube - apply numeric array view to each table in the cube
#[cfg(all(feature = "cube", feature = "views"))]
pub fn broadcast_numericarrayview_to_cube(
    op: ArithmeticOperator,
    num_array_view: &crate::NumericArrayV,
    cube: &Cube,
) -> Result<Cube, MinarrowError> {
    use crate::kernels::broadcast::array_view::broadcast_arrayview_to_table;

    let array_view: ArrayV = num_array_view.clone().into();
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_arrayview_to_table(op, &array_view, table)?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast Cube to NumericArrayView - apply numeric array view to each table in the cube
#[cfg(all(feature = "cube", feature = "views"))]
pub fn broadcast_cube_to_numericarrayview(
    op: ArithmeticOperator,
    cube: &Cube,
    num_array_view: &crate::NumericArrayV,
) -> Result<Cube, MinarrowError> {
    use crate::kernels::broadcast::table::broadcast_table_to_arrayview;

    let array_view: ArrayV = num_array_view.clone().into();
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_table_to_arrayview(op, table, &array_view)?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast TextArrayView to Cube - apply text array view to each table in the cube
#[cfg(all(feature = "cube", feature = "views"))]
pub fn broadcast_textarrayview_to_cube(
    op: ArithmeticOperator,
    text_array_view: &crate::TextArrayV,
    cube: &Cube,
) -> Result<Cube, MinarrowError> {
    use crate::kernels::broadcast::array_view::broadcast_arrayview_to_table;

    let array_view: ArrayV = text_array_view.clone().into();
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_arrayview_to_table(op, &array_view, table)?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast Cube to TextArrayView - apply text array view to each table in the cube
#[cfg(all(feature = "cube", feature = "views"))]
pub fn broadcast_cube_to_textarrayview(
    op: ArithmeticOperator,
    cube: &Cube,
    text_array_view: &crate::TextArrayV,
) -> Result<Cube, MinarrowError> {
    use crate::kernels::broadcast::table::broadcast_table_to_arrayview;

    let array_view: ArrayV = text_array_view.clone().into();
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_table_to_arrayview(op, table, &array_view)?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast TemporalArrayView to Cube - apply temporal array view to each table in the cube
#[cfg(all(feature = "cube", feature = "views", feature = "datetime"))]
pub fn broadcast_temporalarrayview_to_cube(
    op: ArithmeticOperator,
    temporal_array_view: &crate::TemporalArrayV,
    cube: &Cube,
) -> Result<Cube, MinarrowError> {
    use crate::kernels::broadcast::array_view::broadcast_arrayview_to_table;

    let array_view: ArrayV = temporal_array_view.clone().into();
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_arrayview_to_table(op, &array_view, table)?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast Cube to TemporalArrayView - apply temporal array view to each table in the cube
#[cfg(all(feature = "cube", feature = "views", feature = "datetime"))]
pub fn broadcast_cube_to_temporalarrayview(
    op: ArithmeticOperator,
    cube: &Cube,
    temporal_array_view: &crate::TemporalArrayV,
) -> Result<Cube, MinarrowError> {
    use crate::kernels::broadcast::table::broadcast_table_to_arrayview;

    let array_view: ArrayV = temporal_array_view.clone().into();
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_table_to_arrayview(op, table, &array_view)?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast TableView to Cube - apply table view to each table in the cube
#[cfg(all(feature = "cube", feature = "views"))]
pub fn broadcast_tableview_to_cube(
    op: ArithmeticOperator,
    table_view: &crate::TableV,
    cube: &Cube,
) -> Result<Cube, MinarrowError> {
    let table = table_view.to_table();
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for cube_table in &cube.tables {
        let broadcasted = broadcast_table_with_operator(op, table.clone(), cube_table.clone())?;
        result_tables.push(broadcasted);
    }
    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Helper: Materialize SuperArray into a single contiguous Array by concatenating chunks
#[cfg(feature = "chunked")]
fn materialize_super_array(super_array: &crate::SuperArray) -> Result<Array, MinarrowError> {
    use crate::NumericArray;

    let chunks = super_array.chunks();
    if chunks.is_empty() {
        return Ok(Array::Null);
    }

    // Get the first chunk to determine type
    let first_array = &chunks[0].array;

    // For now, handle numeric arrays (can be extended for other types)
    match first_array {
        Array::NumericArray(NumericArray::Int32(_)) => {
            use crate::IntegerArray;
            let mut all_values = Vec::new();
            for chunk in chunks {
                if let Array::NumericArray(NumericArray::Int32(arr)) = &chunk.array {
                    all_values.extend_from_slice(arr.data.as_slice());
                } else {
                    return Err(MinarrowError::TypeError {
                        from: "SuperArray chunks",
                        to: "Int32",
                        message: Some("All chunks must have same type".to_string()),
                    });
                }
            }
            Ok(Array::from_int32(IntegerArray::from_vec(all_values, None)))
        }
        Array::NumericArray(NumericArray::Int64(_)) => {
            use crate::IntegerArray;
            let mut all_values = Vec::new();
            for chunk in chunks {
                if let Array::NumericArray(NumericArray::Int64(arr)) = &chunk.array {
                    all_values.extend_from_slice(arr.data.as_slice());
                } else {
                    return Err(MinarrowError::TypeError {
                        from: "SuperArray chunks",
                        to: "Int64",
                        message: Some("All chunks must have same type".to_string()),
                    });
                }
            }
            Ok(Array::from_int64(IntegerArray::from_vec(all_values, None)))
        }
        Array::NumericArray(NumericArray::Float32(_)) => {
            use crate::FloatArray;
            let mut all_values = Vec::new();
            for chunk in chunks {
                if let Array::NumericArray(NumericArray::Float32(arr)) = &chunk.array {
                    all_values.extend_from_slice(arr.data.as_slice());
                } else {
                    return Err(MinarrowError::TypeError {
                        from: "SuperArray chunks",
                        to: "Float32",
                        message: Some("All chunks must have same type".to_string()),
                    });
                }
            }
            Ok(Array::from_float32(FloatArray::from_vec(all_values, None)))
        }
        Array::NumericArray(NumericArray::Float64(_)) => {
            use crate::FloatArray;
            let mut all_values = Vec::new();
            for chunk in chunks {
                if let Array::NumericArray(NumericArray::Float64(arr)) = &chunk.array {
                    all_values.extend_from_slice(arr.data.as_slice());
                } else {
                    return Err(MinarrowError::TypeError {
                        from: "SuperArray chunks",
                        to: "Float64",
                        message: Some("All chunks must have same type".to_string()),
                    });
                }
            }
            Ok(Array::from_float64(FloatArray::from_vec(all_values, None)))
        }
        _ => {
            // For other types, fall back to slower clone-based concatenation
            Err(MinarrowError::NotImplemented {
                feature: format!(
                    "Materialization not yet implemented for array type: {:?}",
                    first_array
                ),
            })
        }
    }
}

/// Helper: Materialize SuperArrayView into a single contiguous Array by concatenating view slices
#[cfg(all(feature = "chunked", feature = "views"))]
fn materialize_super_array_view(
    super_array_view: &crate::SuperArrayV,
) -> Result<Array, MinarrowError> {
    if super_array_view.slices.is_empty() {
        return Ok(Array::Null);
    }

    // Materialize first slice to determine type
    let first_array = super_array_view.slices[0].to_array();

    use crate::NumericArray;
    match first_array {
        Array::NumericArray(NumericArray::Int32(_)) => {
            use crate::IntegerArray;
            let mut all_values = Vec::new();
            for slice_view in &super_array_view.slices {
                let array = slice_view.to_array();
                if let Array::NumericArray(NumericArray::Int32(arr)) = array {
                    all_values.extend_from_slice(arr.data.as_slice());
                } else {
                    return Err(MinarrowError::TypeError {
                        from: "SuperArrayView slices",
                        to: "Int32",
                        message: Some("All slices must have same type".to_string()),
                    });
                }
            }
            Ok(Array::from_int32(IntegerArray::from_vec(all_values, None)))
        }
        Array::NumericArray(NumericArray::Int64(_)) => {
            use crate::IntegerArray;
            let mut all_values = Vec::new();
            for slice_view in &super_array_view.slices {
                let array = slice_view.to_array();
                if let Array::NumericArray(NumericArray::Int64(arr)) = array {
                    all_values.extend_from_slice(arr.data.as_slice());
                } else {
                    return Err(MinarrowError::TypeError {
                        from: "SuperArrayView slices",
                        to: "Int64",
                        message: Some("All slices must have same type".to_string()),
                    });
                }
            }
            Ok(Array::from_int64(IntegerArray::from_vec(all_values, None)))
        }
        Array::NumericArray(NumericArray::Float32(_)) => {
            use crate::FloatArray;
            let mut all_values = Vec::new();
            for slice_view in &super_array_view.slices {
                let array = slice_view.to_array();
                if let Array::NumericArray(NumericArray::Float32(arr)) = array {
                    all_values.extend_from_slice(arr.data.as_slice());
                } else {
                    return Err(MinarrowError::TypeError {
                        from: "SuperArrayView slices",
                        to: "Float32",
                        message: Some("All slices must have same type".to_string()),
                    });
                }
            }
            Ok(Array::from_float32(FloatArray::from_vec(all_values, None)))
        }
        Array::NumericArray(NumericArray::Float64(_)) => {
            use crate::FloatArray;
            let mut all_values = Vec::new();
            for slice_view in &super_array_view.slices {
                let array = slice_view.to_array();
                if let Array::NumericArray(NumericArray::Float64(arr)) = array {
                    all_values.extend_from_slice(arr.data.as_slice());
                } else {
                    return Err(MinarrowError::TypeError {
                        from: "SuperArrayView slices",
                        to: "Float64",
                        message: Some("All slices must have same type".to_string()),
                    });
                }
            }
            Ok(Array::from_float64(FloatArray::from_vec(all_values, None)))
        }
        _ => Err(MinarrowError::NotImplemented {
            feature: "SuperArrayView materialization for this array type".to_string(),
        }),
    }
}

/// Helper: Merge SuperTableView slices into a single Table by vertical concatenation
#[cfg(all(feature = "chunked", feature = "views"))]
fn merge_supertableview_slices(
    super_table_view: &crate::SuperTableV,
) -> Result<Table, MinarrowError> {
    if super_table_view.slices.is_empty() {
        return Err(MinarrowError::ShapeError {
            message: "Cannot merge empty SuperTableView".to_string(),
        });
    }

    // Convert all slices to tables and merge using existing helper
    let tables: Vec<Table> = super_table_view
        .slices
        .iter()
        .map(|slice| slice.to_table())
        .collect();

    // Create a temporary SuperTable from the tables and merge it
    let temp_super_table = crate::SuperTable::from_batches(
        tables.into_iter().map(Arc::new).collect(),
        Some("temp".to_string()),
    );

    merge_supertable_batches(&temp_super_table)
}

/// Helper: Merge SuperTable batches into a single Table by vertical concatenation
#[cfg(feature = "chunked")]
fn merge_supertable_batches(super_table: &crate::SuperTable) -> Result<Table, MinarrowError> {
    use crate::NumericArray;

    if super_table.batches.is_empty() {
        return Err(MinarrowError::ShapeError {
            message: "Cannot merge empty SuperTable".to_string(),
        });
    }

    let first_table = &super_table.batches[0];
    let n_cols = first_table.n_cols();

    // Validate all batches have same column count
    for (i, batch) in super_table.batches.iter().enumerate() {
        if batch.n_cols() != n_cols {
            return Err(MinarrowError::ShapeError {
                message: format!(
                    "SuperTable batch {} has {} columns, expected {}",
                    i,
                    batch.n_cols(),
                    n_cols
                ),
            });
        }
    }

    // Concatenate each column vertically across all batches
    let mut merged_cols = Vec::with_capacity(n_cols);

    for col_idx in 0..n_cols {
        let first_col = &first_table.cols[col_idx];
        let field = first_col.field.clone();

        // Collect all values for this column across batches
        match &first_col.array {
            Array::NumericArray(NumericArray::Int32(_)) => {
                use crate::IntegerArray;
                let mut all_values = Vec::new();
                for batch in &super_table.batches {
                    if let Array::NumericArray(NumericArray::Int32(arr)) =
                        &batch.cols[col_idx].array
                    {
                        all_values.extend_from_slice(arr.data.as_slice());
                    } else {
                        return Err(MinarrowError::TypeError {
                            from: "SuperTable batch column",
                            to: "Int32",
                            message: Some("All batch columns must have same type".to_string()),
                        });
                    }
                }
                merged_cols.push(FieldArray::new(
                    field.as_ref().clone(),
                    Array::from_int32(IntegerArray::from_vec(all_values, None)),
                ));
            }
            Array::NumericArray(NumericArray::Int64(_)) => {
                use crate::IntegerArray;
                let mut all_values = Vec::new();
                for batch in &super_table.batches {
                    if let Array::NumericArray(NumericArray::Int64(arr)) =
                        &batch.cols[col_idx].array
                    {
                        all_values.extend_from_slice(arr.data.as_slice());
                    } else {
                        return Err(MinarrowError::TypeError {
                            from: "SuperTable batch column",
                            to: "Int64",
                            message: Some("All batch columns must have same type".to_string()),
                        });
                    }
                }
                merged_cols.push(FieldArray::new(
                    field.as_ref().clone(),
                    Array::from_int64(IntegerArray::from_vec(all_values, None)),
                ));
            }
            Array::NumericArray(NumericArray::Float32(_)) => {
                use crate::FloatArray;
                let mut all_values = Vec::new();
                for batch in &super_table.batches {
                    if let Array::NumericArray(NumericArray::Float32(arr)) =
                        &batch.cols[col_idx].array
                    {
                        all_values.extend_from_slice(arr.data.as_slice());
                    } else {
                        return Err(MinarrowError::TypeError {
                            from: "SuperTable batch column",
                            to: "Float32",
                            message: Some("All batch columns must have same type".to_string()),
                        });
                    }
                }
                merged_cols.push(FieldArray::new(
                    field.as_ref().clone(),
                    Array::from_float32(FloatArray::from_vec(all_values, None)),
                ));
            }
            Array::NumericArray(NumericArray::Float64(_)) => {
                use crate::FloatArray;
                let mut all_values = Vec::new();
                for batch in &super_table.batches {
                    if let Array::NumericArray(NumericArray::Float64(arr)) =
                        &batch.cols[col_idx].array
                    {
                        all_values.extend_from_slice(arr.data.as_slice());
                    } else {
                        return Err(MinarrowError::TypeError {
                            from: "SuperTable batch column",
                            to: "Float64",
                            message: Some("All batch columns must have same type".to_string()),
                        });
                    }
                }
                merged_cols.push(FieldArray::new(
                    field.as_ref().clone(),
                    Array::from_float64(FloatArray::from_vec(all_values, None)),
                ));
            }
            _ => {
                return Err(MinarrowError::NotImplemented {
                    feature: format!(
                        "Merging not yet implemented for array type in column {}",
                        col_idx
                    ),
                });
            }
        }
    }

    Ok(Table::new(super_table.name.clone(), Some(merged_cols)))
}

/// Broadcast SuperArray to Cube - apply super array against each table in the cube (left operand)
#[cfg(all(feature = "cube", feature = "chunked"))]
pub fn broadcast_superarray_to_cube(
    op: ArithmeticOperator,
    super_array: &crate::SuperArray,
    cube: &Cube,
) -> Result<Cube, MinarrowError> {
    // Validate cube shape: all tables must have same row count
    let expected_rows = cube
        .tables
        .first()
        .ok_or_else(|| MinarrowError::ShapeError {
            message: "Cannot broadcast to empty cube".to_string(),
        })?
        .n_rows;

    for (i, table) in cube.tables.iter().enumerate() {
        if table.n_rows != expected_rows {
            return Err(MinarrowError::ShapeError {
                message: format!(
                    "Cube tables must have equal row counts: table 0 has {} rows, table {} has {} rows",
                    expected_rows, i, table.n_rows
                ),
            });
        }
    }

    // Validate SuperArray length matches cube row count
    if super_array.len() != expected_rows {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperArray length ({}) doesn't match cube row count ({})",
                super_array.len(),
                expected_rows
            ),
        });
    }

    // Materialize SuperArray chunks into a single array for efficient broadcasting
    // Concatenate all chunks into one contiguous array
    let materialized_array = materialize_super_array(super_array)?;

    // Broadcast materialized array to each table in the cube
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_array_to_table(op, &materialized_array, table)?;
        result_tables.push(broadcasted);
    }

    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast Cube to SuperArray - apply super array to each table in the cube
#[cfg(all(feature = "cube", feature = "chunked"))]
pub fn broadcast_cube_to_superarray(
    op: ArithmeticOperator,
    cube: &Cube,
    super_array: &crate::SuperArray,
) -> Result<Cube, MinarrowError> {
    // Validate cube shape: all tables must have same row count
    let expected_rows = cube
        .tables
        .first()
        .ok_or_else(|| MinarrowError::ShapeError {
            message: "Cannot broadcast empty cube".to_string(),
        })?
        .n_rows;

    for (i, table) in cube.tables.iter().enumerate() {
        if table.n_rows != expected_rows {
            return Err(MinarrowError::ShapeError {
                message: format!(
                    "Cube tables must have equal row counts: table 0 has {} rows, table {} has {} rows",
                    expected_rows, i, table.n_rows
                ),
            });
        }
    }

    // Validate SuperArray length matches cube row count
    if super_array.len() != expected_rows {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperArray length ({}) doesn't match cube row count ({})",
                super_array.len(),
                expected_rows
            ),
        });
    }

    // Materialize SuperArray chunks into a single array for efficient broadcasting
    let materialized_array = materialize_super_array(super_array)?;

    // Broadcast each table to the materialized array
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_table_to_array(op, table, &materialized_array)?;
        result_tables.push(broadcasted);
    }

    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast SuperTable to Cube - apply super table to each table in the cube
#[cfg(all(feature = "cube", feature = "chunked"))]
pub fn broadcast_supertable_to_cube(
    op: ArithmeticOperator,
    super_table: &crate::SuperTable,
    cube: &Cube,
) -> Result<Cube, MinarrowError> {
    // Merge SuperTable batches into a single logical table
    let merged_table = merge_supertable_batches(super_table)?;

    // Broadcast the merged table against each cube table
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for cube_table in &cube.tables {
        let broadcasted =
            broadcast_table_with_operator(op, merged_table.clone(), cube_table.clone())?;
        result_tables.push(broadcasted);
    }

    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast Cube to SuperTable - apply super table to each table in the cube
#[cfg(all(feature = "cube", feature = "chunked"))]
pub fn broadcast_cube_to_supertable(
    op: ArithmeticOperator,
    cube: &Cube,
    super_table: &crate::SuperTable,
) -> Result<Cube, MinarrowError> {
    // Merge SuperTable batches into a single logical table
    let merged_table = merge_supertable_batches(super_table)?;

    // Broadcast each cube table against the merged table
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for cube_table in &cube.tables {
        let broadcasted =
            broadcast_table_with_operator(op, cube_table.clone(), merged_table.clone())?;
        result_tables.push(broadcasted);
    }

    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast SuperArrayView to Cube - apply super array view to each table in the cube
#[cfg(all(feature = "cube", feature = "chunked", feature = "views"))]
pub fn broadcast_superarrayview_to_cube(
    op: ArithmeticOperator,
    super_array_view: &crate::SuperArrayV,
    cube: &Cube,
) -> Result<Cube, MinarrowError> {
    use crate::kernels::broadcast::array::broadcast_array_to_table;

    // Validate cube shape
    let expected_rows = cube
        .tables
        .first()
        .ok_or_else(|| MinarrowError::ShapeError {
            message: "Cannot broadcast to empty cube".to_string(),
        })?
        .n_rows;

    for (i, table) in cube.tables.iter().enumerate() {
        if table.n_rows != expected_rows {
            return Err(MinarrowError::ShapeError {
                message: format!(
                    "Cube tables must have equal row counts: table 0 has {} rows, table {} has {} rows",
                    expected_rows, i, table.n_rows
                ),
            });
        }
    }

    // Validate SuperArrayView length
    if super_array_view.len != expected_rows {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperArrayView length ({}) doesn't match cube row count ({})",
                super_array_view.len, expected_rows
            ),
        });
    }

    // Materialize view slices using helper
    let materialized = materialize_super_array_view(super_array_view)?;

    // Broadcast to each table
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_array_to_table(op, &materialized, table)?;
        result_tables.push(broadcasted);
    }

    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast Cube to SuperArrayView - apply super array view to each table in the cube
#[cfg(all(feature = "cube", feature = "chunked", feature = "views"))]
pub fn broadcast_cube_to_superarrayview(
    op: ArithmeticOperator,
    cube: &Cube,
    super_array_view: &crate::SuperArrayV,
) -> Result<Cube, MinarrowError> {
    // Validate cube shape: all tables must have same row count
    let expected_rows = cube
        .tables
        .first()
        .ok_or_else(|| MinarrowError::ShapeError {
            message: "Cannot broadcast empty cube".to_string(),
        })?
        .n_rows;

    for (i, table) in cube.tables.iter().enumerate() {
        if table.n_rows != expected_rows {
            return Err(MinarrowError::ShapeError {
                message: format!(
                    "Cube tables must have equal row counts: table 0 has {} rows, table {} has {} rows",
                    expected_rows, i, table.n_rows
                ),
            });
        }
    }

    // Validate SuperArrayView length matches cube row count
    if super_array_view.len != expected_rows {
        return Err(MinarrowError::ShapeError {
            message: format!(
                "SuperArrayView length ({}) doesn't match cube row count ({})",
                super_array_view.len, expected_rows
            ),
        });
    }

    // Materialize view slices using helper
    let materialized = materialize_super_array_view(super_array_view)?;

    // Broadcast each table to the materialized array
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for table in &cube.tables {
        let broadcasted = broadcast_table_to_array(op, table, &materialized)?;
        result_tables.push(broadcasted);
    }

    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast SuperTableView to Cube - apply super table view to each table in the cube
#[cfg(all(feature = "cube", feature = "chunked", feature = "views"))]
pub fn broadcast_supertableview_to_cube(
    op: ArithmeticOperator,
    super_table_view: &crate::SuperTableV,
    cube: &Cube,
) -> Result<Cube, MinarrowError> {
    // Merge SuperTableView slices into a single logical table
    let merged_table = merge_supertableview_slices(super_table_view)?;

    // Broadcast the merged table against each cube table
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for cube_table in &cube.tables {
        let broadcasted =
            broadcast_table_with_operator(op, merged_table.clone(), cube_table.clone())?;
        result_tables.push(broadcasted);
    }

    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

/// Broadcast Cube to SuperTableView - apply super table view to each table in the cube
#[cfg(all(feature = "cube", feature = "chunked", feature = "views"))]
pub fn broadcast_cube_to_supertableview(
    op: ArithmeticOperator,
    cube: &Cube,
    super_table_view: &crate::SuperTableV,
) -> Result<Cube, MinarrowError> {
    // Merge SuperTableView slices into a single logical table
    let merged_table = merge_supertableview_slices(super_table_view)?;

    // Broadcast each cube table against the merged table
    let mut result_tables = Vec::with_capacity(cube.tables.len());
    for cube_table in &cube.tables {
        let broadcasted =
            broadcast_table_with_operator(op, cube_table.clone(), merged_table.clone())?;
        result_tables.push(broadcasted);
    }

    Ok(Cube {
        tables: result_tables,
        n_rows: cube.n_rows.clone(),
        name: cube.name.clone(),
        third_dim_index: cube.third_dim_index.clone(),
    })
}

#[cfg(all(test, feature = "cube"))]
mod tests {
    use super::*;
    use crate::{Array, FieldArray, IntegerArray, Table, vec64};

    fn create_test_table(name: &str, base_val: i32) -> Table {
        // Create a simple table with 2 columns and 2 rows
        let col1 = FieldArray::from_inner(
            "col1",
            Array::from_int32(IntegerArray::from_slice(&vec64![base_val, base_val + 1])),
        );
        let col2 = FieldArray::from_inner(
            "col2",
            Array::from_int32(IntegerArray::from_slice(&vec64![
                base_val * 10,
                (base_val + 1) * 10
            ])),
        );

        Table::new(format!("{}_{}", name, base_val), Some(vec![col1, col2]))
    }

    #[test]
    fn test_cube_addition() {
        // Create first cube with 2 tables
        let table1_a = create_test_table("table1", 1); // [1,2] and [10,20]
        let table2_a = create_test_table("table2", 3); // [3,4] and [30,40]
        let cube_a = Cube {
            tables: vec![table1_a, table2_a],
            n_rows: vec![2, 2],
            name: "cubeA".to_string(),
            third_dim_index: None,
        };

        // Create second cube with 2 tables
        let table1_b = create_test_table("table1", 5); // [5,6] and [50,60]
        let table2_b = create_test_table("table2", 7); // [7,8] and [70,80]
        let cube_b = Cube {
            tables: vec![table1_b, table2_b],
            n_rows: vec![2, 2],
            name: "cubeB".to_string(),
            third_dim_index: None,
        };

        // Perform addition
        let result = broadcast_cube_add(cube_a, cube_b, None).unwrap();

        assert_eq!(result.tables.len(), 2);
        assert_eq!(result.name, "cubeA"); // Takes name from left operand
        assert_eq!(result.n_rows, vec![2, 2]);

        // Check first table: [1,2] + [5,6] = [6,8] and [10,20] + [50,60] = [60,80]
        let first_table = &result.tables[0];

        // Check first column of first table: [1,2] + [5,6] = [6,8]
        if let Some(col1) = first_table.col(0) {
            if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &col1.array {
                assert_eq!(arr.data.as_slice(), &[6, 8]);
            } else {
                panic!("Expected Int32 array in first column");
            }
        } else {
            panic!("Could not get first column");
        }

        // Check second column of first table: [10,20] + [50,60] = [60,80]
        if let Some(col2) = first_table.col(1) {
            if let crate::Array::NumericArray(crate::NumericArray::Int32(arr)) = &col2.array {
                assert_eq!(arr.data.as_slice(), &[60, 80]);
            } else {
                panic!("Expected Int32 array in second column");
            }
        } else {
            panic!("Could not get second column");
        }
    }

    #[test]
    #[should_panic(expected = "table count mismatch")]
    fn test_mismatched_table_count() {
        let table1_a = create_test_table("table1", 1);
        let cube_a = Cube {
            tables: vec![table1_a], // 1 table
            n_rows: vec![2],
            name: "cubeA".to_string(),
            third_dim_index: None,
        };

        let table1_b = create_test_table("table1", 5);
        let table2_b = create_test_table("table2", 7);
        let cube_b = Cube {
            tables: vec![table1_b, table2_b], // 2 tables
            n_rows: vec![2, 2],
            name: "cubeB".to_string(),
            third_dim_index: None,
        };

        let _ = broadcast_cube_add(cube_a, cube_b, None).unwrap();
    }
}
