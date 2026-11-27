// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, MapArray, StructArray, UInt64Array, new_null_array,
};
use arrow::buffer::{OffsetBuffer, ScalarBuffer};
use arrow::compute::{take, take_arrays};
use arrow::datatypes::{DataType, Field, FieldRef, Fields};
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, exec_err, plan_err};
use datafusion_expr::{
    ColumnarValue, Documentation, HigherOrderFunctionArgs, HigherOrderReturnFieldArgs,
    HigherOrderSignature, HigherOrderUDFImpl, LambdaArgument, LambdaParametersProgress,
    ValueOrLambda, Volatility,
};
use datafusion_macros::user_doc;

use crate::function::type_coercion::spark_sql_type;

/// Spark-compatible `transform_values` higher-order function.
///
/// <https://spark.apache.org/docs/latest/api/sql/index.html#transform_values>
#[user_doc(
    doc_section(label = "Lambda Functions"),
    description = "Transforms the values of a map using a function. The result has the same keys in \
        the same order as the input, and each value is `func(key, value)`. The key is never NULL, \
        and the value is NULL when the map holds a NULL value. A NULL map returns NULL.",
    syntax_example = "transform_values(expr, func)",
    sql_example = r#"```sql
> SELECT transform_values(map_from_arrays(array(1, 2, 3), array(1, 2, 3)), (k, v) -> v + 1);
{1: 2, 2: 3, 3: 4}
> SELECT transform_values(map_from_arrays(array(1, 2, 3), array(1, 2, 3)), (k, v) -> k + v);
{1: 2, 2: 4, 3: 6}
```"#,
    argument(
        name = "expr",
        description = "Map expression. Other types are rejected at planning time. NULL returns NULL."
    ),
    argument(
        name = "func",
        description = "Lambda function `(key, value) -> new_value` with two parameters: the key and the value of an entry."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TransformValues {
    signature: HigherOrderSignature,
}

impl Default for TransformValues {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformValues {
    pub fn new() -> Self {
        Self {
            signature: HigherOrderSignature::exact(
                vec![ValueOrLambda::Value(()), ValueOrLambda::Lambda(())],
                Volatility::Immutable,
            ),
        }
    }
}

/// The entries field, key field, value field and `sorted` flag of a map type.
fn map_parts(data_type: &DataType) -> Option<(&FieldRef, &FieldRef, &FieldRef, bool)> {
    let DataType::Map(entries, sorted) = data_type else {
        return None;
    };
    let DataType::Struct(fields) = entries.data_type() else {
        return None;
    };
    match &**fields {
        [key, value] => Some((entries, key, value, *sorted)),
        _ => None,
    }
}

impl HigherOrderUDFImpl for TransformValues {
    fn name(&self) -> &str {
        "transform_values"
    }

    fn signature(&self) -> &HigherOrderSignature {
        &self.signature
    }

    fn coerce_value_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        let [map] = take_function_args(self.name(), arg_types)?;
        if map_parts(map).is_none() {
            return plan_err!(
                "[DATATYPE_MISMATCH.UNEXPECTED_INPUT_TYPE] The first parameter of `{}` \
                 requires the \"MAP\" type, however it has the type \"{}\". SQLSTATE: 42K09",
                self.name(),
                spark_sql_type(map)
            );
        }
        Ok(vec![map.clone()])
    }

    fn lambda_parameters(
        &self,
        _step: usize,
        fields: &[ValueOrLambda<FieldRef, Option<FieldRef>>],
    ) -> Result<LambdaParametersProgress> {
        let [ValueOrLambda::Value(map), ValueOrLambda::Lambda(_)] = fields else {
            return plan_err!("{} expects a map followed by a lambda", self.name());
        };
        let Some((_, key, value, _)) = map_parts(map.data_type()) else {
            return plan_err!("{} expects a map, got {}", self.name(), map.data_type());
        };
        // The key is never NULL, and the value is NULL only if the map may hold NULL values
        Ok(LambdaParametersProgress::Complete(vec![vec![
            Arc::new(key.as_ref().clone().with_nullable(false)),
            Arc::clone(value),
        ]]))
    }

    fn return_field_from_args(
        &self,
        args: HigherOrderReturnFieldArgs,
    ) -> Result<FieldRef> {
        let [ValueOrLambda::Value(map), ValueOrLambda::Lambda(lambda)] =
            take_function_args(self.name(), args.arg_fields)?
        else {
            return plan_err!("{} expects a map followed by a lambda", self.name());
        };
        let Some((entries, key, value, sorted)) = map_parts(map.data_type()) else {
            return plan_err!("{} expects a map, got {}", self.name(), map.data_type());
        };

        // The keys are reused as they are. `valueContainsNull` follows the lambda.
        let value = Arc::new(Field::new(
            value.name(),
            lambda.data_type().clone(),
            lambda.is_nullable(),
        ));
        let entries = Arc::new(Field::new(
            entries.name(),
            DataType::Struct(Fields::from(vec![Arc::clone(key), value])),
            entries.is_nullable(),
        ));
        Ok(Arc::new(Field::new(
            "",
            DataType::Map(entries, sorted),
            map.is_nullable(),
        )))
    }

    fn invoke_with_args(&self, args: HigherOrderFunctionArgs) -> Result<ColumnarValue> {
        let [map, lambda] = take_function_args(self.name(), &args.args)?;
        let (ValueOrLambda::Value(map), ValueOrLambda::Lambda(lambda)) = (map, lambda)
        else {
            return plan_err!("{} expects a map followed by a lambda", self.name());
        };

        let map = map.to_array(args.number_rows)?;
        let Some(map) = map.as_map_opt() else {
            return exec_err!("{} expects a map, got {}", self.name(), map.data_type());
        };
        reject_single_parameter_lambda(lambda, map.keys())?;
        let DataType::Map(entries_field, sorted) = args.return_type() else {
            return exec_err!(
                "{} expected the return field to be a map, got {}",
                self.name(),
                args.return_field
            );
        };
        let DataType::Struct(entries_fields) = entries_field.data_type() else {
            return exec_err!(
                "{} expected the map entries to be a struct, got {}",
                self.name(),
                entries_field.data_type()
            );
        };

        if map.null_count() == map.len() {
            return Ok(ColumnarValue::Array(new_null_array(
                args.return_type(),
                map.len(),
            )));
        }

        let FlatEntries {
            keys,
            values,
            offsets,
        } = flatten_entries(map)?;

        // by passing closures, lambda.evaluate calls only those the body needs
        let key_param = || Ok(Arc::clone(&keys));
        let value_param = || Ok(Arc::clone(&values));
        let new_values = lambda
            .evaluate(&[&key_param, &value_param], |captures| {
                // a captured column has one value per map row, so repeat it for every entry
                // of the row and drop it for rows without entries
                let rows = UInt64Array::from_iter_values(
                    offsets
                        .lengths()
                        .enumerate()
                        .flat_map(|(row, len)| std::iter::repeat_n(row as u64, len)),
                );
                Ok(take_arrays(captures, &rows, None)?)
            })?
            .into_array(keys.len())?;

        let entries =
            StructArray::try_new(entries_fields.clone(), vec![keys, new_values], None)?;
        Ok(ColumnarValue::Array(Arc::new(MapArray::try_new(
            Arc::clone(entries_field),
            offsets,
            entries,
            map.nulls().cloned(),
            *sorted,
        )?)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

/// Spark requires the lambda to take both the key and the value, and fails with
/// `INVALID_LAMBDA_FUNCTION_CALL.NUM_ARGS_MISMATCH` for `v -> v + 1`. DataFusion rejects a lambda
/// with more parameters than the function offers, but accepts one with fewer and binds the
/// parameters in order, so `v -> v + 1` would silently receive the key.
///
/// A [`LambdaArgument`] does not expose the parameters it declares. It does fail to evaluate
/// when given fewer arguments than it declares, so a lambda that declares a single parameter is
/// the one that evaluates when given only the keys. That evaluation is on no rows.
///
/// TODO: check the declared parameters directly once [`LambdaArgument`] exposes them.
fn reject_single_parameter_lambda(
    lambda: &LambdaArgument,
    keys: &ArrayRef,
) -> Result<()> {
    let no_keys = || Ok(keys.slice(0, 0));
    let single_parameter = lambda
        .evaluate(&[&no_keys], |captures| {
            Ok(take_arrays(captures, &UInt64Array::new_null(0), None)?)
        })
        .is_ok();
    if single_parameter {
        return plan_err!(
            "[INVALID_LAMBDA_FUNCTION_CALL.NUM_ARGS_MISMATCH] Invalid lambda function call. \
             A higher order function expects 2 arguments, but got 1. SQLSTATE: 42K0D"
        );
    }
    Ok(())
}

/// The entries of a map array that the lambda is evaluated on.
struct FlatEntries {
    keys: ArrayRef,
    values: ArrayRef,
    /// Offsets into `keys` and `values` for each row of the map array.
    offsets: OffsetBuffer<i32>,
}

/// Flattens the entries of the rows of a map array that are not NULL.
///
/// A NULL row can still have entries in the underlying arrays, and a sliced array keeps the
/// entries of the rows outside the slice. Neither may reach the lambda, because evaluating it
/// on them could fail (for example with a division by zero) for rows that are not part of the
/// result.
fn flatten_entries(map: &MapArray) -> Result<FlatEntries> {
    let offsets = map.value_offsets();
    let start = offsets[0];
    let end = offsets[offsets.len() - 1];

    let null_row_has_entries = map.null_count() > 0
        && (0..map.len()).any(|row| map.is_null(row) && offsets[row] != offsets[row + 1]);
    if !null_row_has_entries {
        // The entries of the rows are contiguous, so a slice is enough
        let (start, len) = (start as usize, (end - start) as usize);
        return Ok(FlatEntries {
            keys: map.keys().slice(start, len),
            values: map.values().slice(start, len),
            offsets: OffsetBuffer::new(
                offsets.iter().map(|offset| offset - start as i32).collect(),
            ),
        });
    }

    let mut indices = Vec::new();
    let mut new_offsets = Vec::with_capacity(map.len() + 1);
    new_offsets.push(0);
    for row in 0..map.len() {
        if map.is_valid(row) {
            indices.extend(offsets[row] as u64..offsets[row + 1] as u64);
        }
        new_offsets.push(indices.len() as i32);
    }
    let indices = UInt64Array::from(indices);
    Ok(FlatEntries {
        keys: take(map.keys().as_ref(), &indices, None)?,
        values: take(map.values().as_ref(), &indices, None)?,
        offsets: OffsetBuffer::new(ScalarBuffer::from(new_offsets)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::buffer::NullBuffer;

    /// Rows `[1: 10, 2: 20]`, `[3: 30]` (NULL), `[]`, `[4: 40, 5: 50]`
    fn map_with_null_row_entries() -> MapArray {
        let entries_field = Arc::new(Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Int32, false),
                Field::new("value", DataType::Int32, true),
            ])),
            false,
        ));
        let DataType::Struct(fields) = entries_field.data_type() else {
            unreachable!()
        };
        let entries = StructArray::new(
            fields.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50])),
            ],
            None,
        );
        MapArray::new(
            entries_field,
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 2, 3, 3, 5])),
            entries,
            Some(NullBuffer::from(vec![true, false, true, true])),
            false,
        )
    }

    fn ints(array: &ArrayRef) -> Vec<i32> {
        array
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    #[test]
    fn flatten_entries_drops_entries_of_null_rows() -> Result<()> {
        let flat = flatten_entries(&map_with_null_row_entries())?;
        assert_eq!(ints(&flat.keys), vec![1, 2, 4, 5]);
        assert_eq!(ints(&flat.values), vec![10, 20, 40, 50]);
        assert_eq!(
            flat.offsets.iter().copied().collect::<Vec<_>>(),
            [0, 2, 2, 2, 4]
        );
        Ok(())
    }

    #[test]
    fn flatten_entries_keeps_only_the_entries_of_a_slice() -> Result<()> {
        // rows 2 and 3 only: entries outside the slice must not reach the lambda
        let sliced = map_with_null_row_entries().slice(2, 2);
        let flat = flatten_entries(&sliced)?;
        assert_eq!(ints(&flat.keys), vec![4, 5]);
        assert_eq!(ints(&flat.values), vec![40, 50]);
        assert_eq!(flat.offsets.iter().copied().collect::<Vec<_>>(), [0, 0, 2]);
        Ok(())
    }

    #[test]
    fn flatten_entries_slice_with_null_row_entries() -> Result<()> {
        // rows 1..4: a NULL row that has entries, inside a slice that does not start at 0
        let sliced = map_with_null_row_entries().slice(1, 3);
        let flat = flatten_entries(&sliced)?;
        assert_eq!(ints(&flat.keys), vec![4, 5]);
        assert_eq!(
            flat.offsets.iter().copied().collect::<Vec<_>>(),
            [0, 0, 0, 2]
        );
        Ok(())
    }
}
