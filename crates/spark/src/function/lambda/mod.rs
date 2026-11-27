// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Spark higher-order (lambda) functions.

pub mod transform_values;

use datafusion_expr::HigherOrderUDF;
use std::sync::Arc;

pub mod expr_fn {}

pub fn functions() -> Vec<Arc<HigherOrderUDF>> {
    vec![Arc::new(HigherOrderUDF::new_from_impl(
        transform_values::TransformValues::new(),
    ))]
}
