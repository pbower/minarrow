// Copyright Peter Bower 2025. All Rights Reserved.
// Licensed under Mozilla Public License (MPL) 2.0.

//! Contains basic numeric kernel operators for matching and routing purposes

/// Arithmetic operators for numeric computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticOperator {
    /// Addition (`lhs + rhs`)
    Add,
    /// Subtraction (`lhs - rhs`)
    Subtract,
    /// Multiplication (`lhs * rhs`)
    Multiply,
    /// Division (`lhs / rhs`)
    ///
    /// For integers, division by zero panics in dense arrays and nullifies in masked arrays.
    /// For floating-point, follows IEEE 754 (yields ±Inf or NaN).
    Divide,
    /// Modulus/remainder operation (`lhs % rhs`)
    ///
    /// Behaviour matches Rust's `%` operator. Division by zero handling follows same
    /// rules as `Divide` operation.
    Remainder,
    /// Exponentiation (`lhs ^ rhs`)
    ///
    /// For integers, uses repeated multiplication. For floating-point, uses `pow()` function.
    /// Negative exponents on integers may yield zero due to truncation.
    Power,
}

/// Comparison operators for binary predicates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOperator {
    /// Equality comparison (`lhs == rhs`)
    Equals,
    /// Inequality comparison (`lhs != rhs`)
    NotEquals,
    /// Less-than comparison (`lhs < rhs`)
    LessThan,
    /// Less-than-or-equal comparison (`lhs <= rhs`)
    LessThanOrEqualTo,
    /// Greater-than comparison (`lhs > rhs`)
    GreaterThan,
    /// Greater-than-or-equal comparison (`lhs >= rhs`)
    GreaterThanOrEqualTo,
    /// Tests if value is null (`lhs IS NULL`)
    ///
    /// Always returns a valid boolean, never null.
    IsNull,
    /// Tests if value is not null (`lhs IS NOT NULL`)
    ///
    /// Always returns a valid boolean, never null.
    IsNotNull,
    /// Range membership test (`lhs BETWEEN min AND max`)
    ///
    /// Equivalent to `lhs >= min AND lhs <= max` with appropriate null handling.
    Between,
    /// Set membership test (`lhs IN (set)`)
    ///
    /// Returns true if lhs matches any value in the provided set.
    In,
    /// Set exclusion test (`lhs NOT IN (set)`)
    ///
    /// Returns true if lhs doesn't match any value in the provided set.
    NotIn,
}

/// Logical/boolean operators for conditional expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalOperator {
    /// Logical AND (`lhs AND rhs`)
    ///
    /// Returns false if either operand is false, otherwise propagates nulls.
    And,
    /// Logical OR (`lhs OR rhs`)
    ///
    /// Returns true if either operand is true, otherwise propagates nulls.
    Or,
    /// Logical XOR (`lhs XOR rhs`)
    ///
    /// Returns true if operands differ, false if same, null if either is null.
    Xor,
}

/// Unary operators for single-operand transformations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOperator {
    /// Arithmetic negation (`-operand`)
    ///
    /// Negates numeric values. For unsigned integers, uses wrapping negation.
    Negative,
    /// Logical/bitwise NOT (`!operand` or `~operand`)
    ///
    /// For booleans: logical NOT. For integers: bitwise complement.
    Not,
    /// Unary plus (`+operand`)
    ///
    /// Identity operation that explicitly indicates positive values.
    /// Primarily used for symmetry with negation operator.
    Positive,
}
