use std::{any::Any, sync::Arc};

/// Trait for any object that can be stored in `enums::Value::Custom`.
///
/// `CustomValue` extends *MinArrow's* `Value` universe, allowing engines or 
/// analytics to handle intermediate states and custom types 
/// within the same pipeline abstraction as scalars, arrays, and tables.
///
/// You must then manage downcasting on top of the base enum match, so it 
/// it's not the most ergonomic situation, but is available.
/// 
/// Typical use cases include:
/// - Accumulators, partial aggregates, or sketches.
/// - Custom algorithm outputs.
/// - Arbitrary user-defined types requiring unified pipeline integration.
///
/// **Dynamic dispatch and downcasting** are used at runtime to recover the inner type 
/// and perform type-specific logic, such as merging, reduction, or finalisation.
///
/// ### Implementation Notes:
/// - **Manual implementation is not required**.  
/// - Any type that implements `Debug`, `Clone`, `PartialEq`, and is `Send + Sync + 'static` 
///   automatically satisfies `CustomValue` via the blanket impl.
/// - `Any` is automatically implemented by Rust for all `'static` types.  
///
/// ### Borrowing Constraints:
/// - **Borrowed types cannot be used in `Value::Custom` directly**, since `Any` requires `'static`.  
/// - To store borrowed data, first promote it to an owned type or wrap it in `Arc`.
pub trait CustomValue: Any + Send + Sync + std::fmt::Debug {
    
    /// Downcasts the type as `Any`
    fn as_any(&self) -> &dyn Any;
    /// Returns a deep clone of the object.
    /// 
    /// Additionally, the `Value` enum automatically derives `Clone`, which is a 
    /// shallow `Arc` clone by default.
    fn deep_clone(&self) -> Arc<dyn CustomValue>;

    /// Performs semantic equality on the boxed object.
    ///
    /// This enables `PartialEq` to be implemented for `Value`,
    /// since `dyn CustomValue` cannot use `==` directly.
    fn eq_box(&self, other: &dyn CustomValue) -> bool;
}

/// Provided extension types implement `Clone`, `PartialEq`, `Debug`
/// and is `Send` + `Sync + Any`, these methods implement automatically.
impl<T> CustomValue for T
where
    T: Any + Send + Sync + Clone + PartialEq + std::fmt::Debug,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn deep_clone(&self) -> Arc<dyn CustomValue> {
        Arc::new(self.clone())
    }

    fn eq_box(&self, other: &dyn CustomValue) -> bool {
        other.as_any().downcast_ref::<T>().map_or(false, |o| self == o)
    }
}
