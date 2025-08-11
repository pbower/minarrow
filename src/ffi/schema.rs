use std::collections::BTreeMap;

use crate::Field;

/// Schema struct supporting `RecordBatch` construction for Arrow FFI compatibility only.
///
/// In `Minarrow`, prefer `FieldArray` for typical use, as it holds `Field` directly.
///
/// This type resides in the FFI module. A dedicated schema abstraction is otherwise
/// not utilised within this crate, as the same field definitions are embedded
/// within the `Table` structure, to avoid layered indirection.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Schema {
    pub fields: Vec<Field>,
    pub metadata: BTreeMap<String, String>
}

impl Schema {
    #[inline]
    pub fn new(fields: Vec<Field>, metadata: BTreeMap<String, String>) -> Self {
        Self { fields: fields, metadata }
    }
}

impl From<Vec<Field>> for Schema {
    fn from(fields: Vec<Field>) -> Self {
        Self { fields, ..Default::default() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::arrow_dtype::ArrowType;

    #[test]
    fn test_schema_new_and_from_fields() {
        let f1 = Field::new("c1", ArrowType::Int32, false, None);
        let f2 = Field::new("c2", ArrowType::String, true, None);

        let meta = std::collections::BTreeMap::new();
        let schema = Schema::new(vec![f1.clone(), f2.clone()], meta.clone());
        assert_eq!(schema.fields[0], f1);
        assert_eq!(schema.fields[1], f2);

        let schema2: Schema = vec![f1.clone(), f2.clone()].into();
        assert_eq!(schema2.fields[0], f1);
        assert!(schema2.metadata.is_empty());
    }
}
