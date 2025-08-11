/*
 * Arrow C Data Interface integration tests
 *.
 * Validates the contents of the ArrowArray buffers that comes
 * from the Rust side via export_to_c FFI.
 *
 * Compile to a static archive - build.rs handles this.
 */
 #include <stdint.h>
 #include <stddef.h>
 #include <string.h>
 #include <stdio.h>
 
/* ----------------------------------------------------------
    Arrow C Data Interface struct declarations
--------------------------------------------------------*/
typedef struct ArrowArray {
    int64_t  length;
    int64_t  null_count;
    int64_t  offset;
    int64_t  n_buffers;
    int64_t  n_children;
    const void       **buffers;     /* [0]=nulls, [1]=offsets, [2]=values */
    struct ArrowArray **children;   
    struct ArrowArray  *dictionary; 
    void (*release)(struct ArrowArray*);
    void *private_data;
} ArrowArray;

 typedef struct ArrowSchema
{
    const char*     format;
    const char*     name;
    const char*     metadata;
    int64_t         flags;
    int64_t         n_children;
    struct ArrowSchema** children;
    struct ArrowSchema* dictionary;
    void          (*release)(struct ArrowSchema*);
    void*           private_data;
} ArrowSchema;

 
 /* ---------- helper ------------------------------------- */
 
 static int bitmap_lsb_is_set(const uint8_t* bm, size_t idx)
 {
     return (bm[idx >> 3] >> (idx & 7)) & 1;
 }
 
 /* --------------------------------------------------------
    Per-type inspection functions.
    They all return 1 on success, 0 on failure.
    -------------------------------------------------------- */
 
// int32: expect [11,22,33]
 int c_arrow_check_i32(const ArrowArray* arr)
 {
     if (!arr || arr->n_buffers != 2 || arr->length != 3) return 0;
     const int32_t* v = (const int32_t*)arr->buffers[1];
     return v && v[0]==11 && v[1]==22 && v[2]==33;
 }
 
// int64: [1001,-42,777]
 int c_arrow_check_i64(const ArrowArray* a)
 {
     if (!a || a->n_buffers!=2 || a->length!=3) return 0;
     const int64_t* v = (const int64_t*)a->buffers[1];
     return v && v[0]==1001 && v[1]==-42 && v[2]==777;
 }
 
// uint32: [1,2,3]
 int c_arrow_check_u32(const ArrowArray* a)
 {
     if (!a || a->n_buffers!=2 || a->length!=3) return 0;
     const uint32_t* v = (const uint32_t*)a->buffers[1];
     return v && v[0]==1 && v[1]==2 && v[2]==3;
 }
 
// float32: [1.5,-2.0,3.25]
 int c_arrow_check_f32(const ArrowArray* a)
 {
     if (!a || a->n_buffers!=2 || a->length!=3) return 0;
     const float* v = (const float*)a->buffers[1];
     return v && v[0]==1.5f && v[1]==-2.0f && v[2]==3.25f;
 }
 
// float64: [0.1,0.2,0.3]
 int c_arrow_check_f64(const ArrowArray* a)
 {
     if (!a || a->n_buffers!=2 || a->length!=3) return 0;
     const double* v = (const double*)a->buffers[1];
     return v && v[0]==0.1 && v[1]==0.2 && v[2]==0.3;
 }
 
// boolean bit-packed: true,false,true -> bitmap 0b00000101
 int c_arrow_check_bool(const ArrowArray* a)
 {
     if (!a || a->n_buffers!=2 || a->length!=3) return 0;
     const uint8_t* data = (const uint8_t*)a->buffers[1];
     return data && data[0]==0x05;
 }
 
// UTF-8 values buffer must equal "foo" "bar" -> "foobar"
int c_arrow_check_str(const ArrowArray* a)
{
    if (!a || a->n_buffers!=3 || a->length!=2) return 0;
    const uint32_t* offs = (const uint32_t*)a->buffers[1];  /* offsets */
    const uint8_t*  vals = (const uint8_t*) a->buffers[2];  /* values  */
    return vals && offs && offs[0]==0 && offs[1]==3 && offs[2]==6
           && memcmp(vals,"foobar",6)==0;
}
 
// int32 with null mask: values [42,null,88], bitmap LSB
 int c_arrow_check_i32_null(const ArrowArray* a)
 {
     if (!a || a->n_buffers!=2 || a->length!=3) return 0;
     const uint8_t* bitmap = (const uint8_t*)a->buffers[0];
     if (!bitmap || bitmap_lsb_is_set(bitmap,0)==0 || bitmap_lsb_is_set(bitmap,1)!=0
         || bitmap_lsb_is_set(bitmap,2)==0) return 0;
     const int32_t* v = (const int32_t*)a->buffers[1];
     return v && v[0]==42 && v[2]==88;
 }
 
// dictionary<u32> with codes [0,1,0] and dict ["A","B"]
int c_arrow_check_dict32(const ArrowArray* a)
{
    if (!a || a->n_buffers != 2 || a->length != 3) return 0;

    // codes buffer [nulls, codes]
    const uint32_t* codes = (const uint32_t*)a->buffers[1];
    if (!codes || codes[0] != 0 || codes[1] != 1 || codes[2] != 0) return 0;

    // dictionary must be present: a->dictionary is a UTF8 array: [nulls, offsets, values]
    const ArrowArray* dict = a->dictionary;
    if (!dict || dict->n_buffers != 3 || dict->length != 2) return 0;

    const uint32_t* offs = (const uint32_t*)dict->buffers[1];
    const uint8_t*  vals = (const uint8_t*) dict->buffers[2];
    if (!offs || !vals) return 0;

    // expect ["A","B"] -> offsets [0,1,2], values "AB"
    if (offs[0] != 0 || offs[1] != 1 || offs[2] != 2) return 0;
    if (!(vals[0] == 'A' && vals[1] == 'B')) return 0;

    return 1;
}

// datetime<i64>: [1,2] 
 int c_arrow_check_dt64(const ArrowArray* a)
 {
     if (!a || a->n_buffers!=2 || a->length!=2) return 0;
     const int64_t* v = (const int64_t*)a->buffers[1];
     return v && v[0]==1 && v[1]==2;
 }
 
 /*
 * Validates the ArrowSchema's name and format fields.
 * Expects name and format to match the provided strings.
 * Returns 1 if both match, 0 otherwise.
 */
int c_arrow_check_schema(const ArrowSchema* schema, const char* expected_name, const char* expected_format)
{
    if (!schema || !schema->name || !schema->format)
        return 0;
    if (strcmp(schema->name, expected_name) != 0)
        return 0;
    if (strcmp(schema->format, expected_format) != 0)
        return 0;
    return 1;
}
