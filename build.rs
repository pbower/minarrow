fn main() {
    // C FFI Integration tests
    #[cfg(feature = "c_ffi_tests")]
    cc::Build::new().file("tests/c_inspect_arrow.c").compile("cinspect_arrow");

    #[cfg(feature = "c_ffi_tests")]
    println!("cargo:rerun-if-changed=tests/c_inspect_arrow.c");
}
