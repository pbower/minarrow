//! # **Internal module**
//!
//! Static vtable implementations for SharedBuffer backends.
//!
//! Provides function tables for different buffer ownership models:
//! - `OWNED_VT`: Reference-counted arbitrary backing types (Vec, Vec64, custom containers)
//! - `STATIC_VT`: Read-only static/const data (zero refcounting)
//! - `PROMO_*_VT`: Lazy promotion from stack to heap allocation
//!
//! Comparable to Tokio's `Bytes` implementation but avoids external dependencies
//! and ensures 64-byte SIMD alignment support.

use core::slice;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use crate::Vec64;
use crate::structs::shared_buffer::SharedBuffer;
use crate::structs::shared_buffer::internal::vtable::Vtable;

/// Reference-counted wrapper for arbitrary backing storage types.
///
/// Enables SharedBuffer to manage any container implementing AsRef<[u8]>
/// with atomic reference counting for safe sharing.
#[repr(C)]
pub(crate) struct Owned<T: AsRef<[u8]> + Send + Sync + 'static> {
    pub(crate) ref_cnt: AtomicUsize,
    pub(crate) owner: T,
}

/// Clones owned buffer by incrementing reference count.
unsafe fn owned_clone(h: &AtomicPtr<()>, p: *const u8, l: usize) -> SharedBuffer {
    let raw = h.load(Ordering::Acquire);
    assert!(!raw.is_null());
    // SAFETY: ref_cnt is first field, layout #[repr(C)]
    let ref_cnt = unsafe { &*(raw as *const AtomicUsize) };
    ref_cnt.fetch_add(1, Ordering::Relaxed);
    SharedBuffer {
        ptr: p,
        len: l,
        data: AtomicPtr::new(raw),
        vtable: &OWNED_VT,
    }
}

/// Decrements reference count, deallocating if last reference.
unsafe fn owned_drop(h: &mut AtomicPtr<()>, _p: *const u8, _l: usize) {
    let raw = h.load(Ordering::Acquire);
    if raw.is_null() {
        return;
    }
    let ref_cnt = unsafe { &*(raw as *const AtomicUsize) };
    if ref_cnt.fetch_sub(1, Ordering::AcqRel) == 1 {
        drop(unsafe { Box::from_raw(raw) });
    }
}

/// Vtable for reference-counted owned buffers (Vec, Vec64, custom containers).
pub(crate) static OWNED_VT: Vtable = Vtable {
    clone: owned_clone,
    drop: owned_drop,
    is_unique: |h| {
        let raw = h.load(Ordering::Acquire);
        if raw.is_null() {
            return false;
        }
        let ref_cnt = unsafe { &*(raw as *const AtomicUsize) };
        ref_cnt.load(Ordering::Acquire) == 1
    },
    to_vec: |_, p, l| unsafe { slice::from_raw_parts(p, l) }.to_vec(),
    to_vec64: |_, p, l| {
        let mut v = Vec64::with_capacity(l);
        unsafe {
            v.extend_from_slice(slice::from_raw_parts(p, l));
        }
        v
    },
    #[cfg(all(target_os = "linux", feature = "memfd"))]
    memfd_fd: |_| None,
};
