//! # **Internal module**
//!
//! Promotable vector types for SharedBuffer lazy heap allocation.
//!
//! Provides reference-counted wrappers that enable SharedBuffer to promote
//! stack or foreign buffers to heap-allocated, shareable storage on first clone.
//! Supports both standard Vec<u8> and SIMD-aligned Vec64<u8> variants.

use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use crate::Vec64;
use crate::structs::shared_buffer::SharedBuffer;
use crate::structs::shared_buffer::internal::vtable::{PROMO_EVEN_VT, PROMO64_EVEN_VT};

/// Reference-counted, heap-allocated buffer for SharedBuffer promotion.
///
/// Wraps Vec<T> or Vec64<T> with atomic reference counting to enable
/// safe sharing after promotion from stack/foreign memory.
#[repr(C)]
pub(crate) struct PromotableVec<T> {
    pub(crate) ref_cnt: AtomicUsize,
    pub(crate) inner: T,
}

/// Checks if promotable buffer has exclusive ownership (refcount == 1).
#[inline]
pub(crate) fn promo_is_unique<T>(h: &AtomicPtr<()>) -> bool {
    let raw = h.load(Ordering::Acquire);
    if raw.is_null() {
        true
    } else {
        unsafe {
            (*(raw as *const PromotableVec<T>))
                .ref_cnt
                .load(Ordering::Acquire)
                == 1
        }
    }
}

// Vec<u8> promotion functions

/// Clones promotable Vec<u8> buffer, promoting to heap on first clone.
pub(crate) unsafe fn promo_clone(h: &AtomicPtr<()>, ptr: *const u8, len: usize) -> SharedBuffer {
    let raw = h.load(Ordering::Acquire);
    if raw.is_null() {
        // Promote stack/foreign buffer to heap and vtable
        let promoted = Box::into_raw(Box::new(PromotableVec::<Vec<u8>> {
            ref_cnt: AtomicUsize::new(1),
            inner: unsafe { Vec::from_raw_parts(ptr as *mut u8, len, len) },
        }));
        h.store(promoted.cast(), Ordering::Release);
        return SharedBuffer {
            ptr,
            len,
            data: AtomicPtr::new(promoted.cast()),
            vtable: &PROMO_EVEN_VT,
        };
    }
    let header = unsafe { &*(raw as *const PromotableVec<Vec<u8>>) };
    header.ref_cnt.fetch_add(1, Ordering::Relaxed);
    SharedBuffer {
        ptr,
        len,
        data: AtomicPtr::new(raw),
        vtable: &PROMO_EVEN_VT,
    }
}

/// Decrements reference count, deallocating if last reference.
pub(crate) unsafe fn promo_drop(h: &mut AtomicPtr<()>, _p: *const u8, _l: usize) {
    let raw = h.load(Ordering::Acquire);
    if raw.is_null() {
        return;
    }
    let header = unsafe { &*(raw as *const PromotableVec<Vec<u8>>) };
    if header.ref_cnt.fetch_sub(1, Ordering::AcqRel) == 1 {
        drop(unsafe { Box::from_raw(raw as *mut PromotableVec<Vec<u8>>) });
    }
}

// Vec64<u8> promotion functions

/// Clones promotable Vec64<u8> buffer, promoting to heap on first clone.
pub(crate) unsafe fn promo64_clone(h: &AtomicPtr<()>, ptr: *const u8, len: usize) -> SharedBuffer {
    let raw = h.load(Ordering::Acquire);
    if raw.is_null() {
        // Promote stack/foreign buffer to heap and vtable
        let promoted = Box::into_raw(Box::new(PromotableVec::<Vec64<u8>> {
            ref_cnt: AtomicUsize::new(1),
            inner: unsafe { Vec64::from_raw_parts(ptr as *mut u8, len, len) },
        }));
        h.store(promoted.cast(), Ordering::Release);
        return SharedBuffer {
            ptr,
            len,
            data: AtomicPtr::new(promoted.cast()),
            vtable: &PROMO64_EVEN_VT,
        };
    }
    let header = unsafe { &*(raw as *const PromotableVec<Vec64<u8>>) };
    header.ref_cnt.fetch_add(1, Ordering::Relaxed);
    SharedBuffer {
        ptr,
        len,
        data: AtomicPtr::new(raw),
        vtable: &PROMO64_EVEN_VT,
    }
}

/// Decrements reference count, deallocating if last reference.
pub(crate) unsafe fn promo64_drop(h: &mut AtomicPtr<()>, _p: *const u8, _l: usize) {
    let raw = h.load(Ordering::Acquire);
    if raw.is_null() {
        return;
    }
    let header = unsafe { &*(raw as *const PromotableVec<Vec64<u8>>) };
    if header.ref_cnt.fetch_sub(1, Ordering::AcqRel) == 1 {
        drop(unsafe { Box::from_raw(raw as *mut PromotableVec<Vec64<u8>>) });
    }
}
