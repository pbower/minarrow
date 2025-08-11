//! 64-byte-aligned allocator for AVX-512 / Arrow buffers.
//!
//! Integrates with `Vec64<T>` and overrides allocations
//! to ensure 64-byte alignment, consistent
//! with the Apache Arrow specification and for compatibility
//! with SIMD processing (e.g. Intel AVX-512).
//!
//! See: https://arrow.apache.org/docs/format/Columnar.html
use core::alloc::{AllocError, Allocator, Layout};
use core::ptr::NonNull;
use std::alloc::dealloc;
use std::ptr::slice_from_raw_parts_mut;

const ALIGN_64: usize = 64;

/// Global, zero-sized allocator that enforces 64-bit alignment.
///
/// Hooks into Vec64<T>. Behind the scenes, Vec64<T, Vec64>:
///
/// - Calls Vec64::allocate() when allocating new memory.
/// - Calls Vec64::grow() when reallocation is needed.
/// - Calls Vec64::deallocate() when the vector is dropped. etc,
/// ensuring these calls go through the alignment-enforcing logic.
///
/// Guarantees starting pointer alignment for all allocations it manages
/// including allocations due to growth, mutation, extension,
/// and insertion—in all scenarios except for zero-sized types (ZSTs)
/// and capacity 0.
///
/// ### Padding
/// This allocator does ***not* pad data automatically** - it's here to ensure
/// starting alignment for the memory allocation. When 'flatbuffering'
/// multiple buffers, e.g., over the network, or as part of framed payloads,
/// that you later plan to "steal", for zero-copy memory access, keep in mind
/// that manual padding may be required to ensure all relevant sub-elements **also**
/// start on a 64-byte boundary.
#[derive(Copy, Clone, Default, Debug)]
pub struct Alloc64;

/// Ensures the layout alignment is at least 64 bytes.
#[inline]
fn align_layout(mut layout: Layout) -> Layout {
    // Never reduce alignment; only bump to ≥64.
    if layout.align() < ALIGN_64 {
        layout = Layout::from_size_align(layout.size(), ALIGN_64).expect("Invalid 64-bit layout");
    }
    layout
}

unsafe impl Allocator for Alloc64 {
    /// Allocates memory with at least 64-byte alignment.
    #[inline]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let layout = align_layout(layout);
        let ptr = unsafe { std::alloc::alloc(layout) }; // *mut u8
        NonNull::new(ptr)
            .map(|nn| {
                // SAFETY: slice len = layout.size()
                unsafe {
                    NonNull::new_unchecked(slice_from_raw_parts_mut(nn.as_ptr(), layout.size()))
                }
            })
            .ok_or(AllocError)
    }

    /// Allocates zero-initialised memory with 64-byte alignment.
    #[inline]
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let layout = align_layout(layout);
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        NonNull::new(ptr)
            .map(|nn| unsafe {
                NonNull::new_unchecked(slice_from_raw_parts_mut(nn.as_ptr(), layout.size()))
            })
            .ok_or(AllocError)
    }

    /// Deallocates memory with alignment correction.
    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { dealloc(ptr.as_ptr(), align_layout(layout)) };
    }

    /// Grows an existing allocation while preserving 64-byte alignment.
    #[inline]
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout
    ) -> Result<NonNull<[u8]>, AllocError> {
        let new = align_layout(new);
        let raw = unsafe { std::alloc::realloc(ptr.as_ptr(), align_layout(old), new.size()) };
        NonNull::new(raw)
            .map(|nn| unsafe {
                NonNull::new_unchecked(slice_from_raw_parts_mut(nn.as_ptr(), new.size()))
            })
            .ok_or(AllocError)
    }

    /// Shrinks an allocation while preserving 64-byte alignment.
    #[inline]
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout
    ) -> Result<NonNull<[u8]>, AllocError> {
        let new = align_layout(new);
        let raw = unsafe { std::alloc::realloc(ptr.as_ptr(), align_layout(old), new.size()) };
        NonNull::new(raw)
            .map(|nn| unsafe {
                NonNull::new_unchecked(slice_from_raw_parts_mut(nn.as_ptr(), new.size()))
            })
            .ok_or(AllocError)
    }

    /// Grows the allocation to a new layout and zero-initialises any newly allocated region.
    /// Existing data is preserved; the new memory is zeroed.
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout
    ) -> Result<NonNull<[u8]>, AllocError> {
        std::debug_assert!(new.size() >= old.size());

        // Allocate new zero-filled block
        let new_block = self.allocate_zeroed(new)?;

        // Copy old bytes
        unsafe { core::ptr::copy_nonoverlapping(ptr.as_ptr(), new_block.as_mut_ptr(), old.size()) };

        // Free old block
        unsafe { self.deallocate(ptr, old) };

        Ok(new_block)
    }

    /// Returns a reference to the allocator. Useful for adapter traits and APIs requiring allocator references.
    fn by_ref(&self) -> &Self
    where
        Self: Sized
    {
        self
    }
}

#[cfg(test)]
mod alloc64_tests {
    use std::alloc::{Allocator, Layout};

    use super::*;

    #[test]
    fn test_allocate_and_deallocate_alignment() {
        let layout = Layout::from_size_align(4096, 1).unwrap();
        let a = Alloc64;
        let ptr = a.allocate(layout).expect("allocate failed");
        let addr = ptr.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(addr % 64, 0);
        // SAFETY: valid ptr/layout
        unsafe {
            a.deallocate(ptr.as_non_null_ptr(), layout);
        }
    }

    #[test]
    fn test_allocate_zeroed() {
        let layout = Layout::from_size_align(32, 1).unwrap();
        let a = Alloc64;
        let ptr = a.allocate_zeroed(layout).expect("allocate_zeroed failed");
        let addr = ptr.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(addr % 64, 0);
        let data =
            unsafe { std::slice::from_raw_parts(ptr.as_non_null_ptr().as_ptr(), layout.size()) };
        assert!(data.iter().all(|&b| b == 0));
        // SAFETY: valid ptr/layout
        unsafe {
            a.deallocate(ptr.as_non_null_ptr(), layout);
        }
    }

    #[test]
    fn test_grow_and_shrink() {
        let layout = Layout::from_size_align(64, 1).unwrap();
        let a = Alloc64;
        let ptr = a.allocate(layout).expect("allocate failed");

        let big = Layout::from_size_align(256, 1).unwrap();
        let grown = unsafe { a.grow(ptr.as_non_null_ptr(), layout, big).expect("grow failed") };
        let addr = grown.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(addr % 64, 0);

        let shrunk =
            unsafe { a.shrink(grown.as_non_null_ptr(), big, layout).expect("shrink failed") };
        let addr2 = shrunk.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(addr2 % 64, 0);

        unsafe {
            a.deallocate(shrunk.as_non_null_ptr(), layout);
        }
    }

    #[test]
    fn test_grow_zeroed() {
        let layout = Layout::from_size_align(16, 1).unwrap();
        let a = Alloc64;
        let ptr = a.allocate(layout).expect("allocate failed");

        let bigger = Layout::from_size_align(128, 1).unwrap();
        let grown = unsafe {
            a.grow_zeroed(ptr.as_non_null_ptr(), layout, bigger).expect("grow_zeroed failed")
        };
        let addr = grown.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(addr % 64, 0);
        // Check new region is zeroed
        let data =
            unsafe { std::slice::from_raw_parts(grown.as_non_null_ptr().as_ptr(), bigger.size()) };
        assert!(data[16..].iter().all(|&b| b == 0));
        unsafe {
            a.deallocate(grown.as_non_null_ptr(), bigger);
        }
    }

    #[test]
    fn test_by_ref() {
        let a = Alloc64;
        let b = a.by_ref();
        assert!(std::ptr::eq(&a, b));
    }
    #[test]
    fn test_allocator_produces_64_alignment() {
        let a = Alloc64;
        for size in [1, 7, 32, 64, 256, 4096] {
            let layout = Layout::from_size_align(size, 1).unwrap();
            let ptr = a.allocate(layout).unwrap();
            let addr = ptr.as_non_null_ptr().as_ptr() as usize;
            assert_eq!(addr % 64, 0, "Pointer {:#x} not 64-byte aligned for size {}", addr, size);
            unsafe {
                a.deallocate(ptr.as_non_null_ptr(), layout);
            }
        }
    }

    #[test]
    fn test_allocator_zeroed_alignment() {
        let a = Alloc64;
        let layout = Layout::from_size_align(128, 1).unwrap();
        let ptr = a.allocate_zeroed(layout).unwrap();
        let addr = ptr.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(addr % 64, 0, "Pointer {:#x} not 64-byte aligned", addr);
        unsafe {
            a.deallocate(ptr.as_non_null_ptr(), layout);
        }
    }

    #[test]
    fn test_grow_and_shrink_alignment() {
        let a = Alloc64;
        let small = Layout::from_size_align(64, 1).unwrap();
        let big = Layout::from_size_align(512, 1).unwrap();
        let ptr = a.allocate(small).unwrap();
        let grown = unsafe { a.grow(ptr.as_non_null_ptr(), small, big).unwrap() };
        let addr = grown.as_non_null_ptr().as_ptr() as usize;
        assert_eq!(addr % 64, 0, "Grown pointer {:#x} not 64-byte aligned", addr);
        unsafe {
            a.deallocate(grown.as_non_null_ptr(), big);
        }
    }
}
