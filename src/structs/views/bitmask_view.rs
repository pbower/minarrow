//! # BitmaskV Module
//!
//! `BitmaskV` is a **logical, zero-copy, read-only window** into a contiguous
//! region of a [`Bitmask`].
//!
//! ## Purpose
//! - **Indexable** and **bounds-checked** access to a subset of a bit-packed mask.
//! - All logical indices are **relative to the window**.
//! - Avoids copying — shares the parent mask buffer via `Arc`.
//!
//! ## Behaviour
//! - All operations remap indices internally to the correct positions in the parent mask.
//! - Window slicing (`slice`) is O(1) — pointer and metadata updates only.
//! - Arc Clones cheaply.
//! - Use [`to_bitmask`](BitmaskV::to_bitmask) for a materialised copy of the view.
//!
//! ## Threading
//! - Thread-safe by virtue of immutability — no interior mutability.
//!
//! ## Performance Notes
//! - Before introducing a `BitmaskV`, consider whether simply cloning the [`Bitmask`]
//!   would be sufficient, since cloning is already extremely cheap.
//!
//! ## Related
//! - [`Bitmask`] — the full mask structure this views into.
//! - [`BitmaskVT`] — the tuple form returned by [`as_tuple`](BitmaskV::as_tuple).

use std::fmt::{self, Debug, Display, Formatter};
use std::ops::Index;
use std::sync::Arc;

use crate::traits::print::MAX_PREVIEW;
use crate::{Bitmask, BitmaskVT};

/// # BitmaskView
///
/// Zero-copy, bounds-checked window over a [`Bitmask`].
///
/// ## Fields
/// - `bitmask`: backing [`Bitmask`] (shared via `Arc`).
/// - `offset`: start bit position in the parent mask.
/// - `len`: number of bits in the view.
///
/// ## Behaviour
/// - All indexing is **relative** to the view's start.
/// - All accesses are in-bounds (panics if violated).
/// - No allocation or buffer copying occurs when creating or slicing views.
///
/// ## Example
/// ```rust
/// use minarrow::Bitmask;
/// use minarrow::BitmaskV;
///
/// let mask = Bitmask::from_bools(&[true, false, true, true, false]);
/// let view = BitmaskV::new(mask, 1, 3); // window: false, true, true
///
/// assert_eq!(view.len(), 3);
/// assert!(!view.get(0));
/// assert!(view.get(1));
/// assert!(view.get(2));
/// ```
#[derive(Clone, PartialEq)]
pub struct BitmaskV {
    pub bitmask: Arc<Bitmask>,
    pub offset: usize,
    len: usize
}

impl BitmaskV {
    /// Construct a view over `bitmask[offset..offset+len)`.
    #[inline]
    pub fn new(bitmask: Bitmask, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= bitmask.len(),
            "BitmaskView: out of bounds (offset + len = {}, bitmask.len = {})",
            offset + len,
            bitmask.len()
        );
        Self { bitmask: bitmask.into(), offset, len }
    }

    /// Returns the length (number of bits) in the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the value at logical index `i` within the view.
    #[inline]
    pub fn get(&self, i: usize) -> bool {
        assert!(i < self.len, "BitmaskView: index {i} out of bounds for window len {}", self.len);
        self.bitmask.get(self.offset + i)
    }

    /// Returns a slice of the bitmask’s bytes
    /// Due to the booleans being bitpacked in a u8,
    /// the slice retains:\
    /// *Pos 0*: **Buffer**: The bitpacked u8 buffer.\
    /// *Pos 1*: **Offset**: Bit offset indicating where it starts in that byte.\
    /// *Pos 2*: **Length**: Logical length in bits of the slice
    #[inline]
    pub fn as_bytes_window(&self) -> (&[u8], usize, usize) {
        self.bitmask.slice(self.offset, self.len)
    }

    /// Returns a Bitmask copy of the view.
    #[inline]
    pub fn to_bitmask(&self) -> Bitmask {
        self.bitmask.slice_clone(self.offset, self.len)
    }

    /// Returns an iterator over all set bits (indices relative to the window).
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len).filter(move |&i| self.get(i))
    }

    /// Returns an iterator over all cleared bits (indices relative to the window).
    pub fn iter_cleared(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.len).filter(move |&i| !self.get(i))
    }

    /// Counts number of set bits in the view.
    pub fn count_ones(&self) -> usize {
        self.iter_set().count()
    }

    /// Counts number of cleared bits in the view.
    pub fn count_zeros(&self) -> usize {
        self.iter_cleared().count()
    }

    /// Returns true if all bits in the view are set.
    #[inline]
    pub fn all_set(&self) -> bool {
        self.count_ones() == self.len
    }

    /// Returns true if all bits in the view are cleared.
    #[inline]
    pub fn all_unset(&self) -> bool {
        self.count_zeros() == self.len
    }

    /// Returns true if any bit in the view is cleared.
    #[inline]
    pub fn has_cleared(&self) -> bool {
        !self.all_set()
    }

    /// Returns true if any bit in the view is set.
    #[inline]
    pub fn any_set(&self) -> bool {
        self.count_ones() > 0
    }

    /// Slices the view further by logical offset and len (relative to this window).
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(offset + len <= self.len, "BitmaskView::slice: out of bounds");
        Self {
            bitmask: self.bitmask.clone(),
            offset: self.offset + offset,
            len
        }
    }

    /// Returns the exclusive end row index of the window (relative to bitmask).
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.len
    }

    /// Returns the underlying window as a tuple: (&Bitmask, offset, len).
    #[inline]
    pub fn as_tuple(&self) -> BitmaskVT {
        (&self.bitmask, self.offset, self.len)
    }
}

impl Index<usize> for BitmaskV {
    type Output = bool;

    /// Returns a reference to a constant `true` or `false` value depending on the bit at `index`.
    ///
    /// Note: This does **not** return a reference into the underlying bitmask storage.
    /// The reference points to a compiler-promoted static constant, so its address is
    /// unrelated to the internal buffer. Use [`Self::get`] if you need the value directly.
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        static TRUE_CONST: bool = true;
        static FALSE_CONST: bool = false;
        if self.get(index) {
            &TRUE_CONST
        } else {
            &FALSE_CONST
        }
    }
}

impl Debug for BitmaskV {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("BitmaskView")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("set_count", &self.count_ones())
            .field("unset_count", &self.count_zeros())
            .finish()
    }
}

impl Display for BitmaskV {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let len = self.len;
        let offset = self.offset;
        let set = self.count_ones();
        let unset = len - set;

        writeln!(
            f,
            "BitmaskView [{} bits] (offset: {}, set: {}, unset: {})",
            len, offset, set, unset
        )?;

        let limit = len.min(MAX_PREVIEW);
        write!(f, "  ")?;
        for i in 0..limit {
            let symbol = if self.get(i) { '1' } else { '.' };
            write!(f, "{symbol}")?;
        }
        if len > MAX_PREVIEW {
            write!(f, "... ({} more bits)", len - MAX_PREVIEW)?;
        }
        writeln!(f)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Bitmask;

    #[test]
    fn test_bitmask_view_basic_access() {
        // 8 bits: 1 0 1 0 1 0 1 0
        let bits = [true, false, true, false, true, false, true, false];
        let mask = Bitmask::from_bools(&bits);

        let view = BitmaskV::new(mask, 0, 8);
        assert_eq!(view.len(), 8);
        for i in 0..8 {
            assert_eq!(view.get(i), bits[i]);
            assert_eq!(view[i], bits[i]);
        }
        assert_eq!(view.count_ones(), 4);
        assert_eq!(view.count_zeros(), 4);
        assert!(!view.is_empty());
        assert!(!view.all_set());
        assert!(!view.all_unset());
        assert!(view.any_set());
        assert!(view.has_cleared());
        assert_eq!(view.end(), 8);

        // All iter_set/iter_cleared
        let set_indices: Vec<_> = view.iter_set().collect();
        assert_eq!(set_indices, vec![0, 2, 4, 6]);
        let unset_indices: Vec<_> = view.iter_cleared().collect();
        assert_eq!(unset_indices, vec![1, 3, 5, 7]);
    }

    #[test]
    fn test_bitmask_view_offset_and_window() {
        // bits: 1 1 0 0 1 1 0 0
        let bits = [true, true, false, false, true, true, false, false];
        let mask = Bitmask::from_bools(&bits);

        // view over [2..6): 0 0 1 1
        let view = BitmaskV::new(mask, 2, 4);
        assert_eq!(view.len(), 4);
        assert_eq!((0..4).map(|i| view.get(i)).collect::<Vec<_>>(), vec![false, false, true, true]);
        assert_eq!(view.count_ones(), 2);
        assert_eq!(view.count_zeros(), 2);

        // view.as_bytes_window returns the correct window
        let (bytes, bit_offset, len) = view.as_bytes_window();
        assert!(len == 4 && bit_offset < 8 && !bytes.is_empty());
    }

    #[test]
    fn test_bitmask_view_slice_and_to_bitmask() {
        // 10 bits: 1 1 1 0 0 0 1 0 1 1
        let bits = [true, true, true, false, false, false, true, false, true, true];
        let mask = Bitmask::from_bools(&bits);

        let view = BitmaskV::new(mask, 2, 6); // [2..8): 1 0 0 0 1 0
        assert_eq!(view.len(), 6);
        assert_eq!(view.get(0), true);
        assert_eq!(view.get(1), false);
        assert_eq!(view.get(5), false);

        let subview = view.slice(2, 3); // [4..7): 0 0 1
        assert_eq!(subview.len(), 3);
        assert_eq!(subview.get(0), false);
        assert_eq!(subview.get(1), false);
        assert_eq!(subview.get(2), true);

        let new_mask = subview.to_bitmask();
        let expected = Bitmask::from_bools(&[false, false, true]);
        assert_eq!(new_mask, expected);
    }

    #[test]
    fn test_bitmask_view_empty_and_single_bit() {
        let mask = Bitmask::from_bools(&[]);
        let view = BitmaskV::new(mask, 0, 0);
        assert_eq!(view.len(), 0);
        assert!(view.is_empty());
        assert!(view.all_set()); // vacuously true
        assert!(view.all_unset()); // vacuously true

        let mask2 = Bitmask::from_bools(&[true]);
        let view2 = BitmaskV::new(mask2, 0, 1);
        assert_eq!(view2.len(), 1);
        assert!(!view2.is_empty());
        assert_eq!(view2.get(0), true);
        assert!(view2.all_set());
        assert!(!view2.all_unset());
        assert!(!view2.has_cleared());
        assert!(view2.any_set());
    }

    #[test]
    #[should_panic(expected = "BitmaskView: out of bounds")]
    fn test_bitmask_view_out_of_bounds() {
        let mask = Bitmask::from_bools(&[true, false, true]);
        let _ = BitmaskV::new(mask, 2, 2); // Exceeds mask length (should panic)
    }

    #[test]
    #[should_panic(expected = "BitmaskView: index 3 out of bounds")]
    fn test_bitmask_view_get_oob() {
        let mask = Bitmask::from_bools(&[true, false, true, true]);
        let view = BitmaskV::new(mask, 1, 2);
        let _ = view.get(3); // out-of-window
    }

    #[test]
    fn test_bitmask_view_debug() {
        let bits = [true, false, true, false];
        let mask = Bitmask::from_bools(&bits);
        let view = BitmaskV::new(mask, 0, 4);
        let dbg = format!("{:?}", view);
        assert!(dbg.contains("BitmaskView"));
        assert!(dbg.contains("offset"));
        assert!(dbg.contains("set_count"));
    }
}
