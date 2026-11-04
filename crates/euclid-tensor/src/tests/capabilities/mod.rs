//! Value-level tests for the eight capability extensions, run against [`Dense`].
//!
//! These exist because the capability traits previously had no implementation at all: they
//! type-checked, but nothing demonstrated that the signatures can express the operations they
//! name. Every assertion here is a hand-computed expected value.
//!
//! One module per capability trait.

mod contract;
mod conv;
mod fft;
mod indexing;
mod reduce;
mod sort;
mod transform;
mod window;
