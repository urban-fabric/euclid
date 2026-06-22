//! Concrete carrier implementations of the numeric traits, kept apart from their definitions
//! (the `real.rs` / `real_impl` split, applied to every tier). The trait *definitions* live in
//! the parent [`numeric`](crate::numeric) module; this folder holds the `f64`/`f32`/`i*`/`u*`/
//! `bool` `impl` blocks. `real` is split per carrier (`real_f64`/`real_f32`, each large); the
//! smaller tiers are grouped (`integer_impl`/`natural_impl`/`boolean_impl`). These modules
//! export nothing — they exist for their `impl` blocks' side effects.

mod boolean_impl;
mod integer_impl;
mod natural_impl;
mod real_f32;
mod real_f64;
