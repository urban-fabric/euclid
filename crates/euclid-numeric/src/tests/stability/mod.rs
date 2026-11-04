//! **Does it survive the inputs where the naive formula fails?**
//!
//! Each test here pairs euclid's function against the obvious hand-rolled expression it
//! replaces, at arguments chosen so the naive form *demonstrably* breaks — returning `NaN`,
//! `±inf`, exactly `0`, or losing most of its significant digits. The naive failure is
//! asserted too, so these tests document *why* the dedicated method has to exist rather than
//! just checking that it happens to work.
//!
//! This is the suite that justifies `ln_1p`, `exp_m1`, `ln_1p_exp`, `ln_1m_exp`, `erfcx`,
//! `xlogy`, `ln_add_exp`, `hypot`, and the scaled Bessel forms being named operations instead
//! of caller-side formulas.
//!
//! Split by failure mode.

mod cancellation;
mod classical;
mod extremes;
mod saturation;
mod tails;
