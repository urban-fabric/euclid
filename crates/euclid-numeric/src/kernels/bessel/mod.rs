//! The Bessel family: solutions of `x²y″ + xy′ + (x² − n²)y = 0` and of its modified
//! counterpart `x²y″ + xy′ − (x² + n²)y = 0`.
//!
//! Four modules, one per kind, because the numerics divide exactly along those lines:
//!
//! | | first kind | second kind |
//! |---|---|---|
//! | **ordinary** | [`ordinary_first`] — `J₀`, `J₁` | [`ordinary_second`] — `Y₀`, `Y₁` |
//! | **modified** | [`modified_first`] — `I₀`, `I₁` | [`modified_second`] — `K₀`, `K₁` |
//!
//! The *ordinary* pair both decay like `x^{−1/2}` while oscillating, so neither solution
//! dominates the other and the two are numerically peers. That is what makes marching the
//! recurrence outward in `x` a stable way to reach the middle range, and it is also why
//! relative accuracy cannot be promised near a zero.
//!
//! The *modified* pair are maximally unequal: `I` grows like `e^x` and `K` decays like
//! `e^{−x}`, so every method has to carry the scale explicitly, and the scaled forms are the
//! primitive ones rather than an afterthought. They exist because the unscaled values leave
//! `f64`'s range long before the arguments stop being useful: `I₀(710)` overflows while
//! `I₀ᵉ(710)` is `0.015`, and `K₀(746)` underflows while `K₀ᵉ(746)` is `0.046`.
//!
//! [`asymptotic`] sits outside the table: the large-argument expansion is the *same* series for
//! `J` and `Y`, read through different trigonometric factors, so neither kind owns it.

pub(crate) mod asymptotic;
pub(crate) mod modified_first;
pub(crate) mod modified_second;
pub(crate) mod ordinary_first;
pub(crate) mod ordinary_second;

#[cfg(test)]
mod tests;

pub(crate) use modified_first::{i0, i0e, i1, i1_over_i0, i1e, ln_i0};
pub(crate) use modified_second::{k0, k0e, k1, k1e};
pub(crate) use ordinary_first::{j0, j1};
pub(crate) use ordinary_second::{y0, y1};

/// `ln(x/2)`, as `ln x − ln 2`.
///
/// Shared by the small-argument series of `K₀`, `K₁`, `Y₀` and `Y₁`, all four of which carry a
/// `ln(x/2)` factor.
///
/// Spelled as a difference rather than `(x * 0.5).ln()` because the product form underflows: at
/// the smallest subnormal `x = 5e-324` the halving rounds to exactly zero and `ln` returns
/// `−∞`, where the true `ln(x/2) ≈ −745` is an entirely ordinary number. The subtraction also
/// avoids one rounding.
fn ln_half_x(x: f64) -> f64 {
    x.ln() - core::f64::consts::LN_2
}
