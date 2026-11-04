//! **How do the functions behave where they are singular, zero, or out of range?**
//!
//! The other suites ask whether a function is right where it is well behaved. This one goes
//! looking for the arguments that break implementations: the poles, the zeros, the domain
//! edges, the overflow and underflow boundaries, the branch points, and the subnormals.
//!
//! Three things are checked at each such point, and they are different questions:
//!
//! 1. **The value at the singularity itself** — `±∞`, `NaN`, or an exact zero, deliberately
//!    chosen and asserted rather than whatever the arithmetic happened to produce.
//! 2. **The approach to it** — that the function tends to its known local form
//!    (`K₀ ~ −ln(x/2) − γ`, `Y₁ ~ −2/πx`, `ζ ~ 1/(s−1) + γ`) rather than merely being large.
//!    A function can return `inf` at the pole and still be wrong just beside it.
//! 3. **That nothing produces a `NaN` by accident.** A `NaN` from an in-domain argument is the
//!    worst failure mode available to a numeric library, because it propagates silently.
//!
//! # On accuracy at a zero
//!
//! `J`, `Y`, `Si` and `Ci` cross zero infinitely often, and no implementation can be accurate
//! in the *relative* sense at a zero. The zeros are irrational, so the nearest `f64` to one
//! already differs from the true zero by up to `ε·x`; with a non-vanishing derivative there,
//! the returned value is dominated by that input offset rather than by anything the
//! implementation did. These tests therefore assert *absolute* smallness at a zero, and check
//! the structural facts that remain meaningful: that the sign changes across it, and that
//! consecutive zeros bracket exactly one crossing.

mod airy_edges;
mod domain_edges;
mod extremes;
mod log_domain;
mod singularities;
mod small_argument_limits;
mod ulp_boundaries;
mod zeros;
