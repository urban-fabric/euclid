use euclid_numeric::{Boolean, Integer, Real};

use super::RandomKey;

/// Basic explicit-key random samplers.
///
/// `shape` is the host-known output shape. Scalar implementations such as an `f64` backend use
/// `&[]`; array, graph, and dataframe backends can use non-empty shapes for tensor/column draws.
pub trait RandomSampler: RandomKey {
    /// Real-valued draw result: `f64` for scalar host sampling, or a backend value such as a JAX
    /// array / Polars expression wrapper.
    type Real: Real;
    /// Boolean-valued draw result for Bernoulli sampling.
    type Bool: Boolean;
    /// Signed integer draw result for integer sampling.
    type Int: Integer<Real = Self::Real>;

    /// Draw from `Uniform(low, high)`.
    fn uniform(key: &Self::Key, shape: &[usize], low: Self::Real, high: Self::Real) -> Self::Real;

    /// Draw from `Normal(mean, stddev)`.
    fn normal(key: &Self::Key, shape: &[usize], mean: Self::Real, stddev: Self::Real)
    -> Self::Real;

    /// Draw Bernoulli values with success probability `p`.
    fn bernoulli(key: &Self::Key, shape: &[usize], p: Self::Real) -> Self::Bool;

    /// Draw uniformly from the half-open integer interval `[low, high)`.
    fn randint(key: &Self::Key, shape: &[usize], low: Self::Int, high: Self::Int) -> Self::Int;
}

/// Gamma-family draws: the one keystone primitive a backend supplies so that the whole gamma
/// family composes on top. [`standard_gamma`](Self::standard_gamma) is `Gamma(concentration,
/// rate = 1)`; from it (with [`RandomSampler::normal`] and [`RandomKey::split`]) a consumer
/// builds `Gamma(α, β)`, `Beta`, `ChiSquared`, `StudentT`, `FisherSnedecor`, and `InverseGamma`.
///
/// The gamma quantile has no closed form, so — unlike the location-scale family, which is a
/// closed-form transform of a uniform draw — it cannot be composed from the core
/// [`RandomSampler`] alone. A backend implements it however suits its carrier: Marsaglia–Tsang
/// rejection on an eager scalar, or a fixed-iteration inverse-CDF on the regularized incomplete
/// gamma ([`Real::gamma_lr`]) on a graph carrier. On a
/// differentiable carrier it should be reparameterizable (differentiable in `concentration`).
pub trait GammaSampler: RandomSampler {
    /// Draw from `Gamma(concentration, rate = 1)` over `shape`. Divide by `rate` for a general
    /// rate.
    fn standard_gamma(key: &Self::Key, shape: &[usize], concentration: Self::Real) -> Self::Real;
}

/// Integer-count draws whose quantile has no closed form, so they don't reduce to the core
/// [`RandomSampler`] integer surface. [`poisson`](Self::poisson) is the workhorse;
/// [`binomial`](Self::binomial) and [`beta_binomial`](Self::beta_binomial) cover the
/// trial-count family (the beta-binomial being a binomial whose success probability is itself a
/// `Beta(a, b)` draw). Counts come back in the signed integer carrier [`Int`](RandomSampler::Int).
pub trait CountSampler: RandomSampler {
    /// Draw from `Poisson(rate)` over `shape`.
    fn poisson(key: &Self::Key, shape: &[usize], rate: Self::Real) -> Self::Int;

    /// Draw from `Binomial(trials, p)` over `shape`. `trials` is a (real-valued) count `n`.
    fn binomial(key: &Self::Key, shape: &[usize], trials: Self::Real, p: Self::Real) -> Self::Int;

    /// Draw from `BetaBinomial(trials, a, b)` over `shape` — `p ~ Beta(a, b)` then
    /// `Binomial(trials, p)`.
    fn beta_binomial(
        key: &Self::Key,
        shape: &[usize],
        trials: Self::Real,
        a: Self::Real,
        b: Self::Real,
    ) -> Self::Int;
}
