use crate::numeric::{Boolean, Integer, Natural, Real};

use super::KeyedRandomBackend;

/// Explicit-key scalar random draws.
///
/// Implement this for backends that can draw scalar real values without committing to any
/// vector, matrix, or tensor representation.
pub trait KeyedRandomScalar: KeyedRandomBackend {
    /// The backend's scalar real value carrier.
    type Scalar: Real;
    /// The backend's scalar boolean/mask carrier.
    type Bool: Boolean;
    /// The backend's scalar signed-integer carrier.
    type Int: Integer<Real = Self::Scalar>;
    /// The backend's scalar unsigned-integer carrier.
    type UInt: Natural<Real = Self::Scalar, Integer = Self::Int>;

    /// Draw a scalar from `Uniform(0, 1)`.
    fn uniform01_scalar(key: &Self::Key) -> Self::Scalar;

    /// Draw one raw uniformly-random `u32` word, lifted into the backend's unsigned carrier.
    fn bits_u32_scalar(key: &Self::Key) -> Self::UInt;

    /// Draw one raw uniformly-random `u64` word, lifted into the backend's unsigned carrier.
    fn bits_u64_scalar(key: &Self::Key) -> Self::UInt;

    /// Draw one raw uniformly-random word with an explicit bit width.
    ///
    /// The provided default dispatches the two common widths to
    /// [`bits_u32_scalar`](Self::bits_u32_scalar)/[`bits_u64_scalar`](Self::bits_u64_scalar);
    /// a backend that needs other widths overrides this method.
    #[allow(clippy::panic)] // default-only guard: backends with other widths override this.
    fn random_bits_scalar(key: &Self::Key, bit_width: u32) -> Self::UInt {
        match bit_width {
            32 => Self::bits_u32_scalar(key),
            64 => Self::bits_u64_scalar(key),
            _ => panic!("random_bits_scalar only has default support for 32- or 64-bit words"),
        }
    }

    /// Draw a scalar standard normal.
    fn standard_normal_scalar(key: &Self::Key) -> Self::Scalar;

    /// Draw a scalar normal with explicit mean and standard deviation.
    fn normal_scalar(key: &Self::Key, mean: Self::Scalar, stddev: Self::Scalar) -> Self::Scalar {
        Self::standard_normal_scalar(key) * stddev + mean
    }

    /// Draw a scalar `Gamma(shape, 1)` variate.
    fn standard_gamma_scalar(key: &Self::Key, shape_parameter: Self::Scalar) -> Self::Scalar;

    /// Draw a scalar Bernoulli mask with success probability `p`.
    fn bernoulli_scalar(key: &Self::Key, p: Self::Scalar) -> Self::Bool;

    /// Draw uniformly from the half-open integer interval `[low, high)`.
    fn randint_scalar(key: &Self::Key, low: Self::Int, high: Self::Int) -> Self::Int;
}

/// Optional scalar distribution samplers supplied by a backend.
///
/// Implement this when the backend has native or carefully tuned algorithms for these
/// distributions. Generic code that cares about sampler quality can require this trait instead of
/// constructing samples from lower-level uniforms/normals/gammas.
pub trait KeyedRandomScalarDistributions: KeyedRandomScalar {
    /// Draw a scalar `Exponential(1)` variate.
    fn standard_exponential_scalar(key: &Self::Key) -> Self::Scalar;

    /// Draw a scalar standard Gumbel variate.
    fn standard_gumbel_scalar(key: &Self::Key) -> Self::Scalar;

    /// Draw a scalar standard logistic variate.
    fn standard_logistic_scalar(key: &Self::Key) -> Self::Scalar;

    /// Draw a scalar standard Cauchy variate.
    fn standard_cauchy_scalar(key: &Self::Key) -> Self::Scalar;

    /// Draw a scalar `Beta(alpha, beta)` variate.
    fn beta_scalar(key: &Self::Key, alpha: Self::Scalar, beta: Self::Scalar) -> Self::Scalar;

    /// Draw a scalar `Poisson(rate)` variate.
    fn poisson_scalar(key: &Self::Key, rate: Self::Scalar) -> Self::Int;
}
