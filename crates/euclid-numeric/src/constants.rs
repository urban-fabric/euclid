//! Mathematical constants that `core` does not provide.
//!
//! `core::f64::consts` covers what a systems language needs: `π`, `τ`, `e`, `√2`, the
//! logarithm conversion factors, the fractions of `π`. This module covers what a *mathematics*
//! library needs beyond that — the constants that turn up in analysis, number theory, and
//! combinatorics but have no reason to be in a standard library.
//!
//! Nothing here duplicates `core`. Where a constant is available there, use `core`; the
//! [`Real`](crate::Real) trait's accessors already route to it.
//!
//! That rule is maintained rather than assumed: `core` grows. Rust 1.99 added
//! `EULER_GAMMA` and `GOLDEN_RATIO`, both of which this module had been carrying as its own
//! digits. They are still named here, so the catalogue stays complete and the identity tests
//! keep covering them, but they are now **defined as** the `core` values rather than retyped —
//! verified bit-identical to the literals they replaced, on both carriers. Clippy's
//! `approx_constant` lint is what caught the overlap, and is why this is worth re-checking on a
//! toolchain bump.
//!
//! # The split between this module and the trait
//!
//! [`Real`](crate::Real) carries a constant as a **method** when a carrier-generic formula
//! needs to lift it — an expression or graph carrier has to be able to intern its own
//! representation of `π`, so `T::pi()` is the only portable way to write it. That set is
//! deliberately small.
//!
//! This module carries the **catalogue**: every constant, as a plain `f64` or `f32`, for code
//! that is already concrete. A consumer writing against `f64` directly should reach here; a
//! consumer writing against `T: Real` reaches for the trait method and gets only what the trait
//! chose to expose.
//!
//! The digits also live here *once*. They were previously typed into both the `f64` and `f32`
//! trait implementations, which is two chances to mistype the same number.
//!
//! # Provenance
//!
//! Every constant below is **verified by an identity** in the test module — against a closed
//! form, a defining equation, or a function elsewhere in this crate. Nothing is asserted on the
//! strength of transcribed digits alone:
//!
//! | constant | verified by |
//! |----------|-------------|
//! | `EULER_GAMMA` | `γ = −ψ(1)` |
//! | `GOLDEN_RATIO` | `φ² = φ + 1` |
//! | `SQRT_PI`, `SQRT_TAU` | squaring |
//! | `LN_PI`, `LN_TAU`, `HALF_LN_TAU` | exponentiating |
//! | `APERY` | `ζ(3) = −ψ''(1)/2` |
//! | `CATALAN` | `G = (ψ'(¼) − π²)/8` |
//! | `OMEGA` | `Ω e^Ω = 1` |
//! | `NEG_INV_E` | `−1/e = (−1)·e^{−1}`, and `W(−1/e) = −1` |
//! | `PLASTIC_NUMBER` | `ρ³ = ρ + 1` |
//! | `DOTTIE` | `cos(d) = d` |
//! | `LEMNISCATE` | `ϖ = Γ(¼)²/(2√(2π))` |
//! | `GAUSS` | `G = ϖ/π` |
//! | `ZETA_2`, `ZETA_4` | `π²/6`, `π⁴/90` |
//!
//! Constants with no cheap verification route — Glaisher–Kinkelin, the Feigenbaum constants,
//! Khinchin's — are deliberately **omitted** rather than included on trust. They can be added
//! once there is a way to check them here.

/// Double-precision constants.
#[allow(clippy::excessive_precision)]
pub mod f64 {
    /// The Euler–Mascheroni constant `γ = 0.5772156649…`, equal to `−ψ(1)`.
    ///
    /// The additive term in the Gumbel, Weibull, and Gamma entropies, and the limit of
    /// `Hₙ − ln n`.
    ///
    /// In `core` as of Rust 1.99; taken from there rather than retyped.
    pub const EULER_GAMMA: f64 = core::f64::consts::EULER_GAMMA;

    /// The golden ratio `φ = 1.6180339887…`, the positive root of `x² = x + 1`.
    ///
    /// In `core` as of Rust 1.99; taken from there rather than retyped.
    pub const GOLDEN_RATIO: f64 = core::f64::consts::GOLDEN_RATIO;

    /// `√π = 1.7724538509…`, equal to `Γ(½)`.
    ///
    /// The Gaussian normalizer, and the factor relating `erf` to the incomplete gamma.
    pub const SQRT_PI: f64 = 1.772_453_850_905_516_027_298_167_483_341_145_183;

    /// `√τ = √(2π) = 2.5066282746…`, the normal density's normalizing divisor.
    pub const SQRT_TAU: f64 = 2.506_628_274_631_000_502_415_765_284_811_045_253;

    /// `ln π = 1.1447298858…`.
    pub const LN_PI: f64 = 1.144_729_885_849_400_174_143_427_351_353_058_712;

    /// `ln τ = ln 2π = 1.8378770664…`.
    pub const LN_TAU: f64 = 1.837_877_066_409_345_483_560_659_472_811_235_280;

    /// `½ ln τ = 0.9189385332…`, the log-normalizer of a standard normal density.
    pub const HALF_LN_TAU: f64 = 0.918_938_533_204_672_741_780_329_736_405_617_640;

    /// Apéry's constant `ζ(3) = 1.2020569032…`.
    ///
    /// Appears in the third moment of a log-gamma and in several lattice sums; equals
    /// `−ψ''(1)/2`.
    pub const APERY: f64 = 1.202_056_903_159_594_285_399_738_161_511_449_991;

    /// Catalan's constant `G = 0.9159655942…`, the value of `Σ (−1)ⁿ/(2n+1)²`.
    pub const CATALAN: f64 = 0.915_965_594_177_219_015_054_603_514_932_384_111;

    /// The omega constant `Ω = 0.5671432904…`, the unique real solution of `Ω e^Ω = 1`.
    ///
    /// Equivalently `W(1)`, the Lambert `W` function at one.
    pub const OMEGA: f64 = 0.567_143_290_409_783_872_999_968_662_210_355_550;

    /// `−1/e = −0.3678794412…`, the minimum of `x e^x` and the branch point of the Lambert
    /// `W` function.
    ///
    /// The map `w ↦ w e^w` turns around at `w = −1`, taking the value `−1/e` there. That is
    /// what makes this the edge of `W`'s real domain — below it there is no real solution of
    /// `w e^w = x`, and above it there are two — and it is why `W` loses half its digits as it
    /// approaches: the derivative vanishes, so the inverse behaves like a square root.
    ///
    /// Carried as a constant rather than written `-core::f64::consts::E.recip()` because that
    /// spelling rounds twice, once into `E` and once through the reciprocal, and is not
    /// guaranteed to land on the nearest `f64` to the true `−1/e`.
    pub const NEG_INV_E: f64 = -0.367_879_441_171_442_321_595_523_770_161_460_867;

    /// The plastic number `ρ = 1.3247179572…`, the real root of `x³ = x + 1`.
    ///
    /// The cubic analogue of the golden ratio, and the limiting ratio of the Padovan sequence.
    pub const PLASTIC_NUMBER: f64 = 1.324_717_957_244_746_025_960_908_854_478_097_341;

    /// The Dottie number `0.7390851332…`, the unique real fixed point of `cos`.
    pub const DOTTIE: f64 = 0.739_085_133_215_160_641_655_312_087_673_873_404;

    /// The lemniscate constant `ϖ = 2.6220575543…`, equal to `Γ(¼)²/(2√(2π))`.
    ///
    /// The arc length of a lemniscate of unit parameter, and the `π` of the elliptic world.
    pub const LEMNISCATE: f64 = 2.622_057_554_292_119_810_464_839_589_891_119_414;

    /// Gauss's constant `G = 0.8346268417…`, equal to `ϖ/π` and to `1/AGM(1, √2)`.
    pub const GAUSS: f64 = 0.834_626_841_674_073_186_281_429_732_799_046_809;

    /// `ζ(2) = π²/6 = 1.6449340668…`, the Basel sum.
    pub const ZETA_2: f64 = 1.644_934_066_848_226_436_472_415_166_646_025_189;

    /// `ζ(4) = π⁴/90 = 1.0823232337…`.
    pub const ZETA_4: f64 = 1.082_323_233_711_138_191_516_003_696_541_167_903;
}

/// Single-precision constants, each the nearest `f32` to its [`f64`](mod@self::f64)
/// counterpart.
#[allow(clippy::excessive_precision)]
pub mod f32 {
    /// See [`f64::EULER_GAMMA`](super::f64::EULER_GAMMA). In `core` as of Rust 1.99.
    pub const EULER_GAMMA: f32 = core::f32::consts::EULER_GAMMA;
    /// See [`f64::GOLDEN_RATIO`](super::f64::GOLDEN_RATIO). In `core` as of Rust 1.99.
    pub const GOLDEN_RATIO: f32 = core::f32::consts::GOLDEN_RATIO;
    /// See [`f64::SQRT_PI`](super::f64::SQRT_PI).
    pub const SQRT_PI: f32 = 1.772_453_850_905_516_027_298_167_483_341_145_183;
    /// See [`f64::SQRT_TAU`](super::f64::SQRT_TAU).
    pub const SQRT_TAU: f32 = 2.506_628_274_631_000_502_415_765_284_811_045_253;
    /// See [`f64::LN_PI`](super::f64::LN_PI).
    pub const LN_PI: f32 = 1.144_729_885_849_400_174_143_427_351_353_058_712;
    /// See [`f64::LN_TAU`](super::f64::LN_TAU).
    pub const LN_TAU: f32 = 1.837_877_066_409_345_483_560_659_472_811_235_280;
    /// See [`f64::HALF_LN_TAU`](super::f64::HALF_LN_TAU).
    pub const HALF_LN_TAU: f32 = 0.918_938_533_204_672_741_780_329_736_405_617_640;
    /// See [`f64::APERY`](super::f64::APERY).
    pub const APERY: f32 = 1.202_056_903_159_594_285_399_738_161_511_449_991;
    /// See [`f64::CATALAN`](super::f64::CATALAN).
    pub const CATALAN: f32 = 0.915_965_594_177_219_015_054_603_514_932_384_111;
    /// See [`f64::OMEGA`](super::f64::OMEGA).
    pub const OMEGA: f32 = 0.567_143_290_409_783_872_999_968_662_210_355_550;
    /// See [`f64::NEG_INV_E`](super::f64::NEG_INV_E).
    pub const NEG_INV_E: f32 = -0.367_879_441_171_442_321_595_523_770_161_460_867;
    /// See [`f64::PLASTIC_NUMBER`](super::f64::PLASTIC_NUMBER).
    pub const PLASTIC_NUMBER: f32 = 1.324_717_957_244_746_025_960_908_854_478_097_341;
    /// See [`f64::DOTTIE`](super::f64::DOTTIE).
    pub const DOTTIE: f32 = 0.739_085_133_215_160_641_655_312_087_673_873_404;
    /// See [`f64::LEMNISCATE`](super::f64::LEMNISCATE).
    pub const LEMNISCATE: f32 = 2.622_057_554_292_119_810_464_839_589_891_119_414;
    /// See [`f64::GAUSS`](super::f64::GAUSS).
    pub const GAUSS: f32 = 0.834_626_841_674_073_186_281_429_732_799_046_809;
    /// See [`f64::ZETA_2`](super::f64::ZETA_2).
    pub const ZETA_2: f32 = 1.644_934_066_848_226_436_472_415_166_646_025_189;
    /// See [`f64::ZETA_4`](super::f64::ZETA_4).
    pub const ZETA_4: f32 = 1.082_323_233_711_138_191_516_003_696_541_167_903;
}

#[cfg(test)]
mod tests;
