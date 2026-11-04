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
    pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_860_606_512_090_082_402_431;

    /// The golden ratio `φ = 1.6180339887…`, the positive root of `x² = x + 1`.
    pub const GOLDEN_RATIO: f64 = 1.618_033_988_749_894_848_204_586_834_365_638_118;

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
    /// See [`f64::EULER_GAMMA`](super::f64::EULER_GAMMA).
    pub const EULER_GAMMA: f32 = 0.577_215_664_901_532_860_606_512_090_082_402_431;
    /// See [`f64::GOLDEN_RATIO`](super::f64::GOLDEN_RATIO).
    pub const GOLDEN_RATIO: f32 = 1.618_033_988_749_894_848_204_586_834_365_638_118;
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
mod tests {
    use crate::Real;
    use euclid_utils::assert_close;

    /// Every constant is checked against an identity, not against transcribed digits. A typo in
    /// any of them breaks the relation it is defined by.
    const TOL: f64 = 1e-15;

    #[test]
    fn euler_gamma_is_negative_digamma_at_one() {
        assert_close(
            super::f64::EULER_GAMMA,
            -Real::digamma(1.0f64),
            1e-11,
            "gamma = -psi(1)",
        );
    }

    #[test]
    fn golden_ratio_solves_its_quadratic() {
        let phi = super::f64::GOLDEN_RATIO;
        assert_close(phi * phi, phi + 1.0, TOL, "phi^2 = phi + 1");
        // And equals (1 + sqrt 5)/2.
        assert_close(phi, (1.0 + 5.0f64.sqrt()) / 2.0, TOL, "phi = (1+sqrt5)/2");
        // Its reciprocal is phi - 1.
        assert_close(1.0 / phi, phi - 1.0, TOL, "1/phi = phi - 1");
    }

    #[test]
    fn the_square_roots_square_back() {
        assert_close(
            super::f64::SQRT_PI * super::f64::SQRT_PI,
            core::f64::consts::PI,
            TOL,
            "sqrt_pi^2 = pi",
        );
        assert_close(
            super::f64::SQRT_TAU * super::f64::SQRT_TAU,
            core::f64::consts::TAU,
            TOL,
            "sqrt_tau^2 = tau",
        );
        // sqrt(pi) is Gamma(1/2).
        assert_close(
            super::f64::SQRT_PI,
            Real::gamma(0.5f64),
            1e-14,
            "sqrt_pi = Gamma(1/2)",
        );
    }

    #[test]
    fn the_logarithms_exponentiate_back() {
        assert_close(
            super::f64::LN_PI.exp(),
            core::f64::consts::PI,
            TOL,
            "exp(ln pi) = pi",
        );
        assert_close(
            super::f64::LN_TAU.exp(),
            core::f64::consts::TAU,
            TOL,
            "exp(ln tau) = tau",
        );
        assert_close(
            super::f64::HALF_LN_TAU,
            0.5 * super::f64::LN_TAU,
            TOL,
            "half_ln_tau = ln_tau / 2",
        );
        // And the normal-density use: exp(-half_ln_tau) = 1/sqrt(2 pi).
        assert_close(
            (-super::f64::HALF_LN_TAU).exp(),
            1.0 / super::f64::SQRT_TAU,
            TOL,
            "the normal normalizer",
        );
    }

    #[test]
    fn apery_is_minus_half_the_second_polygamma_at_one() {
        // zeta(3) = -psi''(1)/2
        assert_close(
            super::f64::APERY,
            -Real::polygamma(1.0f64, 2) / 2.0,
            1e-11,
            "zeta(3) = -psi''(1)/2",
        );
    }

    #[test]
    fn catalan_comes_from_trigamma_at_a_quarter() {
        // psi'(1/4) = pi^2 + 8G, so G = (psi'(1/4) - pi^2)/8.
        let pi2 = core::f64::consts::PI * core::f64::consts::PI;
        assert_close(
            super::f64::CATALAN,
            (Real::trigamma(0.25f64) - pi2) / 8.0,
            1e-10,
            "G = (psi'(1/4) - pi^2)/8",
        );
    }

    #[test]
    fn omega_solves_its_defining_equation() {
        let w = super::f64::OMEGA;
        assert_close(w * w.exp(), 1.0, TOL, "Omega e^Omega = 1");
        // Equivalently, Omega = e^{-Omega} = ln(1/Omega).
        assert_close(w, (-w).exp(), TOL, "Omega = e^-Omega");
        assert_close(w, (1.0 / w).ln(), 1e-14, "Omega = ln(1/Omega)");
    }

    #[test]
    fn the_plastic_number_solves_its_cubic() {
        let rho = super::f64::PLASTIC_NUMBER;
        assert_close(rho * rho * rho, rho + 1.0, TOL, "rho^3 = rho + 1");
    }

    #[test]
    fn the_dottie_number_is_the_fixed_point_of_cosine() {
        let d = super::f64::DOTTIE;
        assert_close(d.cos(), d, TOL, "cos(d) = d");
        // And iterating cosine from anywhere converges to it, which is what makes it the
        // *unique* fixed point.
        let mut x = 1.0f64;
        for _ in 0..200 {
            x = x.cos();
        }
        assert_close(x, d, 1e-14, "cosine iteration converges to Dottie");
    }

    #[test]
    fn the_lemniscate_constant_comes_from_gamma_at_a_quarter() {
        // varpi = Gamma(1/4)^2 / (2 sqrt(2 pi))
        let g = Real::gamma(0.25f64);
        assert_close(
            super::f64::LEMNISCATE,
            g * g / (2.0 * super::f64::SQRT_TAU),
            1e-13,
            "varpi = Gamma(1/4)^2 / (2 sqrt(2 pi))",
        );
    }

    #[test]
    fn gauss_constant_is_the_lemniscate_over_pi() {
        assert_close(
            super::f64::GAUSS,
            super::f64::LEMNISCATE / core::f64::consts::PI,
            TOL,
            "G = varpi / pi",
        );
        // It is also 1/AGM(1, sqrt 2) — computed here by the arithmetic-geometric mean, which
        // converges quadratically.
        let (mut a, mut g) = (1.0f64, 2.0f64.sqrt());
        for _ in 0..10 {
            let next_a = 0.5 * (a + g);
            g = (a * g).sqrt();
            a = next_a;
        }
        assert_close(super::f64::GAUSS, 1.0 / a, 1e-15, "G = 1/AGM(1, sqrt 2)");
    }

    #[test]
    fn the_zeta_values_match_their_closed_forms() {
        let pi = core::f64::consts::PI;
        assert_close(super::f64::ZETA_2, pi * pi / 6.0, TOL, "zeta(2) = pi^2/6");
        assert_close(
            super::f64::ZETA_4,
            pi.powi(4) / 90.0,
            TOL,
            "zeta(4) = pi^4/90",
        );
        // zeta(2) is psi'(1), which ties them to the polygamma ladder.
        assert_close(
            super::f64::ZETA_2,
            Real::trigamma(1.0f64),
            1e-11,
            "zeta(2) = psi'(1)",
        );
    }

    #[test]
    fn the_trait_accessors_return_the_module_constants() {
        // The digits live in this module once; the trait must agree with them on both carriers.
        assert_eq!(<f64 as Real>::euler_gamma(), super::f64::EULER_GAMMA);
        assert_eq!(<f64 as Real>::golden_ratio(), super::f64::GOLDEN_RATIO);
        assert_eq!(<f64 as Real>::sqrt_pi(), super::f64::SQRT_PI);
        assert_eq!(<f64 as Real>::sqrt_tau(), super::f64::SQRT_TAU);
        assert_eq!(<f64 as Real>::apery(), super::f64::APERY);
        assert_eq!(<f64 as Real>::catalan(), super::f64::CATALAN);
        assert_eq!(<f64 as Real>::omega(), super::f64::OMEGA);
        assert_eq!(<f64 as Real>::ln_pi(), super::f64::LN_PI);
        assert_eq!(<f64 as Real>::ln_tau(), super::f64::LN_TAU);
        assert_eq!(<f64 as Real>::half_ln_tau(), super::f64::HALF_LN_TAU);

        assert_eq!(<f32 as Real>::euler_gamma(), super::f32::EULER_GAMMA);
        assert_eq!(<f32 as Real>::golden_ratio(), super::f32::GOLDEN_RATIO);
        assert_eq!(<f32 as Real>::sqrt_pi(), super::f32::SQRT_PI);
        assert_eq!(<f32 as Real>::omega(), super::f32::OMEGA);
    }

    #[test]
    fn the_f32_constants_are_the_narrowed_f64_ones() {
        // Each f32 value must be the f64 value rounded, not an independently-typed literal
        // that drifted.
        macro_rules! narrows {
            ($($name:ident),+ $(,)?) => {
                $({
                    let wide = super::f64::$name;
                    let narrow = super::f32::$name;
                    assert_eq!(
                        narrow, wide as f32,
                        concat!(stringify!($name), " f32 is not the narrowed f64")
                    );
                })+
            };
        }
        narrows!(
            EULER_GAMMA,
            GOLDEN_RATIO,
            SQRT_PI,
            SQRT_TAU,
            LN_PI,
            LN_TAU,
            HALF_LN_TAU,
            APERY,
            CATALAN,
            OMEGA,
            PLASTIC_NUMBER,
            DOTTIE,
            LEMNISCATE,
            GAUSS,
            ZETA_2,
            ZETA_4,
        );
    }
}
