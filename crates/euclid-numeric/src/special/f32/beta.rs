//! Native `f32` beta family — the single-precision port of [`mod@super::super::beta`]. Continued
//! fraction from Numerical Recipes, evaluated in `f32` with the convergence tolerance at the
//! `f32` noise floor.

use super::gamma::ln_gamma;

/// `ln B(a, b) = ln Γ(a) + ln Γ(b) - ln Γ(a+b)`.
pub(crate) fn ln_beta(a: f32, b: f32) -> f32 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// The beta function `B(a, b)`.
pub(crate) fn beta(a: f32, b: f32) -> f32 {
    ln_beta(a, b).exp()
}

/// The regularized incomplete beta `Iₓ(a, b)`, `x ∈ [0, 1]`, `a, b > 0`.
pub(crate) fn beta_reg(a: f32, b: f32, x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b)).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, 1.0 - x) / b
    }
}

const MAX_ITER: usize = 300;
// See `gamma::EPS`: loosened from the `f64` `1e-15` to just above `f32::EPSILON`.
const EPS: f32 = 1e-6;

/// Lentz continued fraction for the incomplete beta (`betacf` in Numerical Recipes).
fn beta_cf(a: f32, b: f32, x: f32) -> f32 {
    let tiny = 1e-30;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..MAX_ITER {
        let m = m as f32;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn agrees(a: f32, b: f64, tol: f32) {
        let b = b as f32;
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn beta_matches_f64_reference() {
        for &(a, b) in &[(1.0, 1.0), (2.0, 3.0), (0.5, 0.5), (5.0, 2.0)] {
            agrees(
                beta(a, b),
                crate::special::beta::beta(a as f64, b as f64),
                1e-5,
            );
        }
    }

    #[test]
    fn beta_reg_matches_f64_reference() {
        assert_eq!(beta_reg(2.0, 3.0, 0.0), 0.0);
        assert_eq!(beta_reg(2.0, 3.0, 1.0), 1.0);
        for &(a, b, x) in &[(2.0, 3.0, 0.4), (0.5, 0.5, 0.3), (5.0, 2.0, 0.7)] {
            agrees(
                beta_reg(a, b, x),
                crate::special::beta::beta_reg(a as f64, b as f64, x as f64),
                1e-5,
            );
        }
    }
}
