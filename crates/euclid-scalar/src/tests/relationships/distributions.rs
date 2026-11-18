//! Discrete-to-continuous edges tying the integer-tier combinatorics to the incomplete functions.

use crate::{Natural, Real};
use euclid_utils::assert_close;

// --- discrete <-> continuous edges --------------------------------------

#[test]
fn the_poisson_tail_equals_the_upper_incomplete_gamma() {
    // sum_{i<=k} e^{-lambda} lambda^i / i!  =  Q(k+1, lambda)
    //
    // This ties `Natural::ln_factorial` to `gamma_ur`: the left side is built purely from the
    // combinatorics on the integer tier, the right from the continuous incomplete gamma.
    for &lambda in &[0.5f64, 2.0, 7.0, 20.0] {
        let mut cumulative = 0.0f64;
        for k in 0..40u64 {
            // pmf through the trait's log-factorial, not a local factorial.
            let ln_pmf = -lambda + f64::from(u32::try_from(k).unwrap_or(0)) * lambda.ln()
                - Natural::ln_factorial(&k);
            cumulative += ln_pmf.exp();
            let continuous = Real::gamma_ur(lambda, f64::from(u32::try_from(k).unwrap_or(0)) + 1.0);
            assert_close(
                cumulative,
                continuous,
                1e-9,
                &format!("Poisson({lambda}) CDF at {k} = Q({}, {lambda})", k + 1),
            );
        }
    }
}

#[test]
fn the_binomial_cdf_equals_the_incomplete_beta() {
    // sum_{i<=k} C(n,i) p^i (1-p)^{n-i}  =  I_{1-p}(n-k, k+1)
    //
    // Ties `Natural::binomial` to `beta_reg`. The discrete side uses exact binomial
    // coefficients; the continuous side a continued fraction.
    for &n in &[5u64, 12, 30] {
        for &p in &[0.2f64, 0.5, 0.75] {
            let mut cumulative = 0.0f64;
            for k in 0..n {
                let c = Natural::binomial(&n, &k);
                cumulative += c
                    * Real::powf(&p, f64::from(u32::try_from(k).unwrap_or(0)))
                    * Real::powf(&(1.0 - p), f64::from(u32::try_from(n - k).unwrap_or(0)));
                let kf = f64::from(u32::try_from(k).unwrap_or(0));
                let nf = f64::from(u32::try_from(n).unwrap_or(0));
                let continuous = Real::beta_reg(1.0 - p, nf - kf, kf + 1.0);
                assert_close(
                    cumulative,
                    continuous,
                    1e-9,
                    &format!("Binomial({n},{p}) CDF at {k} = I_{{1-p}}"),
                );
            }
        }
    }
}

#[test]
fn log_factorial_and_log_gamma_are_the_same_function() {
    // ln(n!) = ln Gamma(n+1). Ties the integer-tier combinatorics to the continuous gamma.
    for n in 0..100u64 {
        let nf = f64::from(u32::try_from(n).unwrap_or(0));
        assert_close(
            Natural::ln_factorial(&n),
            Real::ln_gamma(nf + 1.0),
            1e-12,
            &format!("ln({n}!) = ln_gamma({})", n + 1),
        );
    }
}

#[test]
fn log_binomial_matches_the_log_gamma_expression() {
    // ln C(n,k) = lnGamma(n+1) - lnGamma(k+1) - lnGamma(n-k+1). Ties `ln_binomial` to
    // `ln_gamma`, and `binomial` to its own log form.
    for n in 1..40u64 {
        for k in 0..=n {
            let (nf, kf) = (
                f64::from(u32::try_from(n).unwrap_or(0)),
                f64::from(u32::try_from(k).unwrap_or(0)),
            );
            let want =
                Real::ln_gamma(nf + 1.0) - Real::ln_gamma(kf + 1.0) - Real::ln_gamma(nf - kf + 1.0);
            assert_close(
                Natural::ln_binomial(&n, &k),
                want,
                1e-10,
                &format!("ln C({n},{k})"),
            );
            // And the direct form is its exponential.
            assert_close(
                Natural::binomial(&n, &k),
                Natural::ln_binomial(&n, &k).exp(),
                1e-9,
                &format!("C({n},{k}) = exp(ln C)"),
            );
        }
    }
}
