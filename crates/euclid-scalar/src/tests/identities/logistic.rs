//! Identities of the logistic family.

use crate::Real;
use euclid_utils::assert_close;

// --- logistic family ----------------------------------------------------

#[test]
fn logit_and_sigmoid_are_inverses() {
    for i in 1..100 {
        let p = f64::from(i) * 0.01;
        assert_close(
            Real::sigmoid(&Real::logit(&p)),
            p,
            1e-12,
            &format!("sigmoid(logit({p})) = {p}"),
        );
    }
    // Beyond |x| ≈ 15, σ(x) is within one f64 ulp of 1 and `logit` cannot invert it. The
    // round trip is checked where it carries information.
    for i in -20..20 {
        let x = f64::from(i) * 0.5;
        assert_close(
            Real::logit(&Real::sigmoid(&x)),
            x,
            1e-10,
            &format!("logit(sigmoid({x})) = {x}"),
        );
    }
}

#[test]
fn sigmoid_reflects_about_one_half() {
    // σ(−x) = 1 − σ(x)
    for i in -100..100 {
        let x = f64::from(i) * 0.25;
        assert_close(
            Real::sigmoid(&-x),
            1.0 - Real::sigmoid(&x),
            1e-13,
            "sigmoid(-x) = 1 - sigmoid(x)",
        );
    }
}

#[test]
fn log_sigmoid_forms_agree_with_their_logs() {
    for i in -60..60 {
        let x = f64::from(i) * 0.5;
        assert_close(
            Real::ln_sigmoid(&x),
            Real::sigmoid(&x).ln(),
            1e-12,
            "ln_sigmoid = ln(sigmoid)",
        );
        // Only where `1 - sigmoid(x)` has not cancelled away; `ln_1m_sigmoid` exists exactly
        // because that expression is unusable for larger x. The stable relation
        // `ln(1-sigma(x)) = -softplus(x)` is checked for all x below.
        if x < 15.0 {
            assert_close(
                Real::ln_1m_sigmoid(&x),
                (1.0 - Real::sigmoid(&x)).ln(),
                1e-9,
                "ln_1m_sigmoid = ln(1 - sigmoid)",
            );
        }
        assert_close(
            Real::ln_1m_sigmoid(&x),
            -Real::ln_1p_exp(&x),
            1e-13,
            "ln(1 - sigmoid(x)) = -softplus(x)",
        );
    }
    // softplus relates them: ln σ(x) = −softplus(−x)
    for i in -100..100 {
        let x = f64::from(i) * 0.5;
        assert_close(
            Real::ln_sigmoid(&x),
            -Real::ln_1p_exp(&-x),
            1e-13,
            "ln sigmoid(x) = -softplus(-x)",
        );
    }
}
