//! Inverse relationships among the exp/log helpers.

use crate::Real;
use euclid_utils::assert_close;

// --- exp / log pairs ----------------------------------------------------

#[test]
fn ln_1p_and_exp_m1_are_inverses() {
    // `ln_1p` is defined for x > -1 only.
    for i in -9..30 {
        let x = f64::from(i) * 0.1;
        assert_close(
            Real::exp_m1(&Real::ln_1p(&x)),
            x,
            1e-13,
            &format!("exp_m1(ln_1p({x})) = {x}"),
        );
    }
}

#[test]
fn ln_1m_exp_and_ln_1m_relate_through_exp() {
    // ln(1 − e^x) with x = ln p is ln(1 − p)
    for i in 1..99 {
        let p = f64::from(i) * 0.01;
        assert_close(
            Real::ln_1m_exp(&p.ln()),
            Real::ln_1m(&p),
            1e-11,
            &format!("ln_1m_exp(ln {p}) = ln_1m({p})"),
        );
    }
}

#[test]
fn ln_exp_m1_inverts_ln_1p_exp() {
    // softplus and its inverse
    for i in 1..100 {
        let x = f64::from(i) * 0.25;
        let soft = Real::ln_1p_exp(&x);
        assert_close(
            Real::ln_exp_m1(&soft),
            x,
            1e-10,
            &format!("ln_exp_m1(softplus({x})) = {x}"),
        );
    }
}
