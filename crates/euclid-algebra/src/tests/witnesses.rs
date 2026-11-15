//! The standard semiring witnesses, checked against the witness laws.
//!
//! Samples are chosen per witness rather than shared, because each algebra's hard cases are
//! different: min-plus breaks at `+∞`, max-plus at `−∞`, widest-path at both extremes, and
//! modular arithmetic at the modulus. A sample that omits a witness's own `ZERO` would miss the
//! absorbing law, which is the one these definitions exist to get right.

use crate::conformance as laws;
use crate::witness::{Arithmetic, MaxPlus, MinPlus, ModP, SemiringWitness, WidestPath};

#[test]
fn min_plus_is_a_dioid() {
    // Includes ZERO (+∞) so the absorbing guard is exercised: without it, extending a
    // nonexistent path wraps to a small number and a shortest path comes out too short.
    let xs = [MinPlus::ZERO, MinPlus::ONE, 1, 7, -3, 1_000];
    laws::witness_dioid_laws::<MinPlus>(&xs, "MinPlus");

    assert_eq!(MinPlus::add(3, 5), 3, "⊕ is min");
    assert_eq!(MinPlus::mul(3, 5), 8, "⊗ is +");
    assert_eq!(
        MinPlus::mul(MinPlus::ZERO, 5),
        MinPlus::ZERO,
        "+∞ must absorb rather than wrap"
    );
}

#[test]
fn max_plus_is_a_dioid() {
    let xs = [MaxPlus::ZERO, MaxPlus::ONE, 1, 7, -3, 1_000];
    laws::witness_dioid_laws::<MaxPlus>(&xs, "MaxPlus");

    assert_eq!(MaxPlus::add(3, 5), 5, "⊕ is max");
    assert_eq!(MaxPlus::mul(3, 5), 8, "⊗ is +");
    assert_eq!(
        MaxPlus::mul(MaxPlus::ZERO, 5),
        MaxPlus::ZERO,
        "−∞ must absorb rather than wrap"
    );
}

#[test]
fn widest_path_is_a_dioid() {
    let xs = [WidestPath::ZERO, WidestPath::ONE, 0, 7, 100, -5];
    laws::witness_dioid_laws::<WidestPath>(&xs, "WidestPath");

    assert_eq!(WidestPath::add(3, 5), 5, "⊕ is max");
    assert_eq!(
        WidestPath::mul(3, 5),
        3,
        "⊗ is min — a path is its narrowest link"
    );
}

#[test]
fn arithmetic_is_a_semiring_but_not_a_dioid() {
    // Small values only: the laws multiply triples, and saturation is not associative once a
    // product reaches the i64 bound. That is a property of the saturating choice rather than a
    // law violation, so the sample stays inside the exact range.
    let xs = [0, 1, 2, 3, -2];
    laws::witness_semiring_laws::<Arithmetic>(&xs, "Arithmetic");

    // `a ⊕ a = a` fails, which is why it carries no `DioidWitness` marker and a closure
    // algorithm cannot accept it.
    assert_ne!(Arithmetic::add(3, 3), 3);
}

#[test]
fn mod_p_is_a_semiring() {
    type M7 = ModP<7>;
    // Canonical representatives only. `ModP` reduces its *output* into `[0, P)` but accepts any
    // `i64`, so `-1` would break `a ⊕ 0 = a`: the result is `6`, and `i64` equality cannot see
    // that `-1` and `6` are the same residue. That is a precondition on the sample, not a flaw
    // in the witness — see the note on `witness_semiring_laws`.
    let xs = [0, 1, 3, 6, 5, 2];
    laws::witness_semiring_laws::<M7>(&xs, "ModP<7>");

    assert_eq!(M7::add(6, 3), 2, "6 + 3 = 2 mod 7");
    assert_eq!(M7::mul(3, 5), 1, "3 · 5 = 1 mod 7");

    // Non-canonical input is still accepted and normalized, which is why it cannot be fed to
    // the laws but is fine at a call site.
    assert_eq!(
        M7::add(-1, 0),
        6,
        "a negative input normalizes to its representative"
    );
    assert_eq!(M7::add(9, 0), 2, "so does one above the modulus");

    // The i128 widening: a product that would overflow i64 still reduces correctly.
    assert_eq!(M7::mul(i64::MAX, i64::MAX), {
        let r = (i128::from(i64::MAX) * i128::from(i64::MAX)).rem_euclid(7);
        r as i64
    });
}

#[test]
fn one_domain_carries_several_algebras_at_once() {
    // The whole point of the witness layer: `i64` is simultaneously min-plus, max-plus,
    // widest-path and arithmetic, which coherence forbids on the element layer.
    assert_eq!(MinPlus::add(3, 5), 3);
    assert_eq!(MaxPlus::add(3, 5), 5);
    assert_eq!(WidestPath::add(3, 5), 5);
    assert_eq!(Arithmetic::add(3, 5), 8);
}
