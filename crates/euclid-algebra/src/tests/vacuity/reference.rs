//! Reference contracts and rejection controls.

use super::*;

// ===== The ref-magma suite, checked against itself ========================

/// A carrier whose `add_ref` disagrees with its `add`.
///
/// [`conformance::ref_magma_laws`] was added in response to a mutation finding, so it owes the
/// same debt every other suite here owes: a suite nobody has seen fail is a suite nobody has
/// tested. Without this, `ref_magma_laws` could itself be replaced with an empty body — which
/// would restore exactly the hole it was written to close.
#[derive(Clone, Copy, PartialEq, Debug)]
struct DivergentRef(i64);

impl AdditiveMagma for DivergentRef {
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}
impl MultiplicativeMagma for DivergentRef {
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0))
    }
}
impl RefAdditiveMagma for DivergentRef {
    /// Disagrees with `add` by one — the shape of a copy-paste slip between two impl blocks.
    fn add_ref(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_add(rhs.0).wrapping_add(1))
    }
}
impl RefMultiplicativeMagma for DivergentRef {
    fn mul_ref(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0))
    }
}
impl ConformanceSample for DivergentRef {
    fn sample() -> Vec<Self> {
        vec![Self(0), Self(1), Self(2), Self(-3)]
    }
}

#[test]
fn the_ref_magma_laws_reject_a_divergent_by_reference_operation() {
    // Through the `_for` wrapper, so the sample-reading delegation is pinned as well as the
    // laws it delegates to — the wrappers were dead code until this suite started using them.
    let message = rejects(|| conformance::ref_magma_laws_for::<DivergentRef>("DivergentRef"))
        .expect("ref_magma_laws must reject an add_ref that disagrees with add");
    assert!(
        message.contains("add_ref"),
        "the failure must name the operation that diverged: {message}"
    );
}
