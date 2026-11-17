//! Finite-domain boundaries and bounded metadata allocation, including large choice tapes.

use super::*;
use crate::{Config, run};
use std::rc::Rc;

#[test]
fn nonempty_variable_length_domains_enumerate_every_length() {
    let mut seen = std::collections::BTreeSet::new();
    let report = run(
        "nonempty lists",
        &Config::default().exhaustive(),
        &vec(boolean(), 1, 2),
        |value| {
            seen.insert(value);
            Ok(())
        },
    )
    .unwrap();
    assert!(report.exhaustive);
    assert_eq!(
        seen,
        [
            std::vec![false],
            std::vec![true],
            std::vec![false, false],
            std::vec![false, true],
            std::vec![true, false],
            std::vec![true, true],
        ]
        .into_iter()
        .collect()
    );
}

#[test]
fn fixed_arrays_at_the_supported_length_limit_can_be_exhausted() {
    let report = run(
        "64 constants",
        &Config::default().cases(1).exhaustive(),
        &crate::generate::array::<_, 64>(just(3)),
        |values| {
            assert_eq!(values, [3; 64]);
            Ok(())
        },
    )
    .unwrap();
    assert!(report.exhaustive);
    assert_eq!(report.passed, 1);
}

#[test]
fn cartesian_metadata_accepts_the_word_limit_and_refuses_an_oversized_allocation() {
    let budget = 262_144;
    let full = std::vec![std::vec![0; budget]];
    let empty = std::vec![std::vec![]];
    let extra = std::vec![std::vec![0]];
    let product = super::super::explore::product;
    assert_eq!(product(Some(&full), Some(&empty)).unwrap()[0].len(), budget);
    assert!(product(Some(&full), Some(&extra)).is_none());
}

fn consumes_words(count: usize) -> Gen<()> {
    Gen::new(move |source| {
        for _ in 0..count {
            source.word()?;
        }
        Ok(())
    })
    .with_metadata(move || (None, Rc::new(std::vec![std::vec![0; count]])))
}

#[test]
fn edge_metadata_obeys_its_word_budget_without_losing_the_exact_boundary() {
    // Prefix + 63 elements of 4161 words exactly fills the documented 262144-word budget.
    let exact = vec(consumes_words(4161), 63, 63);
    let (_, edges) = exact.exploration();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].len(), 262_144);
    assert!(exact.draw(&mut Source::replaying(edges[0].clone())).is_ok());

    // Prefix + 64 elements of 4096 words exceeds it by one; no partial tape is an edge.
    let oversized = vec(consumes_words(4096), 64, 64);
    assert!(oversized.exploration().1.is_empty());
}
