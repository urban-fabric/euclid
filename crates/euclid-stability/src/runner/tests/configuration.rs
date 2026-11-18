//! Configuration checks for the runner module.

use super::*;

/// The overrides are off unless asked for, which is what keeps the fixed-seed guarantee.
#[test]
fn an_unset_override_changes_nothing() {
    let config = Config::default().cases(400);
    let applied = crate::runner::apply_overrides(&config, None, None);
    assert_eq!(applied.cases, 400);
    assert_eq!(applied.seed, config.seed);
    assert_eq!(applied.max_rejects, config.max_rejects);
}

/// A soak factor multiplies the case count **and** the rejection budget.
///
/// Both, or a soak turns every filtered generator into a spurious "gave up" report: ten times the
/// cases need ten times the draws to feed them.
#[test]
fn a_soak_factor_scales_the_search() {
    let config = Config::default().cases(400).max_rejects(100);

    let applied = crate::runner::apply_overrides(&config, Some("10"), None);
    assert_eq!(applied.cases, 4_000);
    assert_eq!(applied.max_rejects, 1_000);
    assert_eq!(applied.seed, config.seed, "a soak must not move the seed");

    // Whitespace is tolerated, since these arrive from a shell.
    assert_eq!(
        crate::runner::apply_overrides(&config, Some(" 2 "), None).cases,
        800
    );
}

/// A factor of zero, or nonsense, leaves the configuration alone.
///
/// Zero is the dangerous one and the reason for the guard: multiplying the case count by it
/// would run no cases at all, which every property then reports as "never exercised". An
/// environment variable must not be able to turn the whole suite into a failure that looks like
/// a generator bug.
#[test]
fn a_degenerate_soak_factor_is_ignored_rather_than_applied() {
    let config = Config::default().cases(400).max_rejects(100);
    for raw in [
        "0",
        "",
        "nonsense",
        "-1",
        "1.5",
        "99999999999999999999999999",
    ] {
        let applied = crate::runner::apply_overrides(&config, Some(raw), None);
        assert_eq!(
            applied.cases, 400,
            "SOAK={raw:?} must leave the case count alone"
        );
        assert_eq!(applied.max_rejects, 100, "SOAK={raw:?}");
    }

    // And the smallest meaningful factor is honoured rather than swept up with the degenerate
    // ones, which is what distinguishes this guard from one that rejects everything.
    assert_eq!(
        crate::runner::apply_overrides(&config, Some("1"), None).cases,
        400
    );
    assert_eq!(
        crate::runner::apply_overrides(&config, Some("3"), None).cases,
        1_200
    );
}

/// A seed override replaces the seed and touches nothing else.
#[test]
fn a_seed_override_replaces_only_the_seed() {
    let config = Config::default().cases(400);

    let applied = crate::runner::apply_overrides(&config, None, Some("0xFF"));
    assert_eq!(applied.seed, 255);
    assert_eq!(
        applied.cases, 400,
        "a seed override must not move the cases"
    );

    assert_eq!(
        crate::runner::apply_overrides(&config, None, Some("42")).seed,
        42
    );
    // Nonsense leaves the fixed default in place rather than seeding from zero.
    assert_eq!(
        crate::runner::apply_overrides(&config, None, Some("nonsense")).seed,
        config.seed
    );

    // Both at once compose.
    let both = crate::runner::apply_overrides(&config, Some("5"), Some("7"));
    assert_eq!((both.cases, both.seed), (2_000, 7));
}

/// A seed is accepted in decimal or hex, since the default is written in hex and a reader
/// copying it back in should not have to convert.
#[test]
fn a_seed_parses_in_both_bases() {
    assert_eq!(crate::runner::parse_seed("42"), Some(42));
    assert_eq!(crate::runner::parse_seed("0xFF"), Some(255));
    assert_eq!(crate::runner::parse_seed("0Xff"), Some(255));
    assert_eq!(crate::runner::parse_seed(""), None);
    assert_eq!(crate::runner::parse_seed("nonsense"), None);
    // The default seed, round-tripped through its own notation.
    assert_eq!(
        crate::runner::parse_seed("0x5EED123456789ABC"),
        Some(0x5EED_1234_5678_9ABC)
    );
}

/// A builder sets its own field and **keeps** everything set before it.
///
/// Mutation testing replaced `Config::regressions` and `Config::budget` with
/// `Default::default()`, which silently discards every earlier call in the chain *and* turns the
/// feature off. `settings("round_trips").cases(400)` would quietly become a 256-case run
/// persisting nothing, and no test noticed.
#[test]
fn the_builders_compose_rather_than_reset() {
    let config = Config::default()
        .cases(400)
        .max_rejects(99)
        .targeted_fraction(0.25)
        .require("label", 5.0)
        .regressions("regressions.txt")
        .budget("budgets.txt");

    assert_eq!(
        config.cases, 400,
        "a later builder discarded the case count"
    );
    assert_eq!(config.max_rejects, 99);
    assert!((config.targeted_fraction - 0.25).abs() < 1e-12);
    assert_eq!(config.requirements.len(), 1);
    assert_eq!(config.regressions.as_deref(), Some("regressions.txt"));
    assert_eq!(config.budget.as_deref(), Some("budgets.txt"));
}
