//! Structured outcomes, retaining Result-based property bodies and the check wrapper.

use std::collections::BTreeMap;
use std::fmt;

use super::Config;
use crate::shrink::Cost;

/// A property's return value. Rejections use `Failure::discard` and are not test failures.
pub type Outcome = Result<(), Failure>;

/// Why a property returned early.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    /// An assertion or explicitly returned failure.
    Assertion,
    /// An unwinding panic from user code.
    Panic,
    /// The case lies outside the property's domain and must be replaced.
    Discard,
}

/// A property's failure or domain rejection, compatible with Result-based bodies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// What failed, or why this case was rejected.
    pub message: String,
    /// The assertion's source location.
    pub location: &'static str,
    /// Distinguishes assertions, panics and discards.
    pub kind: FailureKind,
    /// Stable identity used to keep shrinking on the same failure.
    pub identity: String,
}

impl Failure {
    /// Construct a failed assertion. The location is its default identity.
    pub fn new(message: impl Into<String>, location: &'static str) -> Self {
        Self {
            message: message.into(),
            location,
            kind: FailureKind::Assertion,
            identity: location.into(),
        }
    }

    /// Reject a case without counting it as a success or a failure.
    pub fn discard(reason: impl Into<String>, location: &'static str) -> Self {
        Self {
            kind: FailureKind::Discard,
            ..Self::new(reason, location)
        }
    }

    /// Supply a stable identity when several assertions share a helper's source location.
    pub fn with_identity(mut self, identity: impl Into<String>) -> Self {
        self.identity = identity.into();
        self
    }

    pub(super) fn same_failure(&self, other: &Self) -> bool {
        self.kind == other.kind && self.identity == other.identity
    }
}

/// How the runner explores generated cases. Random is the compatibility default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Exploration {
    /// Preserve the existing deterministic random and targeted search.
    #[default]
    Random,
    /// Try the generator's complete edge tapes before random exploration.
    EdgesFirst,
    /// Visit every supported finite case; error when a domain cannot be enumerated.
    Exhaustive,
}

/// Counts, effective settings and evidence from a run, including failed runs.
#[derive(Clone, Debug)]
pub struct RunReport {
    /// Property name.
    pub name: String,
    /// Effective configuration after seed and soak overrides.
    pub config: Config,
    /// Fresh search draws, including rejected and failing cases.
    pub attempts: usize,
    /// Successful cases; discarded cases do not count.
    pub passed: usize,
    /// Fresh property cases that failed, excluding shrinking and replay probes.
    pub failed: usize,
    /// Cases rejected by generators or assumptions.
    pub rejected: usize,
    /// Persisted regression tapes examined before fresh search.
    pub replayed: usize,
    /// Edge tapes tried before random exploration.
    pub edge_cases: usize,
    /// Whether the entire finite domain was examined.
    pub exhaustive: bool,
    /// Counts for labels applied to accepted cases, at most once per label per case.
    pub labels: BTreeMap<&'static str, usize>,
    /// Rejection reasons with their draw counts.
    pub rejection_reasons: BTreeMap<String, usize>,
    /// Highest finite target score on accepted search cases.
    pub best_score: Option<f64>,
}

impl RunReport {
    pub(super) fn new(name: &str, config: Config) -> Self {
        Self {
            name: name.into(),
            config,
            attempts: 0,
            passed: 0,
            failed: 0,
            rejected: 0,
            replayed: 0,
            edge_cases: 0,
            exhaustive: false,
            labels: BTreeMap::new(),
            rejection_reasons: BTreeMap::new(),
            best_score: None,
        }
    }

    /// Accepted-case label shares, without rejected cases diluting the denominator.
    pub fn statistics(&self) -> String {
        let mut out = format!("{} cases passed", self.passed);
        for (label, hits) in &self.labels {
            let accepted = self.passed + self.failed;
            let share = if accepted == 0 {
                0.0
            } else {
                *hits as f64 * 100.0 / accepted as f64
            };
            out.push_str(&format!("\n    {label}: {hits} ({share:.2}%)"));
        }
        out
    }
}

/// A minimized counterexample, including the source tape needed for exact replay.
#[derive(Clone, Debug)]
pub struct Counterexample {
    /// Tape found before shrinking.
    pub original_tape: Vec<u64>,
    /// Smallest reproducing tape found within the shrink budget.
    pub tape: Vec<u64>,
    /// Debug rendering of the final case.
    pub case: String,
    /// The same failure observed before and after minimisation.
    pub failure: Failure,
    /// Probe and memo-reuse counts.
    pub cost: Cost,
    /// Whether this case was loaded from a regression file.
    pub replayed: bool,
    /// Configured regression destination when saving a fresh case.
    pub saved_to: Option<String>,
}

/// The reason a run could not establish the property.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunErrorKind {
    /// A stable counterexample was found.
    Property,
    /// The generator or case formatter panicked during generation or regression replay.
    Generation,
    /// Too few accepted cases within the rejection budget.
    Rejections,
    /// Required labels were not sufficiently represented.
    Coverage,
    /// No target score, or an accuracy baseline was exceeded.
    Budget,
    /// Persisted tapes no longer render as their recorded cases.
    StaleRegression,
    /// A counterexample or its generator did not reproduce consistently.
    Unstable,
    /// A configuration value is outside its documented domain.
    InvalidConfig,
    /// The requested domain is unknown, too large, or exceeds exploration tape limits.
    UnsupportedExploration,
}

/// A structured run failure. `check` displays this error in its panic for libtest.
#[derive(Clone, Debug)]
pub struct RunError {
    /// Machine-readable reason for the failure.
    pub kind: RunErrorKind,
    /// Human-readable explanation.
    pub message: String,
    /// Run settings, counts, coverage and scores at the failure.
    pub report: Box<RunReport>,
    /// Present only for a stable, minimized property failure.
    pub counterexample: Option<Box<Counterexample>>,
}

impl RunError {
    pub(super) fn new(kind: RunErrorKind, report: &RunReport, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            report: Box::new(report.clone()),
            counterexample: None,
        }
    }
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RunError {}
