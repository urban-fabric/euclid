//! The runner: that it finds failures, shrinks them, and does not mistake inconclusive for
//! passing.
//!
//! Wrapper tests catch [`check`]'s libtest panic. `structured`, `exploration` and `health`
//! inspect [`super::run`]'s returned evidence directly.

use std::panic::{self, AssertUnwindSafe};

use super::{Config, check};

#[test]
fn panic_output_policy_covers_probes_and_verbose_mode() {
    for (probing, verbose, expected) in [
        (false, false, true),
        (false, true, true),
        (true, false, false),
        (true, true, true),
    ] {
        assert_eq!(super::forward_panic(probing, || verbose), expected);
    }
}

#[test]
fn shrink_diagnostics_report_only_nonzero_reuse() {
    assert_eq!(super::failure::reuse_suffix(0), "");
    assert_eq!(super::failure::reuse_suffix(3), ", 3 reused");
}
use crate::Failure;
use crate::generate::{Gen, int, vec as gen_vec};

/// Run `check` and capture its panic message, if any.
///
/// `Debug` is required because the runner renders each case to put it on the failure report and
/// into the regression file — see [`persist`](crate::persist) for why that rendering is checked
/// rather than decorative.
fn failure_message<T: 'static + core::fmt::Debug>(
    generator: Gen<T>,
    property: impl FnMut(T) -> Result<(), Failure>,
) -> Option<String> {
    let config = Config::default().cases(512);
    let mut property = property;
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        check("probe", &config, &generator, &mut property);
    }));
    match result {
        Ok(()) => None,
        Err(payload) => Some(
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "non-string panic".to_string()),
        ),
    }
}

/// A temp path no other test is using.
fn scratch(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "euclid-stability-runner-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::create_dir_all(&dir);
    dir.join("file.txt").to_string_lossy().to_string()
}

mod configuration;
mod coverage;
mod evidence;
mod search;
mod targeting;

mod exploration;
mod health;
mod structured;
