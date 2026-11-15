//! Persisting a minimised tape, so a found counterexample is kept.
//!
//! A random search that finds a bug, shrinks it to four entries, prints it and then forgets it
//! has done half a job: the next run has to get lucky again. These files turn each find into a
//! permanent, zero-cost regression test — the runner replays them before it starts searching,
//! so the known-bad cases are checked first and in order.
//!
//! The format is one line per case, `<hex> # <name>`, which is deliberately boring: it diffs
//! cleanly, merges without conflict markers in the middle of a case, and a human can delete a
//! line. These files are meant to be committed.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

/// Render a tape as lowercase hex, 16 characters per entry.
#[must_use]
pub fn encode(tape: &[u64]) -> String {
    let mut out = String::with_capacity(tape.len() * 16);
    for entry in tape {
        let _ = write!(out, "{entry:016x}");
    }
    out
}

/// Parse a tape from hex. Returns `None` on anything malformed, so a hand-edited file degrades
/// to "no regressions" rather than to a panic inside the harness.
#[must_use]
pub fn decode(text: &str) -> Option<Vec<u64>> {
    let text = text.trim();
    if text.is_empty() || !text.len().is_multiple_of(16) {
        return None;
    }
    let mut tape = Vec::with_capacity(text.len() / 16);
    for chunk in text.as_bytes().chunks(16) {
        let hex = std::str::from_utf8(chunk).ok()?;
        tape.push(u64::from_str_radix(hex, 16).ok()?);
    }
    Some(tape)
}

/// Every tape recorded in `path`. Missing or unreadable file means no regressions.
#[must_use]
pub fn load(path: &str) -> Vec<Vec<u64>> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .map(|line| line.split('#').next().unwrap_or("").trim())
        .filter(|payload| !payload.is_empty())
        .filter_map(decode)
        .collect()
}

/// Append `tape` to `path` unless it is already recorded.
///
/// Failure to write is reported to stderr and otherwise ignored: a read-only checkout or a
/// sandboxed CI runner should not turn a legitimate test failure into a confusing I/O panic on
/// top of it. The property failure is the news; persistence is a convenience.
pub fn save(path: &str, tape: &[u64], name: &str) {
    let encoded = encode(tape);
    if load(path).iter().any(|existing| existing == tape) {
        return;
    }

    if let Some(parent) = Path::new(path).parent()
        && !parent.as_os_str().is_empty()
        && let Err(error) = fs::create_dir_all(parent)
    {
        eprintln!(
            "euclid-property: could not create {}: {error}",
            parent.display()
        );
        return;
    }

    let existing = fs::read_to_string(path).unwrap_or_default();
    let header = if existing.is_empty() {
        "# Minimised counterexamples found by euclid-property.\n\
         # Commit this file: each line is a regression test that cost a random search to find,\n\
         # and the runner replays them before searching for anything new.\n"
    } else {
        ""
    };

    let body = format!("{existing}{header}{encoded} # {name}\n");
    if let Err(error) = fs::write(path, body) {
        eprintln!("euclid-property: could not write {path}: {error}");
    }
}

#[cfg(test)]
mod tests;
