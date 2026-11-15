//! Persisting a minimised tape, so a found counterexample is kept.
//!
//! A random search that finds a bug, shrinks it to four entries, prints it and then forgets it
//! has done half a job: the next run has to get lucky again. These files turn each find into a
//! permanent, zero-cost regression test — the runner replays them before it starts searching,
//! so the known-bad cases are checked first and in order.
//!
//! The format is one line per case, `<hex> # <name> :: <value>`, which is deliberately boring:
//! it diffs cleanly, merges without conflict markers in the middle of a case, and a human can
//! delete a line. These files are meant to be committed.
//!
//! # The rendering is not a comment
//!
//! `<value>` is the `Debug` form of the case the tape produced when it was saved, and it is load
//! bearing in two ways.
//!
//! It makes the file **readable**. A tape is otherwise opaque — that is the acknowledged cost of
//! generating from one — so a reviewer looking at `0000…cccd # dense_satisfies… :: Dense { shape:
//! [1, 1, 1, 1], … }` can see what is pinned without running anything. The hex is the input; the
//! rendering is what it means.
//!
//! More importantly, it is the only way to notice that a tape has gone **stale**. A tape is a
//! sequence of choices, not a value, so it is only meaningful for the generator that consumed it.
//! Widen a range, add a `weighted` arm, reorder two draws, and every committed tape still decodes
//! — into a *different* case. The line stays in the file, the suite stays green, and the
//! regression nobody realises is no longer pinned stops being tested. That is a silent loss of
//! coverage, and silent losses are the ones worth engineering against.
//!
//! So the runner re-renders each tape on load and compares. A mismatch is reported as staleness
//! rather than ignored, with the recorded and current values side by side, and
//! `EUCLID_STABILITY_RECORD=1` refreshes the file. Note what refreshing does and does not do: it
//! makes the file honest again, but the tape now pins a different case than the one that was
//! originally found. If the original case mattered, the line should be deleted and the property
//! given an explicit test instead.
//!
//! A line with no `:: <value>` is from before this was recorded and is accepted as-is — the
//! check degrades to the old behaviour rather than failing a suite on adoption.

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

/// Separates the property name from the rendered case in a line's comment.
///
/// Spaced on both sides so it cannot occur inside a Rust identifier, and so a `Debug` rendering
/// that happens to contain `::` — a path, an enum variant — does not split the line in the wrong
/// place. Only the *first* occurrence is a separator; everything after it is the rendering.
const SEPARATOR: &str = " :: ";

/// One persisted case: the tape, which property saved it, and what it rendered as.
///
/// The name matters because **one file holds several properties.** The house pattern is a
/// `settings(suite)` helper giving every property in a suite the same path, so a file is
/// per-suite rather than per-property. Every property still replays every tape in its file —
/// harmless, and occasionally useful, since a counterexample for one law sometimes breaks a
/// sibling — but a *rendering* may only be compared against the property that recorded it. Two
/// properties with different generators render the same tape differently, and comparing across
/// them reports staleness that is not there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The choice tape, which replays to a case.
    pub tape: Vec<u64>,
    /// The property that saved it, from the line's comment. `None` for a line with no comment.
    pub name: Option<String>,
    /// The `Debug` rendering recorded alongside it, if the line carried one. `None` for a line
    /// written before renderings were recorded, in which case staleness cannot be detected.
    pub rendering: Option<String>,
}

impl Entry {
    /// Whether this entry was recorded by `name`, and so may have its rendering checked.
    #[must_use]
    pub fn belongs_to(&self, name: &str) -> bool {
        self.name.as_deref() == Some(name)
    }
}

/// Every case recorded in `path`. Missing or unreadable file means no regressions.
#[must_use]
pub fn load(path: &str) -> Vec<Entry> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines().filter_map(parse_line).collect()
}

/// Just the tapes, for a caller that does not care what they rendered as.
#[must_use]
pub fn load_tapes(path: &str) -> Vec<Vec<u64>> {
    load(path).into_iter().map(|entry| entry.tape).collect()
}

/// `<hex> # <name> :: <rendering>`, tolerating every part after the hex being absent.
fn parse_line(line: &str) -> Option<Entry> {
    let (payload, comment) = match line.split_once('#') {
        Some((payload, comment)) => (payload, Some(comment.trim())),
        None => (line, None),
    };
    let tape = decode(payload.trim())?;

    let (name, rendering) = match comment {
        None => (None, None),
        Some(comment) => match comment.split_once(SEPARATOR) {
            Some((name, rendering)) => (
                non_empty(name),
                non_empty(rendering).map(|rendering| rendering.to_string()),
            ),
            None => (non_empty(comment), None),
        },
    };

    Some(Entry {
        tape,
        name: name.map(str::to_string),
        rendering,
    })
}

fn non_empty(text: &str) -> Option<&str> {
    let text = text.trim();
    (!text.is_empty()).then_some(text)
}

/// Collapse a `Debug` rendering onto one line and bound its length.
///
/// One line because the format is line-oriented; bounded because a generated `Vec` of tensors can
/// render to kilobytes, and a regression file nobody will open is a regression file that stops
/// being reviewed. The consequence of truncation is that two cases agreeing for the first
/// `MAX_RENDERING` characters compare equal, so staleness detection is slightly weaker for very
/// large values — which is the right trade against an unreadable file.
#[must_use]
pub fn render(value: &impl core::fmt::Debug) -> String {
    let mut text = format!("{value:?}").replace(['\n', '\r'], " ");
    if text.len() > MAX_RENDERING {
        // Truncate on a character boundary, not a byte one: a `Debug` rendering can contain
        // multi-byte characters, and slicing one in half panics.
        let cut = text
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|i| *i <= MAX_RENDERING)
            .last()
            .unwrap_or(0);
        text.truncate(cut);
        text.push('…');
    }
    text
}

/// How much of a rendering to keep on the line.
const MAX_RENDERING: usize = 240;

/// Append `tape` to `path` unless it is already recorded, noting what it rendered as.
///
/// Failure to write is reported to stderr and otherwise ignored: a read-only checkout or a
/// sandboxed CI runner should not turn a legitimate test failure into a confusing I/O panic on
/// top of it. The property failure is the news; persistence is a convenience.
pub fn save(path: &str, tape: &[u64], name: &str, rendering: Option<&str>) {
    let encoded = encode(tape);
    if load(path).iter().any(|existing| existing.tape == tape) {
        return;
    }

    if let Some(parent) = Path::new(path).parent()
        && !parent.as_os_str().is_empty()
        && let Err(error) = fs::create_dir_all(parent)
    {
        eprintln!(
            "euclid-stability: could not create {}: {error}",
            parent.display()
        );
        return;
    }

    let existing = fs::read_to_string(path).unwrap_or_default();
    let header = if existing.is_empty() {
        "# Minimised counterexamples found by euclid-stability.\n\
         # Commit this file: each line is a regression test that cost a random search to find,\n\
         # and the runner replays them before searching for anything new.\n\
         # The text after `::` is the case the tape rendered as, and is checked on load — a\n\
         # mismatch means the generator changed and the tape no longer pins what it used to.\n"
    } else {
        ""
    };

    let comment = match rendering {
        Some(rendering) => format!("{name}{SEPARATOR}{rendering}"),
        None => name.to_string(),
    };
    let body = format!("{existing}{header}{encoded} # {comment}\n");
    if let Err(error) = fs::write(path, body) {
        eprintln!("euclid-stability: could not write {path}: {error}");
    }
}

/// Rewrite `name`'s entries in `path` so their renderings match what their tapes produce now,
/// leaving every other property's lines untouched.
///
/// What `EUCLID_STABILITY_RECORD=1` does to a file whose generator has moved on. The tapes are
/// kept and the stale renderings replaced, which makes the file honest — not which restores the
/// cases it used to pin. See the module header for why those are different.
///
/// **Only `name`'s lines are rewritten.** A file holds a whole suite, so rewriting it from one
/// property's view would delete its siblings' regressions — which is a worse outcome than the
/// staleness being fixed, and is the bug this signature exists to make impossible.
///
/// Recording is still a read-modify-write, so running it with the default parallel test threads
/// can lose an entry to a race between two properties in the same suite. It is a deliberate,
/// occasional action rather than part of a normal run, so the fix is to do it with
/// `--test-threads=1` rather than to add locking here; the reporting path says so.
pub fn refresh(path: &str, name: &str, renderings: &[(Vec<u64>, Option<String>)]) {
    let mut body = String::from(
        "# Minimised counterexamples found by euclid-stability.\n\
         # Commit this file: each line is a regression test that cost a random search to find,\n\
         # and the runner replays them before searching for anything new.\n\
         # The text after `::` is the case the tape rendered as, and is checked on load — a\n\
         # mismatch means the generator changed and the tape no longer pins what it used to.\n",
    );

    // Create the directory for symmetry with `save`. In the runner this cannot matter — refresh
    // only runs when `load` found stale entries, so the file and its directory exist — but an
    // asymmetry between two functions writing the same file is a trap for the next caller, and
    // a sequential property over the pair is exactly what would walk into it.
    if let Some(parent) = Path::new(path).parent()
        && !parent.as_os_str().is_empty()
        && let Err(error) = fs::create_dir_all(parent)
    {
        eprintln!(
            "euclid-stability: could not create {}: {error}",
            parent.display()
        );
        return;
    }

    // Other properties' lines first, verbatim, then this property's with fresh renderings.
    for entry in load(path) {
        if entry.belongs_to(name) {
            continue;
        }
        let _ = writeln!(body, "{} # {}", encode(&entry.tape), comment_for(&entry));
    }
    for (tape, rendering) in renderings {
        let _ = writeln!(
            body,
            "{} # {}",
            encode(tape),
            join_comment(name, rendering.as_deref())
        );
    }

    if let Err(error) = fs::write(path, body) {
        eprintln!("euclid-stability: could not write {path}: {error}");
    }
}

fn comment_for(entry: &Entry) -> String {
    join_comment(
        entry.name.as_deref().unwrap_or_default(),
        entry.rendering.as_deref(),
    )
}

fn join_comment(name: &str, rendering: Option<&str>) -> String {
    match rendering {
        Some(rendering) => format!("{name}{SEPARATOR}{rendering}"),
        None => name.to_string(),
    }
}

#[cfg(test)]
mod tests;
