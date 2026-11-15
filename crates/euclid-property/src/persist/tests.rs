//! Encoding, and the robustness the file format needs to survive being hand-edited.

use super::{decode, encode, load, save};

#[test]
fn encoding_round_trips() {
    let tape = vec![0, 1, u64::MAX, 42, 0x1234_5678_9ABC_DEF0];
    let encoded = encode(&tape);
    assert_eq!(decode(&encoded), Some(tape));
}

#[test]
fn an_empty_tape_encodes_to_nothing_and_does_not_decode() {
    assert_eq!(encode(&[]), "");
    // An empty payload is not a case; it must not come back as an empty tape, or a blank line
    // in the file would become a phantom regression that replays every run.
    assert_eq!(decode(""), None);
}

/// The whole point of the robustness: these files are committed, so they get merged, truncated
/// by a bad rebase, and hand-edited. None of that may panic inside the harness — a malformed
/// file means "no regressions", not "crash before running any test".
#[test]
fn malformed_input_decodes_to_none_rather_than_panicking() {
    for bad in [
        "zzzz",
        "abc",                   // not a multiple of 16
        "00000000000000",        // 14 chars
        "000000000000000g",      // non-hex
        "0000000000000000 junk", // trailing garbage
    ] {
        assert_eq!(decode(bad), None, "{bad:?} should not decode");
    }
}

#[test]
fn a_missing_file_loads_as_no_regressions() {
    assert!(load("/nonexistent/euclid-property/does-not-exist.txt").is_empty());
}

#[test]
fn saving_then_loading_returns_the_tape() {
    let dir = std::env::temp_dir().join("euclid-property-persist-roundtrip");
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("cases.txt");
    let path = path.to_string_lossy().to_string();

    let tape = vec![7u64, 8, 9];
    save(&path, &tape, "a_property");
    assert_eq!(load(&path), vec![tape.clone()]);

    // Saving the same tape twice must not duplicate it: a flaky test re-found the same case,
    // and the file should not grow without bound.
    save(&path, &tape, "a_property");
    assert_eq!(load(&path), vec![tape.clone()]);

    // A different tape appends alongside it.
    let other = vec![1u64];
    save(&path, &other, "another_property");
    let loaded = load(&path);
    assert_eq!(loaded.len(), 2);
    assert!(loaded.contains(&tape) && loaded.contains(&other));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn comments_and_blank_lines_are_ignored() {
    let dir = std::env::temp_dir().join("euclid-property-persist-comments");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir is writable");
    let path = dir.join("cases.txt");

    std::fs::write(
        &path,
        "# a header comment\n\
         \n\
         0000000000000007 # a_property\n\
         # another comment\n\
         0000000000000001\n",
    )
    .expect("temp file is writable");

    let loaded = load(&path.to_string_lossy());
    assert_eq!(loaded, vec![vec![7u64], vec![1u64]]);

    let _ = std::fs::remove_dir_all(&dir);
}

/// An unwritable path must not panic. A sandboxed CI runner or a read-only checkout should
/// report the real test failure, not an I/O panic layered on top of it.
#[test]
fn an_unwritable_path_is_survivable() {
    save(
        "/proc/euclid-property-cannot-write-here.txt",
        &[1, 2, 3],
        "x",
    );
    // Reaching here without panicking is the assertion.
}
