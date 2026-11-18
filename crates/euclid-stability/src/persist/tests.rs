//! Encoding, and the robustness the file format needs to survive being hand-edited.

use super::{Entry, decode, encode, load, load_tapes, render, save};

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
    assert!(load("/nonexistent/euclid-stability/does-not-exist.txt").is_empty());
}

#[test]
fn saving_then_loading_returns_the_tape() {
    let dir = std::env::temp_dir().join(format!(
        "euclid-stability-persist-roundtrip-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("cases.txt");
    let path = path.to_string_lossy().to_string();

    let tape = vec![7u64, 8, 9];
    save(&path, &tape, "a_property", Some("Len(2.5)"));
    assert_eq!(load_tapes(&path), vec![tape.clone()]);
    assert_eq!(load(&path)[0].rendering.as_deref(), Some("Len(2.5)"));

    // Saving the same tape twice must not duplicate it: a flaky test re-found the same case,
    // and the file should not grow without bound.
    save(&path, &tape, "a_property", Some("Len(2.5)"));
    assert_eq!(load_tapes(&path), vec![tape.clone()]);

    // A different tape appends alongside it.
    let other = vec![1u64];
    save(&path, &other, "another_property", Some("Len(0.0)"));
    let loaded = load_tapes(&path);
    assert_eq!(loaded.len(), 2);
    assert!(loaded.contains(&tape) && loaded.contains(&other));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn comments_and_blank_lines_are_ignored() {
    let dir = std::env::temp_dir().join(format!(
        "euclid-stability-persist-comments-{}",
        std::process::id()
    ));
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

    let loaded = load_tapes(&path.to_string_lossy());
    assert_eq!(loaded, vec![vec![7u64], vec![1u64]]);

    let _ = std::fs::remove_dir_all(&dir);
}

/// A line written before renderings were recorded has to keep working. Adoption would otherwise
/// fail every suite with an existing regression file, which is the one moment a harness cannot
/// afford to be strict.
#[test]
fn a_line_without_a_rendering_loads_with_none() {
    assert_eq!(
        super::parse_line("0000000000000007 # a_property"),
        Some(Entry {
            tape: vec![7],
            name: Some("a_property".to_string()),
            rendering: None,
        })
    );
    // As does a bare tape with no comment at all.
    assert_eq!(
        super::parse_line("0000000000000007"),
        Some(Entry {
            tape: vec![7],
            name: None,
            rendering: None,
        })
    );
}

#[test]
fn a_rendering_is_read_back_from_the_comment() {
    assert_eq!(
        super::parse_line("0000000000000007 # a_property :: Dense { shape: [2, 2] }"),
        Some(Entry {
            tape: vec![7],
            name: Some("a_property".to_string()),
            rendering: Some("Dense { shape: [2, 2] }".to_string()),
        })
    );
}

/// The separator is spaced on both sides so a rendering containing a Rust path does not split
/// the line in the wrong place. `Some(Zp::<7>(3))` has a bare `::` in it and must survive.
#[test]
fn a_rendering_containing_a_path_is_not_split_at_the_wrong_place() {
    let entry =
        super::parse_line("0000000000000007 # p :: Some(Zp::<7>(3))").expect("the line parses");
    assert_eq!(entry.rendering.as_deref(), Some("Some(Zp::<7>(3))"));
}

#[test]
fn rendering_collapses_newlines_and_bounds_length() {
    #[derive(Debug)]
    #[allow(dead_code)]
    struct Multiline(&'static str);

    // `{:?}` on a str escapes newlines already, so feed a real one through a Vec's Debug.
    let rendered = render(&vec!["a\nb"]);
    assert!(!rendered.contains('\n'), "{rendered:?} still has a newline");

    // A long value is truncated with an ellipsis rather than written out in full.
    let long: Vec<u64> = (0..500).collect();
    let rendered = render(&long);
    assert!(rendered.ends_with('…'), "{rendered:?} should be truncated");
    assert!(rendered.len() <= super::MAX_RENDERING + 8);
}

/// Truncation must land on a character boundary. Slicing a multi-byte character in half panics,
/// and a `Debug` rendering can easily contain one — euclid's own failure messages use `·`, `⊕`
/// and `π`.
#[test]
fn truncation_does_not_split_a_multibyte_character() {
    let wide: Vec<&str> = std::iter::repeat_n("π⊕·", 200).collect();
    let rendered = render(&wide);
    assert!(rendered.ends_with('…'));
    // Reaching here without panicking is most of the assertion; this confirms it is still valid
    // UTF-8 that round-trips through a string.
    assert_eq!(rendered, rendered.clone());
}

/// An unwritable path must not panic. A sandboxed CI runner or a read-only checkout should
/// report the real test failure, not an I/O panic layered on top of it.
#[test]
fn an_unwritable_path_is_survivable() {
    save(
        "/proc/euclid-stability-cannot-write-here.txt",
        &[1, 2, 3],
        "x",
        Some("x"),
    );
    super::refresh(
        "/proc/euclid-stability-cannot-write-here.txt",
        "x",
        &[(vec![1], None)],
    );
    // Reaching here without panicking is the assertion.
}

/// Refreshing keeps every tape and replaces the renderings, which is what a recording run does
/// to a file whose generator moved on.
#[test]
fn refreshing_rewrites_renderings_and_keeps_tapes() {
    let dir = std::env::temp_dir().join(format!(
        "euclid-stability-persist-refresh-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("cases.txt").to_string_lossy().to_string();

    save(&path, &[7], "p", Some("old"));
    save(&path, &[8], "p", Some("also old"));

    super::refresh(
        &path,
        "p",
        &[(vec![7], Some("new".to_string())), (vec![8], None)],
    );

    let loaded = load(&path);
    assert_eq!(loaded.len(), 2, "both tapes are kept");
    assert_eq!(loaded[0].tape, vec![7]);
    assert_eq!(loaded[0].rendering.as_deref(), Some("new"));
    assert_eq!(loaded[1].tape, vec![8]);
    assert_eq!(loaded[1].rendering, None);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Refreshing must leave **other properties' lines alone.**
///
/// A regressions file holds a whole suite, because the house pattern gives every property in a
/// suite the same path. The first version of `refresh` rewrote the file from one property's view
/// and deleted its siblings' regressions — silently, and permanently, since the whole point of
/// the file is that it is the only record of a case that cost a search to find. Found by running
/// a recording pass over a suite whose three properties shared a file and watching two of them
/// vanish.
#[test]
fn refreshing_preserves_other_properties_entries() {
    let dir = std::env::temp_dir().join(format!(
        "euclid-stability-persist-siblings-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("cases.txt").to_string_lossy().to_string();

    save(&path, &[1], "mine", Some("stale"));
    save(&path, &[2], "sibling", Some("precious"));
    save(&path, &[3], "other_sibling", None);

    super::refresh(&path, "mine", &[(vec![1], Some("fresh".to_string()))]);

    let loaded = load(&path);
    assert_eq!(loaded.len(), 3, "no line may be lost: {loaded:?}");

    let sibling = loaded
        .iter()
        .find(|e| e.belongs_to("sibling"))
        .expect("the sibling's line survived");
    assert_eq!(sibling.rendering.as_deref(), Some("precious"));
    assert_eq!(sibling.tape, vec![2]);

    // The commentless line survives too, name and all absent.
    assert!(loaded.iter().any(|e| e.tape == vec![3]));

    // And this property's own rendering was in fact refreshed.
    let mine = loaded
        .iter()
        .find(|e| e.belongs_to("mine"))
        .expect("our own line is still there");
    assert_eq!(mine.rendering.as_deref(), Some("fresh"));

    let _ = std::fs::remove_dir_all(&dir);
}

/// `belongs_to` is what scopes the rendering check, so it has to be exact rather than
/// permissive — a prefix match would let `a_sweep` claim `a_sweep_is_monotone`'s lines.
#[test]
fn belongs_to_matches_the_whole_name_only() {
    let entry =
        super::parse_line("0000000000000001 # a_sweep_is_monotone :: 3").expect("the line parses");
    assert!(entry.belongs_to("a_sweep_is_monotone"));
    assert!(!entry.belongs_to("a_sweep"));
    assert!(!entry.belongs_to("a_sweep_is_monotone_too"));

    // A line with no name belongs to nobody, so its rendering is never compared.
    let anonymous = super::parse_line("0000000000000001").expect("parses");
    assert!(!anonymous.belongs_to("anything"));
}

/// A rendering exactly at the limit is kept whole.
///
/// The truncation test is `text.len() > MAX_RENDERING`, and relaxing it to `>=` truncates a
/// rendering that fits exactly — replacing its last character with an ellipsis and, worse, making
/// it compare unequal to itself on the next load. That is the staleness check firing on a case
/// nothing changed.
#[test]
fn a_rendering_at_the_limit_is_kept_whole() {
    // `Debug` for a string adds the two quotes, so this renders to exactly MAX_RENDERING.
    let exactly = "a".repeat(super::MAX_RENDERING - 2);
    let rendered = render(&exactly);

    assert_eq!(
        rendered.chars().count(),
        super::MAX_RENDERING,
        "precondition: this case must land exactly on the limit"
    );
    assert!(
        !rendered.contains('…'),
        "a rendering that fits must not be truncated: {rendered}"
    );

    // One character more is truncated, which is the other side of the same boundary.
    let over = "a".repeat(super::MAX_RENDERING - 1);
    assert!(
        render(&over).contains('…'),
        "a rendering over the limit must be truncated"
    );
}

/// Truncation keeps a **prefix**, not just the ellipsis.
///
/// The character-boundary search is `take_while(|i| *i <= MAX_RENDERING)`, and inverting that
/// comparison makes the search stop at the first index — so every over-long case collapses to a
/// bare `…`. The file would still load, every long case would render identically, and staleness
/// detection over them would be gone entirely while the suite stayed green.
#[test]
fn truncation_keeps_a_readable_prefix() {
    let long = "bcdefghij".repeat(200);
    let rendered = render(&long);

    assert!(rendered.ends_with('…'), "should be truncated: {rendered}");
    assert!(
        rendered.starts_with("\"bcdefghij"),
        "the prefix must survive: {rendered}"
    );
    assert!(
        rendered.chars().count() > super::MAX_RENDERING / 2,
        "kept only {} characters of a {}-character rendering",
        rendered.chars().count(),
        long.len()
    );

    // And two long cases differing early still render differently, which is what makes the
    // staleness check mean anything at all for large values.
    let other = format!("z{}", "bcdefghij".repeat(200));
    assert_ne!(rendered, render(&other));
}

/// `refresh` creates its directory, like `save` does.
///
/// The asymmetry the source comment warns about, and mutation testing found it unpinned: every
/// existing test calls `save` first, which creates the directory, so a `refresh` that never
/// creates one passes all of them. Dropping the negation on the emptiness check is enough to
/// skip the creation entirely.
///
/// Checked by refreshing into a directory that does not exist, which is the only arrangement
/// that can tell the two apart.
#[test]
fn refreshing_creates_its_directory_like_saving_does() {
    let dir = std::env::temp_dir().join(format!(
        "euclid-stability-persist-refresh-mkdir-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("cases.txt").to_string_lossy().to_string();
    assert!(
        !dir.exists(),
        "precondition: the directory must be absent so creation is required"
    );

    super::refresh(&path, "p", &[(vec![7], Some("only".to_string()))]);

    let loaded = load(&path);
    assert_eq!(
        loaded.len(),
        1,
        "refresh did not create its directory, so nothing was written"
    );
    assert_eq!(loaded[0].tape, vec![7]);

    let _ = std::fs::remove_dir_all(&dir);
}
