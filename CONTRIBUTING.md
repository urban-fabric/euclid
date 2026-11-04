# Contributing to euclid

euclid is the numerics, units, and mathematical base for Urban Fabric. It is a
**beta** repository: the trait surfaces are still moving, and downstream projects
are expected to track them rather than pin to a stable contract.

## Checks Run Locally, Not in CI

**There is no CI, by choice.** We do not want to pay for external compute to run
a test suite that takes well under a second on a laptop. The consequence is that
**the checks are your responsibility before you push** — nothing downstream will
catch a regression for you.

Run all four, from the workspace root:

```sh
cargo fmt --all --check
cargo test --all-targets --workspace
cargo clippy --all-targets --workspace -- -D warnings
cargo test --doc --workspace
```

All four must be clean. If you want them enforced mechanically without paying for
anyone's runners, a local pre-push hook is the right place:

```sh
cat > .git/hooks/pre-push <<'EOF'
#!/bin/sh
set -e
cargo fmt --all --check
cargo test --all-targets --workspace
cargo clippy --all-targets --workspace -- -D warnings
cargo test --doc --workspace
EOF
chmod +x .git/hooks/pre-push
```

That runs on your own machine, costs nothing, and is not checked in — so it stays
a personal preference rather than a project mandate.

Lints are declared once in `[workspace.lints]` at the root and inherited by every
crate, so `clippy -- -D warnings` is the same gate everywhere. `unsafe_code`,
`unwrap_used`, `expect_used`, and `panic` are all denied; reference code in tests
is held to the same bar.

## No CHANGELOG

**There is no CHANGELOG, by choice.** While euclid is beta, the version number
and the git log are the record. Maintaining a hand-written changelog for a
surface that is still being reshaped costs more than it informs, and a stale
changelog is worse than none.

Instead:

- Put the explanation in the **commit message** and the **pull request body**.
  That is where a breaking rename or a changed numeric convention gets described.
- Bump the workspace version in the root `Cargo.toml` when the public surface
  moves. All four crates share `[workspace.package]`, so they version together.
- Say so in the PR title when imports or semantics change, since downstream
  projects read titles to decide whether they need to react.

This should be revisited if euclid ever reaches 1.0 and outside consumers depend
on it, but that is not the situation today.

## What a Change Should Come With

euclid ships contracts, so the usual question is not "does the code work" but
"is the contract still implementable and still unambiguous".

- **New or changed trait method** — update the reference implementations that
  must satisfy it. `ToySampler` (`euclid-random`), `ToyTensor` and `Dense`
  (`euclid-tensor`), and the unit carriers in `euclid-units` exist so that every
  contract has at least one implementation proving it can be honored.
- **Anything the contract leaves open** — pick a convention and *write it down*
  in the trait docs, not only in an implementation. Two backends reading the same
  signature differently is the one failure mode that defeats the point of the
  library: the same formula would produce different answers on different
  interfaces.
- **New numeric function** — add it to the accuracy, stability, and identity
  suites in `euclid-numeric/src/tests/`. A function that is merely *plausible* in
  the normal range is not enough; see below.
- **Docs are the product.** These crates are almost entirely trait surface, so a
  method's doc comment is the deliverable. Document panics, saturation and
  rounding behavior, and precision limits. Skip the design-rationale essay unless
  it stops someone from "fixing" something that is deliberate.

## Numerical Work

Numeric changes carry risk that ordinary review does not catch, so they get their
own bar. The suites in `euclid-numeric/src/tests/` are organized by the property
under test:

| suite | question it answers |
|-------|--------------------|
| `accuracy` | how close is this to the true value, across the domain? |
| `stability` | does it survive the inputs where the naive formula fails? |
| `identities` | does it satisfy the mathematical relations it must? |
| `edge_cases` | what happens at `0`, `±∞`, `NaN`, and the representable limits? |

When adding a numeric function, prefer **identity tests** over hand-copied
reference digits where you can: a recurrence, complement, or round-trip relation
validates itself and cannot be transcribed wrongly. Where a reference value is
needed, derive it from an exact closed form or an independent implementation, and
say in a comment where it came from.

Never "fix" a test by loosening a tolerance without understanding why the error
grew. Tolerances in these suites are set from measured error and annotated with
the reason; an unexplained loosening hides the regression it was meant to catch.

## License

By contributing you agree your contributions are licensed under the
[Apache License, Version 2.0](LICENSE), per section 5 of that license.
