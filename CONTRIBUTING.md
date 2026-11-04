# Contributing to euclid

euclid is the numerics, units, and mathematical base for Urban Fabric. It is a
**beta** repository: the trait surfaces are still moving, and downstream projects
are expected to track them rather than pin to a stable contract.

## Version Control: Jujutsu

**This repository uses [Jujutsu](https://github.com/jj-vcs/jj) (`jj`), not git
directly.** It is a *colocated* repo — `.jj` and `.git` sit side by side — so git
tooling can still read it, but `jj` is the interface you should work through.

One consequence to know up front: in a colocated repo git's `HEAD` is normally
**detached**, because jj owns the working copy. That is expected, not a problem to
fix. Running `git commit` here will create a commit git can see but that sits
outside jj's view of history until the next `jj` invocation imports it, so it is
worth avoiding.

The everyday loop:

```sh
jj status                   # what has changed (jj auto-tracks new files)
jj diff                     # review it
jj commit -m "message"      # commit the working copy and start a fresh change
jj log                      # history
```

Useful beyond that:

```sh
jj describe -m "message"    # set/replace the current change's message
jj new                      # start a new empty change on top
jj edit <rev>               # move the working copy to an existing change
jj squash                   # fold the working copy into its parent
jj undo                     # undo the last jj operation
jj op log                   # every operation, so nothing is truly lost
```

Bookmarks (jj's branches) **do not advance on their own**. After committing, the
`main` bookmark still points where it did, so moving it is a deliberate step:

```sh
jj bookmark set main -r @-  # point main at the change you just committed
jj git push                 # push it
```

Both of those reach outside your machine, so treat them as publish actions rather
than part of the commit loop.

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

All four must be clean.

**A git `pre-push` hook will not help you here.** `jj` does not run git hooks, so
a `.git/hooks/pre-push` script is silently skipped on `jj git push` — exactly when
you would want it. Use a `jj` alias instead, in your own
`~/.config/jj/config.toml`:

```toml
[aliases]
check = ["util", "exec", "--", "sh", "-c", """
  set -e
  cargo fmt --all --check
  cargo test --all-targets --workspace
  cargo clippy --all-targets --workspace -- -D warnings
  cargo test --doc --workspace
"""]
```

Then `jj check` before `jj git push`. That lives in your user config rather than
the repo, so it stays a personal habit rather than a project mandate — and it
costs nothing but your own CPU.

Lints are declared once in `[workspace.lints]` at the root and inherited by every
crate, so `clippy -- -D warnings` is the same gate everywhere. `unsafe_code`,
`unwrap_used`, `expect_used`, and `panic` are all denied; reference code in tests
is held to the same bar.

## No CHANGELOG

**There is no CHANGELOG, by choice.** While euclid is beta, the version number
and the `jj log` are the record. Maintaining a hand-written changelog for a
surface that is still being reshaped costs more than it informs, and a stale
changelog is worse than none.

Instead:

- Put the explanation in the **commit message** (`jj describe`) and the **pull
  request body**. That is where a breaking rename or a changed numeric convention
  gets described.
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
| `relationships` | do the functions agree *with each other*? |
| `edge_cases` | what happens at `0`, `±∞`, `NaN`, and the representable limits? |
| `conventions` | which convention does an ambiguous std-mirroring method follow? |
| `constants` | are the lifted constants right, on both carriers? |
| `f32_surface` | is the separate single-precision implementation as good? |

`relationships` is the one worth understanding before adding a function. It ties
functions together through the formulas relating them — `ln_gamma' = digamma`,
`P(½, x) = erf(√x)`, Poisson tail `= gamma_ur`, binomial CDF `= beta_reg` — so
each relation constrains every function it mentions, and the implementations on
either side share no code. A new function should gain at least one edge in that
web: a derivative, an integral, a closed form at special parameters, or a
discrete↔continuous bridge. One edge there is worth more than a dozen sampled
reference values.

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
