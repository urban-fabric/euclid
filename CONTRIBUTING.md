# Contributing to euclid

euclid is the numerics, units, and mathematical base for Urban Fabric.

**It is `0.x`, and consumers pin a tag.** That is a change of posture as of 0.5:
this file used to say downstream projects "are expected to track them rather
than pin to a stable contract", and the README told them to depend on
`branch = "main"`. Together those meant every consumer silently tracked HEAD, so
a renamed crate or a new required method arrived unannounced in the next
`cargo update` — which is incoherent for a repository whose entire purpose is
that independently-written parts of the fleet agree on definitions.

So the surfaces are still allowed to move, but moving them is now a *release*
rather than a push:

- the **minor number is the breaking-change axis** — `0.5.x` is compatible,
  `0.6.0` may not be, which is cargo's own reading of a `0.x` version
- every breaking change is recorded in [RELEASES](RELEASES.md), which lists
  nothing else, because a pinned consumer has exactly one question
- a release is a tag, cut after the merge — see [Releasing](#releasing)

**Not 1.0, and not soon.** 1.0 is a promise that the surface will not break
without a major bump, and three things have to close first: `euclid-matrix`'s
twelve traits have never been executed by any carrier (no tensor exists here to
satisfy them, and the boundary forbids shipping one), `euclid-quantity` has 270
undocumented public items, and the generated half of the conformance strategy is
still internal to this repository. Pinning works today and is the point; 1.0 is
a separate claim.

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

## Publishing: A Branch and a Squash Merge, Never a Direct Push

**Do not push to `main`.** Push the work to a new branch, open a pull request, and
squash-merge that into `main`.

The reason is the section below: there is no CI here, so the pull request is the
only review gate the project has. A direct push to `main` does not skip a
formality — it skips the only point at which a change is read by someone other
than its author. And because the merge is a *squash*, the PR title and body are
not commentary on the commit, they **become** it: whatever you write there is what
lands on `main` and is all anyone reading `git log` later will have.

Bookmarks (jj's branches) **do not advance on their own**, which works in your
favour here — after `jj commit`, nothing has moved and nothing has left your
machine. Publishing is a separate, deliberate step:

```sh
jj git push --named my-feature=@-   # create the bookmark and push it in one go
```

`--named` creates the remote branch and starts tracking it. Push follow-up commits
on the same branch with `jj bookmark set my-feature -r @-` and then
`jj git push --bookmark my-feature`. If you do not care what the branch is called,
`jj git push -c @-` will name one after the change id.

Then open the PR and squash-merge it on GitHub.

**After the squash merge, reconcile deliberately.** A squash merge puts a *new,
different* commit on `main` — it does not fast-forward to yours, so your local
commits are now redundant copies of work that has already landed under another
hash. Do not merge or rebase them forward:

```sh
jj git fetch                 # pick up the squashed commit on main
jj new main@origin           # start the next change on top of it
jj abandon <your-local-commits>   # drop the now-redundant originals
```

Skipping that is how the same change ends up in the history twice.

Pushing and merging both reach outside your machine, so treat them as publish
actions rather than part of the commit loop — and run the checks below first.

## Checks Run Locally, Not in CI

**There is no CI, by choice.** We do not want to pay for external compute to run
a test suite that takes well under a second on a laptop. The consequence is that
**the checks are your responsibility before you push** — nothing downstream will
catch a regression for you.

Run all six, from the workspace root:

```sh
cargo fmt --all --check
cargo test --all-targets --workspace --all-features
cargo clippy --all-targets --workspace --all-features -- -D warnings
cargo test --doc --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items --all-features
cargo deny check
```

All six must be clean.

`--all-features` is not optional padding. The `conformance` suites are behind a
feature so a consumer pays nothing for them, which means the default build never
compiles that code and never runs its doctests — a broken law or a stale link in
a conformance module would sail through every other check.

The last one is not redundant with `cargo test --doc`: that runs doctests, while
this resolves **intra-doc links**. Nothing else catches a `[`foo`](path)` whose
path has gone stale, and a broken link is worse than a missing one — it silently
points somewhere plausible. `--document-private-items` is needed because most of
`euclid-numeric` is `pub(crate)`, so the default doc build never looks at it.

`cargo deny check` is the sixth and the only one that looks *outside* the
workspace — advisories, licences, the shape of the dependency graph, and where
each crate came from. It is configured in `deny.toml`, and three parts of it are
worth knowing before a dependency change:

- **Licences are allow-listed, permissive only.** A copyleft dependency would
  change the terms of the published artifact, so it fails here rather than being
  noticed at release time. It also catches a *missing* licence: `"Proprietary"`
  is not valid SPDX, so three crates that carried it read as `unlicensed` until
  they were folded onto the workspace's Apache-2.0.
- **Build scripts are allow-listed by crate.** A build script runs arbitrary
  code at compile time, so a new one entering the graph should be looked at
  rather than inherited silently.
- **`[bans].deny` makes the architecture enforceable.** `nalgebra`, `ndarray`,
  `num-traits`, `uom`, `proptest` and `rand` are each listed with the euclid
  crate that exists instead of them. The point is not that they are bad crates —
  it is that reaching for one should be a conversation rather than an accident,
  and a comment in a README cannot stop a transitive pull.

It needs the advisory database, so it is the one check that wants a network.

**A git `pre-push` hook will not help you here.** `jj` does not run git hooks, so
a `.git/hooks/pre-push` script is silently skipped on `jj git push` — exactly when
you would want it. Use a `jj` alias instead, in your own
`~/.config/jj/config.toml`:

```toml
[aliases]
check = ["util", "exec", "--", "sh", "-c", """
  set -e
  cargo fmt --all --check
  cargo test --all-targets --workspace --all-features
  cargo clippy --all-targets --workspace --all-features -- -D warnings
  cargo test --doc --workspace --all-features
  RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items --all-features
  cargo deny check
"""]
```

Then `jj check` before `jj git push`. That lives in your user config rather than
the repo, so it stays a personal habit rather than a project mandate — and it
costs nothing but your own CPU.

**Note the `--all-features` on all four cargo commands.** The alias carried the
flagless form for a while, which made it a weaker gate than the check set above
rather than a convenience wrapper around it: the `conformance` modules are
feature-gated, so a run without the flag never compiles the laws and reports a
clean pass over code it did not look at. If you keep your own variant, keep the
flag.

Lints are declared once in `[workspace.lints]` at the root and inherited by every
crate, so `clippy -- -D warnings` is the same gate everywhere. `unsafe_code`,
`unwrap_used`, `expect_used`, and `panic` are all denied; reference code in tests
is held to the same bar.

## Releasing

A release is a **tag**, and consumers pin it. Three steps, in this order:

1. **Bump** `version` in `[workspace.package]` at the root, in its own commit or
   as part of the change that broke something. All ten crates share it, so they
   version together. Minor for a breaking change, patch otherwise — see
   [RELEASES](RELEASES.md#what-counts-as-breaking) for where that line is, and
   note it is wider than "the signature changed".
2. **Add the entry to [RELEASES](RELEASES.md)** in the same change, while you
   still remember what breaks and what a consumer should do about it. An entry
   written a week later is an entry written from the diff.
3. **Merge, then tag.** The tag goes on `main` *after* the squash merge, not on
   your local commit — the squash creates a new commit that does not exist until
   the merge, so a tag cut beforehand points at something that never lands:

   ```sh
   jj git fetch
   jj bookmark set v0.6.0 -r main@origin   # or: git tag v0.6.0 origin/main
   jj git push --bookmark v0.6.0
   ```

   Getting this backwards is the one mistake the ordering exists to prevent, and
   it is quiet: the tag resolves, the code compiles, and it is simply not the
   code that is on `main`.

Then update the version in the README's [Using It](README.md#using-it) block, so
the documented pin is the one that exists.

### What a release costs everyone else

Worth holding in mind while deciding whether something is worth breaking, because
euclid is **layer 1** of a stratified fleet — see
[euclid is layer 1](README.md#euclid-is-layer-1) for the map.

Being the floor means a breaking change here is the most expensive kind that
exists in the fleet: it is the only one every other repository has to absorb, and
they have to absorb it *in layer order*, because cargo resolves one version of a
package per build graph. Nothing above can move until everything below it has.

**That is six sequential migrations**, on the current graph — layer 2 (`bayes`,
`hausdorff`, `cayley`), then `abel`, then `sierpinski` and `galois`, then
`dirichlet` and `weierstrass`, then `cantor`. Not six repositories; six *waves*,
each of which cannot start until the previous one lands.

The current chain is not hypothetical. `galois` is pinned to euclid `0.3.0` and
its own manifest records the migration as "blocked outside this repository":
`hausdorff-base` requires `^0.3.0`, so bumping `galois` alone fails resolution on
hausdorff's requirement rather than on its own. That is the layering working as
intended — the failure is loud and at the bottom — but it means a euclid release
is the start of a multi-repository sequence rather than an event.

The practical consequence for this repository: **prefer additive change, and
batch what must break.** Two breaking releases cost the fleet two six-wave
propagations; one release carrying both costs it one. That asymmetry is the
strongest argument available for holding a breaking change back until there is
another to travel with it.

### No CHANGELOG — RELEASES is not one

The distinction is deliberate rather than pedantic. A changelog lists what
changed; `RELEASES.md` lists only what **breaks**, because the audience is a
consumer pinned to an older tag and their question is "what do I have to fix".
Everything additive, every fix and every speedup stays out of it, and lives
where it already lived:

- The **pull request body**, which the squash merge turns into the commit message
  on `main`. Write it as the permanent record it becomes, not as a note to the
  reviewer.
- The **PR title**, which is the one line that survives into `jj log`. Say in it
  when imports or semantics change, since downstream reads titles to decide
  whether to react.

Maintaining a full hand-written changelog beside those would duplicate them and
go stale in the copy nobody reads, which is why this stops at the breaking
changes — the one thing the PR body cannot serve, because a consumer should not
have to read six months of commits to find the three that affect them.

## What a Change Should Come With

euclid ships contracts, so the usual question is not "does the code work" but
"is the contract still implementable and still unambiguous".

**Before adding an implementation, ask whether it needs a data structure.**
euclid implements over universal datatypes — `f64`, `f32`, the integers, `bool`,
and newtypes over them — because those mean the same thing everywhere and there
is nothing to design. It ships no data structure, because a tensor or a graph
cannot be written without first choosing dense or sparse, owned or borrowed,
host or device, and those choices are what a backend is for.

A new `erf` belongs here. A new `cholesky` does not. If your change requires a
`Vec`, an index, or a layout decision, it is probably a backend's. See the
README's [data-structure section][ds].

[ds]: README.md#the-line-is-the-data-structure

- **New or changed trait method** — update the reference implementations that
  must satisfy it. `ToySampler` (`euclid-random`), `ToyTensor` and `Dense`
  (`euclid-tensor`), and the toy unit carriers in `euclid-units` exist so that
  every contract has at least one implementation proving it can be honored.

  **One contract has none, for a reason that is now principled rather than
  awkward: `euclid-matrix`.** Its twelve traits are backend primitives — a
  Cholesky, a QR, an eigendecomposition — so the only thing that could satisfy
  them is a concrete tensor, and euclid deliberately ships none. The matrix
  kernels it used to carry were deleted in 0.4 rather than given a reference
  carrier here, because that would have meant implementing Cholesky, LU and QR
  for a tensor *inside* euclid.

  The laws for those traits *are* written, in `euclid-matrix`'s `conformance`
  module — `solve_spd(Σ, Σ·X)` must recover `X`, `Q·R` must reconstruct `A`, and
  so on. They can only be **run** by a backend, so unlike the other five modules
  this one is gated on `feature` alone rather than `any(feature, test)`: there is
  no test here for the second arm to serve. `--all-features` type-checks them,
  which for a generic trait surface is a real check, but no law body executes in
  this repository and none can.
- **A law the trait implies** — add it to that crate's `conformance` module, not
  only to a test. A tier is a *claim*: `impl Semiring for MyThing {}` asserts
  that `⊗` distributes over `⊕`, and nothing in the type system checks it. The
  conformance suites are public, feature-gated functions that take a sample of a
  carrier and assert the laws, so an implementor in another repository can ask
  whether they honored the contract:

  ```rust
  euclid_algebra::conformance::semiring_laws(&my_sample, "MyThing");
  ```

  This is the direct defense against the failure mode named below, and it is why
  the laws live in `src/` rather than in `tests/` — a law in a test protects only
  us. **All six contract crates have them**, so a new law goes into an existing
  module rather than starting one.

  Write laws as **relations between results**, never against transcribed
  reference values: `a ⊕ 0 = a` cannot be mistyped into passing, where a
  hard-coded product can.

  **Run every law over both a fixed and a generated sample.** These are two
  different questions and each crate's suites now ask both:

  - The **fixed** sample is the `ConformanceSample` a carrier declares, and it is
    what proves the tier is *satisfiable* and that our own carriers satisfy it.
    It is also the reproducible one — a failure is a value in the source, with no
    seed to recover — which is why the laws themselves take a slice and why
    `conformance`'s own docs argue for it. An algebraic law over a small carrier
    either holds structurally or fails on the first awkward triple.
  - The **generated** sample is a companion suite beside it, written with
    `euclid-stability` (the dev-only engine; take it under `[dev-dependencies]`,
    as eight of the ten members now do). It takes the *same* law functions and
    changes only where the slice comes from. Nothing is restated, so a law cannot
    drift between the two. What it adds is the thing a fixed list cannot cover:
    the carrier's **representation**, which is a large state space rather than a
    structural claim. `Rational` staying reduced
    through arithmetic, `Zp`'s Fermat inverse agreeing with division at every
    residue, a tensor permutation at *repeated extents* where the wrong direction
    still produces the right shape, a key mixing function decorrelating the high
    bits as well as the low ones.

  The reproducibility objection to generating is answered by the engine rather
  than by avoiding it: pass `.regressions("…/<suite>.txt")` in the `Config`, and
  a minimised counterexample is persisted as a line of hex that replays exactly,
  on any machine, before any random case runs. **Commit that file** — each line
  is a regression test that cost a search to find.

  Each line carries the case it rendered as, after a `::`, and that is checked on
  load rather than being a comment. A tape is a sequence of *choices*, not a
  value, so it only means anything for the generator that consumed it: widen a
  range or add a `weighted` arm and every committed tape still decodes, into a
  different case. The line stays, the suite stays green, and the regression
  silently stops being tested. A mismatch is reported instead, and
  `EUCLID_STABILITY_RECORD=1` refreshes the file — after which the tape pins a
  *different* case than the one originally found, so delete the line and assert
  the case directly if it was the case that mattered.

  Put a coverage floor on the cases that motivated the suite — `.require("a
  repeated-extent tensor", 10.0)` — so a generator that stops producing them
  fails loudly instead of reporting a clean pass over input it never reached.
  Note that a floor applies to every property sharing the `#![config(...)]`
  block, so a property that does not apply the label belongs in its own block.

  The two suites together are what caught `structural_laws` building a malformed
  3-cycle at rank ≥ 4 — correct for as long as the only sample reaching that
  branch was rank 3, and wrong for every backend with a rank-4 tensor. A law is
  code, and a public law that no carrier here can execute is the easiest code in
  the repository to get wrong.
- **Anything the contract leaves open** — pick a convention and *write it down*
  in the trait docs, not only in an implementation. Two backends reading the same
  signature differently is the one failure mode that defeats the point of the
  library: the same formula would produce different answers on different
  interfaces.
- **New numeric function** — add it to the accuracy, stability, and identity
  suites in `euclid-numeric/src/tests/`. A function that is merely *plausible* in
  the normal range is not enough; see below.

  **Give its error a score and a budget.** This is the one habit that separates a
  suite which notices accuracy changing from one that does not, and it is cheap:

  ```rust
  Config::default()
      .regressions("tests/regressions/<suite>.txt")
      .budget("tests/budgets.txt")       // and call target(score) in the body
  ```

  The reasoning is that an assertion and a measurement are different statements.
  `check_ulps!(got, want, 8)` is the **contract** — what a caller may rely on —
  and it is written with headroom, so it admits today's error and a worse one.
  `target(score)` makes the search hunt the worst case instead of sampling
  uniformly, which matters because a kernel's worst input is a thin region that
  uniform draws step over. `.budget(path)` then records that worst score and
  fails a later run that scores **higher**, which is the only check here that can
  see a kernel becoming twice as inaccurate while every assertion still holds.

  Score **error, not accuracy** — higher must mean worse — and prefer a *ratio*
  to an admissible bound over a raw error: `relation::forward_stable_margin`
  returns one, `1.0` means "exactly at the limit", and it is comparable across
  kernels and continuous in the input, which is what the hill climb needs to
  follow. Baselines are reproducible because the seed is fixed; editing a
  generator, `cases` or `targeted_fraction` legitimately changes the number, so
  re-record with `EUCLID_STABILITY_RECORD=1` and commit it. The failure message
  says as much, because the alternative is a reader concluding they broke a
  kernel when they had widened a range.

  Where there is a more accurate reference to compare against, state the claim as
  stability rather than as a tolerance: `relation::forward_stable` bounds the
  error by `safety · κ(x) · u`, so the assertion becomes "no worse than the
  problem itself" instead of "within a number I picked" — and its failure reports
  κ, which is what distinguishes an unstable kernel from a genuinely hard input.
  For an `f32` kernel beside an `f64` one, `relation::precision_agrees` compares
  in `f32` ulps against the rounded wide result, which is the only bound that
  does not charge the narrow kernel for narrowing.
- **Docs are the product.** These crates are almost entirely trait surface, so a
  method's doc comment is the deliverable. Document panics, saturation and
  rounding behavior, and precision limits. Skip the design-rationale essay unless
  it stops someone from "fixing" something that is deliberate.

## Numerical Work

Numeric changes carry risk that ordinary review does not catch, so they get their
own bar. The suites in `euclid-numeric/src/tests/` are organized by the property
under test rather than by the function under test, so one function is exercised
from several directions at once.

**The module header of `euclid-numeric/src/tests.rs` is the authoritative
index** — every suite, what question it answers, and the house rules on file size
and tolerances. Read it before adding a suite, and update it when you do. It
lives there rather than here because a list in two places is a list that goes
stale in one of them.

(Note the path: the workspace denies `clippy::mod_module_files`, so a module with
children is `foo.rs` beside `foo/`, never `foo/mod.rs`. That applies to the test
tree like everywhere else.)

Two rules from that header are worth repeating, because they are easy to breach
without noticing:

- **A test belongs beside the implementation** (a `#[cfg(test)]` child module)
  only when it needs private series, quadrature or crossover constants that the
  `tests/` tree cannot see. Anything reachable through the trait surface belongs
  in `tests/`, where it keeps passing when the algorithm behind it is replaced.
- **No file carrying tests goes over 400 lines.** Split by sub-topic into a
  directory whose header explains the split. Pure trait or impl surface is exempt,
  having no sub-topics to split along.

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

**There is no longer a per-crate exception to check.** `euclid-algebra`,
`euclid-scalar` and `euclid-matrix` each declared `license = "Proprietary"`
until 0.4, on the reasoning that a move should not relicense code and neither
should a split. That has been overridden deliberately by the copyright holder:
every member now takes `license.workspace = true`, so whichever crate you are
touching, the terms are Apache-2.0. See the README's
[License](README.md#license) section for what `cargo deny` made of the old
value — `"Proprietary"` is not valid SPDX, so those crates read as *unlicensed*
rather than as restrictively licensed.
