# euclid-algebra

**The abstract-algebra tower: which structure a carrier's `⊕` and `⊗` actually
satisfy.** Part of [euclid](../..).

```toml
[dependencies]
euclid-algebra = "0.3"
```

This crate has **no dependencies at all**. Algebraic structure needs nothing
from analysis — it says which tier a carrier reaches, not what a formula may
compute.

## Two layers

**The trunk** is method-based. `AdditiveMagma::add` and
`MultiplicativeMagma::mul` are *methods*, not operators, because one carrier has
to host several semirings at once — min-plus shortest paths, boolean
reachability, counting — and none of those could overload `+`:

```rust
use euclid_algebra::Semiring;

// One routine, any semiring: ordinary arithmetic, min-plus, or boolean.
fn fold<T: Semiring>(xs: &[T]) -> T {
    xs.iter().copied().fold(T::ZERO, |acc, x| acc.add(x))
}
```

The hierarchy is property-driven and blanket-implemented, so a carrier gets its
tier by declaring which laws hold rather than by naming the tier:

| layer | traits |
|---|---|
| operations | `AdditiveMagma`, `MultiplicativeMagma`, and the `Ref*` variants for non-`Copy` carriers |
| identities | `HasZero`, `HasOne` |
| laws (markers) | `AssociativeAddition`, `CommutativeAddition`, `IdempotentAddition`, `AssociativeMultiplication`, `CommutativeMultiplication`, `Distributive`, `ZeroAnnihilatesMultiplication`, `MonotoneMultiplication` |
| structures | `AdditiveSemigroup` → `AdditiveMonoid` → `AdditiveBand` → `AdditiveSemilattice`, `MultiplicativeSemigroup` → `MultiplicativeMonoid`, then `Semiring` → `CommutativeSemiring` / `IdempotentSemiring` → `Dioid`, `OrderedSemiring`, `StarSemiring` |

**The numeric branch** is operator-based. `Ring` → `Field` add what a semiring
lacks — additive inverses and division — as `std::ops`, and bind `+ *` as
operators too, so numeric and matrix code reads naturally. A numeric type bridges
the layers in a line: its `AdditiveMagma::add` *is* `+`.

`f64`/`f32` climb the whole chain `Semiring → Ring → Field`; an idempotent
carrier like min-plus stops at `Dioid`, having no additive inverse.
`OrderedField` and `AlgebraicallyClosedField` top it out.

## Exact carriers

Concrete number systems that are pure algebraic structure — no analysis, no
floating point:

| type | tier |
|---|---|
| `Rational`, `Rational64`, `Rational128` | `OrderedField` |
| `Zn<N>` | `CommutativeRing` — not a field unless the modulus is prime |
| `Zp<P>` | `Field` |
| `GaussianRational` | `Field` — ℚ(i) |
| `Integer` | `EuclideanDomain` |

## Not Here

**Analysis** — `exp`, `ln`, `conj`, the real transcendentals — is a separate
concern, and lives in `euclid-numeric` as a contract over any carrier.

**The concrete number line** is a third concern again: `euclid-scalar` is where
this tower's `OrderedField`/`AlgebraicallyClosedField` fuse with that analysis
surface into the `Copy` `Real` and `Complex` of `f64`/`f32`.

Until 0.3 this crate held the tower *and* that concrete tier, which forced a
dependency on `euclid-numeric` plus `LazyReal`/`LazyComplex` import aliases to
dodge a name collision with its own `Real`/`Complex`. Splitting the tier out
removed the dependency, the aliases, and the collision together.

## License

euclid-algebra predates the fold into euclid and keeps its own `Proprietary`
licence rather than inheriting the workspace's Apache-2.0 — a move should not
relicense code.
