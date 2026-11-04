# euclid-random

Explicit-key random sampler contracts. Part of [euclid](../..).

```toml
[dependencies]
euclid-random = "0.3"
euclid-numeric = "0.3"   # for the Real/Integer/Boolean carrier bounds
```

Randomness is modeled with explicit immutable keys. A draw is a pure function of
its key, distribution parameters, and result shape: using the same key twice
produces the same value. New independent randomness comes from `RandomKey::split`
or `RandomKey::fold_in`, which keeps the contract reproducible and friendly to
eager, dataframe, and graph backends alike.

| trait | surface |
|-------|---------|
| `RandomKey` | `key`, `split`, `fold_in` |
| `RandomSampler` | `uniform`, `normal`, `bernoulli`, `randint` over a host-known `shape` |
| `CountSampler` | `poisson`, `binomial`, `beta_binomial` |
| `GammaSampler` | `standard_gamma` |

## Shapes

Scalar backends such as an `f64` sampler use `&[]` as the shape. Array,
dataframe, and graph backends can interpret non-empty shapes in their native
way, which keeps generic sampler code small enough to target JAX-like arrays,
Polars-style expressions, or eager scalar values without separate random APIs.

## Not Here

No concrete PRNG, storage type, tensor runtime, or state representation.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
Copyright 2026 Raymond Mead.
