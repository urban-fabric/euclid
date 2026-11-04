# euclid-tensor

The rank-graded array contract. Part of [euclid](../..).

```toml
[dependencies]
euclid-tensor = "0.3"
euclid-numeric = "0.3"   # for the element trait bounds
```

The array surface is a single rank-graded trait, `Tensor`. A vector is a rank-1
tensor and a matrix a rank-2 tensor; there are no separate `Vector`/`Matrix`
contracts. A tensor interacts only with other tensors and with the scalar
`euclid-numeric` surface beneath it.

## Structural Core

`Tensor` starts at `Clone`, so structural operations — shape, reshape,
indexing/gather, control flow — work for real, complex, integer, natural, and
boolean tensors alike. Arithmetic methods add only the bound they need:

| method | requires |
|--------|----------|
| `sum`, `matmul` | `Ring` |
| `prod` | `Semiring` |
| `min`, `max` | `Real` |
| `mean` | `NormedDivisionAlgebra` |
| `any`, `all` | `Boolean` |

## Capability Extensions

Opt-in traits that grow the surface an implementation can offer without widening
the core:

| trait | adds |
|-------|------|
| `TensorReduce` | statistical and scan reductions — `var`, `std`, `logsumexp`, `softmax`, `log_softmax`, `cumsum` |
| `TensorSort` | order statistics — `sort`, `argsort`, `argmax`, `argmin`, `top_k` (index results use `Tensor::Index`) |
| `TensorContract` | general contraction — `dot_general` and the `einsum1`/`einsum2` forms |
| `TensorIndexing` | range and categorical encoding — `iota`, `one_hot`, `bincount` |
| `TensorTransform` | structural conveniences — `roll`, `tile`, `clip` |
| `TensorWindow` | windowed reductions and pooling — `reduce_window_sum`/`_max`/`_mean`, `max_pool2d_nchw`, `avg_pool2d_nchw` |
| `TensorConv` | `conv_general_dilated`, `conv2d_nchw` |
| `TensorFft` | `fft` over `FftKind`, with `fft_forward`/`ifft`/`rfft`/`irfft` forms |

## Metadata

The `spec` module carries backend-neutral metadata: `DType`, `TensorSpec`, and
the `HasTensorSpec`/`Cast` inspection traits. It is pure description — it names
shape and dtype without owning buffers, devices, or execution, and inspection
never implies host materialization.

## Not Here

No concrete tensor type. Axis naming, mask representation, broadcast/gather/
scatter details, and precision policy are outside this contract; consumers
provide implementations for the carrier types they want to use.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
Copyright 2026 Raymond Mead.
