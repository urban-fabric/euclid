//! Execution contracts: tracing, parameters, and compilation.
//!
//! euclid's value constructors are *context-free* —
//! [`Real::scalar`](crate::numeric::Real::scalar),
//! [`Matrix::identity`](crate::array::Matrix::identity), and
//! [`Tensor::from_host`](crate::array::Tensor::from_host) take no builder argument. An eager
//! carrier (the `f64` reference scalars) evaluates them on the spot; a *deferred* carrier — one
//! that records a computation graph to compile and run later (an XLA frontend, a graph IR, a
//! JIT) — needs somewhere to emit them. These traits name that machinery so a deferred
//! implementation is *described by the contract* rather than reinvented privately, and a
//! generic algorithm can be written once against "open a tracing scope, build a graph of
//! parameter inputs, compile it, run it many times".
//!
//! The surface is opt-in and, like [`MatrixDecompose`](crate::array::MatrixDecompose), ships no
//! reference implementation: eager carriers simply do not implement it. The split mirrors how
//! the work divides — [`Traced`] is the construction context, [`Parameter`] is the one piece
//! that lives on the tensor (a graph input), and [`Compile`]/[`Compiled`] are the
//! build-once / run-many halves of execution.

use crate::{array::Tensor, DType};

/// A construction context for a deferred carrier.
///
/// [`scope`](Traced::scope) installs the place that context-free constructors emit into for the
/// duration of a closure — the standard "tracing context" pattern. An eager carrier never needs
/// one (it has no graph to attach nodes to); a deferred carrier without an active scope has
/// nowhere to put a freshly constructed node.
pub trait Traced {
    /// The tensor handle constructed inside a [`scope`](Self::scope).
    type Tensor: Tensor;

    /// Run `f` with a fresh construction context installed under the debug label `name`,
    /// returning its result. Handles constructed inside the closure remain valid after the
    /// scope ends, so a returned tensor can be compiled or run later.
    fn scope<R>(&self, name: &str, f: impl FnOnce() -> R) -> R;
}

/// A tensor that can stand in as a graph *input*: a placeholder bound to a concrete buffer at
/// execution time, the `index`-th input of the compiled computation.
///
/// This is the deferred counterpart of [`Tensor::from_host`], whose data is baked into the
/// graph; a parameter's data is fed at run time instead, which is what lets one compiled graph
/// serve many inputs. Eager carriers have nothing to bind and do not implement it — the one
/// piece of the execution surface that lives on the tensor rather than on its context.
pub trait Parameter: Tensor {
    /// A parameter (graph input) node of the given `shape`, fed the `index`-th buffer at run
    /// time.
    fn parameter(index: usize, shape: &[usize]) -> Self;
}

/// Optional typed parameter construction for deferred tensor carriers.
///
/// This is the dtype-aware sibling of [`Parameter`]. It only names graph input metadata; it does
/// not prescribe host buffers or execution.
pub trait TypedParameter: Parameter {
    /// A dtype-explicit graph input node of the given `shape`, fed the `index`-th buffer at run
    /// time.
    fn parameter_with_dtype(index: usize, shape: &[usize], dtype: DType) -> Self;
}

/// A [`Traced`] context that compiles a traced graph into a reusable executable, so repeated
/// evaluation pays for execution alone — no retracing, no recompilation.
pub trait Compile: Traced {
    /// The reusable executable produced by [`compile`](Self::compile).
    type Compiled: Compiled;

    /// Why a compilation failed — e.g. a shape or type mismatch among the
    /// [`parameter`](Parameter::parameter) inputs, or an unsupported operation.
    type Error;

    /// Compile the computation whose output roots are `roots` — which must share one tracing
    /// scope — into a reusable [`Compiled`](Self::Compiled). The output order matches `roots`.
    fn compile(&self, roots: &[&Self::Tensor]) -> Result<Self::Compiled, Self::Error>;
}

/// A compiled computation, executed many times with flat input buffers.
pub trait Compiled {
    /// Why an execution failed — e.g. a device error or an input-shape mismatch.
    type Error;

    /// Execute with one flat row-major buffer per [`parameter`](Parameter::parameter) index
    /// (in index order), returning one flat row-major buffer per output root (in root order).
    fn run(&self, inputs: &[&[f64]]) -> Result<Vec<Vec<f64>>, Self::Error>;

    /// The input shapes, ordered by parameter index.
    fn input_shapes(&self) -> &[Vec<usize>];

    /// The output shapes, in root order.
    fn output_shapes(&self) -> &[Vec<usize>];
}
