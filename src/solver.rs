//! # Collection of linear algebra Solver
//!
//! Must be updated ...
#![allow(clippy::module_name_repetitions)]
pub mod fdma;
pub mod fdma_tensor;
pub mod hholtz;
pub mod hholtz_adi;
pub mod matvec;
pub mod pdma_plus2;
pub mod poisson;
pub mod sdma;
pub mod tdma;
pub mod utils;
pub use fdma::Fdma;
pub use fdma_tensor::FdmaTensor;
pub use hholtz::Hholtz;
pub use hholtz_adi::HholtzAdi;
pub use matvec::{MatVec, MatVecDot, MatVecFdma};
use ndarray::{Array, ArrayBase, Data, DataMut};
use num_complex::Complex;
pub use pdma_plus2::PdmaPlus2;
pub use poisson::Poisson;
pub use sdma::Sdma;
pub use tdma::Tdma;
use utils::diag;

/// Combination of linear algebra traits
pub trait SolverScalar:
    ndarray::LinalgScalar
    + std::ops::SubAssign
    + std::ops::DivAssign
    + From<f64>
    + num_traits::Zero
    + std::marker::Copy
    + std::ops::Div
    + std::ops::Sub
    + std::marker::Send
    + std::marker::Sync
{
}
impl<T> SolverScalar for T where
    T: ndarray::LinalgScalar
        + std::ops::SubAssign
        + std::ops::DivAssign
        + From<f64>
        + num_traits::Zero
        + num_traits::One
        + std::marker::Copy
        + std::ops::Div
        + std::ops::Sub
        + std::marker::Send
        + std::marker::Sync
{
}

/// Solve linear algebraix systems of the form: M x = b.
#[enum_dispatch]
pub trait Solve<A, D> {
    /// Solves M x = b, returns x, which is of type A
    /// Output (x) matches input (b) in type and size.
    fn solve<S1, S2>(&self, input: &ArrayBase<S1, D>, output: &mut ArrayBase<S2, D>, axis: usize)
    where
        //A: SolverScalar,
        S1: Data<Elem = A>,
        S2: Data<Elem = A> + DataMut;

    /// Solves M x = b, returns x, which is of type A
    /// Output (x) matches input (b) in type and size.
    ///
    /// Use parallel iteratiors for ndim > 1. Might be
    /// unimplemented for some solvers.
    fn solve_par<S1, S2>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        //A: SolverScalar,
        S1: Data<Elem = A>,
        S2: Data<Elem = A> + DataMut;
}

/// Solve linear algebraic systems of the form: M x = b.
#[enum_dispatch]
pub trait SolveReturn<A, D> {
    /// Solves M x = b, returns x, which is of type A
    /// Output (x) matches input (b) in type and size.
    fn solve<S1>(&self, input: &ArrayBase<S1, D>, axis: usize) -> Array<A, D>
    where
        S1: Data<Elem = A>;

    /// Parallel version of solve. Might be unimplemented.
    fn solve_par<S1>(&self, input: &ArrayBase<S1, D>, axis: usize) -> Array<A, D>
    where
        S1: Data<Elem = A>;
}

/// Collection of Linalg Solver, must work for unlimited number of dimensions
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum Solver<T> {
    /// Single-diagonal Solver
    Sdma(Sdma<T>),
    /// Two-diagonal Solver
    Tdma(Tdma<T>),
    /// Four-diagonal Solver
    Fdma(Fdma<T>),
    /// Penta-diagonal solver (diagonals -2, -1, 0, 1, 2, 3, 4)
    PdmaPlus2(PdmaPlus2<T>),
}

/// Intented to solve field equations (limited number of dimensions)
#[derive(Clone)]
pub enum SolverField<T, const N: usize> {
    /// Helmholtz Type Solver
    Hholtz(Hholtz<T, N>),
    /// Helmholtz Type Solver (Alternatic direction method)
    HholtzAdi(HholtzAdi<T, N>),
    /// Poisson Solver
    Poisson(Poisson<T, N>),
}

impl<T, A, D> Solve<A, D> for Solver<T>
where
    T: SolverScalar,
    A: SolverScalar
        + std::ops::Div<T, Output = A>
        + std::ops::Mul<T, Output = A>
        + std::ops::Add<T, Output = A>
        + From<T>,
    D: ndarray::Dimension + ndarray::RemoveAxis,
{
    fn solve<S1, S2>(&self, input: &ArrayBase<S1, D>, output: &mut ArrayBase<S2, D>, axis: usize)
    where
        S1: Data<Elem = A>,
        S2: Data<Elem = A> + DataMut,
    {
        match self {
            Self::Sdma(ref t) => t.solve(input, output, axis),
            Self::Tdma(ref t) => t.solve(input, output, axis),
            Self::Fdma(ref t) => t.solve(input, output, axis),
            Self::PdmaPlus2(ref t) => t.solve(input, output, axis),
        }
    }

    fn solve_par<S1, S2>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) where
        S1: Data<Elem = A>,
        S2: Data<Elem = A> + DataMut,
    {
        match self {
            Self::Sdma(ref t) => t.solve_par(input, output, axis),
            Self::Tdma(ref t) => t.solve_par(input, output, axis),
            Self::Fdma(ref t) => t.solve_par(input, output, axis),
            Self::PdmaPlus2(ref t) => t.solve_par(input, output, axis),
        }
    }
}

/// *a*: Variable Type of in- and output
///
/// *t*: Variable Type in Solver
///
/// *d*: ndarray's Dimensions (Ix1, Ix2...)
///
/// *n*: usize, number of dimensions (1, 2, ..)
/// must match *d* (redundancy)
macro_rules! derive_solver_enum {
    (
        $i: ident, $a: ty, $t: ty, $d: ty, $n:expr
    ) => {
        impl Solve<$a, $d> for $i<$t, $n> {
            fn solve<S1, S2>(
                &self,
                input: &ArrayBase<S1, $d>,
                output: &mut ArrayBase<S2, $d>,
                axis: usize,
            ) where
                S1: Data<Elem = $a>,
                S2: Data<Elem = $a> + DataMut,
            {
                match self {
                    $i::<$t, $n>::Hholtz(ref t) => t.solve(input, output, axis),
                    $i::<$t, $n>::HholtzAdi(ref t) => t.solve(input, output, axis),
                    $i::<$t, $n>::Poisson(ref t) => t.solve(input, output, axis),
                }
            }

            fn solve_par<S1, S2>(
                &self,
                input: &ArrayBase<S1, $d>,
                output: &mut ArrayBase<S2, $d>,
                axis: usize,
            ) where
                S1: Data<Elem = $a>,
                S2: Data<Elem = $a> + DataMut,
            {
                match self {
                    $i::<$t, $n>::Hholtz(ref t) => t.solve_par(input, output, axis),
                    $i::<$t, $n>::HholtzAdi(ref t) => t.solve_par(input, output, axis),
                    $i::<$t, $n>::Poisson(ref t) => t.solve_par(input, output, axis),
                }
            }
        }
    };
}
// derive_solver_enum!(SolverPoisson, f64, f64, ndarray::Ix1, 1);
// derive_solver_enum!(SolverPoisson, f64, f64, ndarray::Ix2, 2);
derive_solver_enum!(SolverField, f64, f64, ndarray::Ix1, 1);
derive_solver_enum!(SolverField, f64, f64, ndarray::Ix2, 2);
derive_solver_enum!(SolverField, Complex<f64>, f64, ndarray::Ix1, 1);
derive_solver_enum!(SolverField, Complex<f64>, f64, ndarray::Ix2, 2);
