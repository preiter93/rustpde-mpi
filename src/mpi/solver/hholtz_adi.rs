//! # Helmoltz Solver with mpi support
//!
//! Input: x pencil distributed data in spectral space.
//!
//! # Examples
//! `examples/solve_hholtz_mpi.rs`
//!
//! # Description
//!  Solve equations of the form:
//!
//!  (I-c*D2) vhat = f
//!
//! where D2 is the second derivative.
//! Alternatively, if defined, multiply rhs
//! before the solve step, i.e.
//!
//!  (I-c*D2) vhat = A f
//!
//! For multidimensional equations, apply
//! alternating-direction implicit method (ADI)
//! to solve each dimension individually. But take
//! in mind that this method introduces a numerical
//! error, which is large if *c* is large.
//!
//! Chebyshev bases: The equation becomes
//! banded after multiplication with the pseudoinverse
//! of D2 (B2). In this case, the second equation is
//! solved, with A = B2.
use crate::bases::BaseSpace;
use crate::mpi::{BaseSpaceMpi, Equivalence, FieldBaseMpi};
use crate::solver::{Fdma, MatVec, MatVecFdma, Solve, SolveReturn, Solver, SolverScalar};
use ndarray::{Array2, ArrayBase, Data, DataMut, Ix2};
use std::ops::{Add, Div, Mul};
/// Container for `HholtzAdi`
/// Solve with mpi support. Input must be x pencil distribution.
#[derive(Clone)]
pub struct HholtzAdiMpi<T, S, const N: usize> {
    solver: Vec<Solver<T>>,
    matvec: Vec<Option<MatVec<T>>>,
    space: S,
}

impl<T2, S, const N: usize> HholtzAdiMpi<f64, S, N>
where
    S: BaseSpace<f64, N, Physical = f64, Spectral = T2> + BaseSpaceMpi<f64, N>,
{
    /// Construct Helmholtz solver from field:
    ///
    ///  (I-c*D2) vhat = A f
    pub fn new(field: &FieldBaseMpi<f64, f64, T2, S, N>, c: [f64; N]) -> Self {
        // Gather matrices and preconditioner
        let mut solver: Vec<Solver<f64>> = Vec::new();
        let mut matvec: Vec<Option<MatVec<f64>>> = Vec::new();
        let space = field.space.clone();
        for (axis, ci) in c.iter().enumerate() {
            // Matrices and preconditioner
            let (mat_a, mat_b, precond) = field.ingredients_for_hholtz(axis);
            let mat: Array2<f64> = mat_a - mat_b * *ci;
            let solver_axis = Solver::Fdma(Fdma::from_matrix(&mat));
            let matvec_axis = precond.map(|x| MatVec::MatVecFdma(MatVecFdma::new(&x)));

            solver.push(solver_axis);
            matvec.push(matvec_axis);
        }

        Self {
            solver,
            matvec,
            space,
        }
    }
}

#[allow(unused_variables)]
impl<T, S, A> Solve<A, ndarray::Ix2> for HholtzAdiMpi<T, S, 2>
where
    T: SolverScalar,
    A: SolverScalar
        + Div<T, Output = A>
        + Mul<T, Output = A>
        + Add<T, Output = A>
        + From<T>
        + Equivalence,
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T> + BaseSpaceMpi<f64, 2>,
{
    /// # Example
    fn solve<S1, S2>(
        &self,
        input: &ArrayBase<S1, Ix2>,
        output: &mut ArrayBase<S2, Ix2>,
        axis: usize,
    ) where
        S1: Data<Elem = A>,
        S2: Data<Elem = A> + DataMut,
    {
        // Matvec axis 0
        let rhs = self.matvec[0]
            .as_ref()
            .map_or_else(|| input.to_owned(), |x| x.solve(input, 0));

        // Decomp 1 for mpi communication
        let dcp = self.space.get_decomp_from_global_shape(&[
            self.space.shape_spectral()[0],
            self.space.shape_physical()[1],
        ]);
        let mut buf = Array2::zeros(dcp.y_pencil.sz);
        dcp.transpose_x_to_y(&rhs, &mut buf);

        // Matvec axis 1
        if let Some(x) = &self.matvec[1] {
            buf = x.solve(&buf, 1);
        }

        // Decomp 2 for mpi communication (array size  has changed)
        let dcp = self.space.get_decomp_from_global_shape(&[
            self.space.shape_spectral()[0],
            self.space.shape_spectral()[1],
        ]);
        let mut buf_ypen: Array2<A> = Array2::zeros(dcp.y_pencil.sz);
        let mut buf_xpen: Array2<A> = Array2::zeros(dcp.x_pencil.sz);
        // Solve
        self.solver[1].solve(&buf, &mut buf_ypen, 1);

        // Transpose y->x
        dcp.transpose_y_to_x(&buf_ypen, &mut buf_xpen);
        self.solver[0].solve(&buf_xpen, output, 0);
    }
}
