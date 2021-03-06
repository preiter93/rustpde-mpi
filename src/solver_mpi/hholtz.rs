//! # Helmoltz Solver with mpi support
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
//! For multidimensional equations, an eigendecomposition
//! is applied to to (n-1) dimensions. See [`crate::solver::FdmaTensor`]
//!
//! Chebyshev bases: The equation becomes
//! banded after multiplication with the pseudoinverse
//! of D2 (B2). In this case, the second equation is
//! solved, with A = B2.
#![allow(clippy::shadow_unrelated)]
use crate::bases::BaseSpace;
use crate::field_mpi::FieldBaseMpi;
use crate::mpi::{BaseSpaceMpi, Equivalence};
use crate::solver::utils::vec_to_array;
use crate::solver::{FdmaTensor, MatVec, MatVecFdma, Solve, SolveReturn, SolverScalar};
use ndarray::{s, Array2, ArrayBase, Ix2, Zip};
use std::convert::Into;
use std::ops::{Add, Div, Mul};

/// Container for Poisson Solver
#[derive(Clone)]
pub struct HholtzMpi<T, S, const N: usize> {
    solver: Box<FdmaTensor<T, N>>,
    matvec: Vec<Option<MatVec<T>>>,
    space: S,
}

impl<S, const N: usize> HholtzMpi<f64, S, N> {
    /// Construct Helmholtz solver from field:
    ///
    ///  (I-c*D2) vhat = A f
    pub fn new<T2>(field: &FieldBaseMpi<f64, f64, T2, S, N>, c: [f64; N]) -> Self
    where
        S: BaseSpace<f64, N, Physical = f64, Spectral = T2>
            + BaseSpaceMpi<f64, N, Physical = f64, Spectral = T2>,
    {
        // Gather matrices and preconditioner
        let mut laplacians: Vec<Array2<f64>> = Vec::new();
        let mut masses: Vec<Array2<f64>> = Vec::new();
        let mut is_diags: Vec<bool> = Vec::new();
        let mut matvec: Vec<Option<MatVec<f64>>> = Vec::new();
        let space = field.space.clone();
        for (axis, ci) in c.iter().enumerate() {
            // Matrices and preconditioner
            let (mat_a, mat_b, precond, is_diag) = field.ingredients_for_poisson(axis);
            let mass = mat_a;
            let laplacian = -1.0 * mat_b * *ci;
            let matvec_axis = precond.map(|x| MatVec::MatVecFdma(MatVecFdma::new(&x)));

            laplacians.push(laplacian);
            masses.push(mass);
            matvec.push(matvec_axis);
            is_diags.push(is_diag);
        }

        // Vectors -> Arrays
        let laplacians = vec_to_array::<&Array2<f64>, N>(laplacians.iter().collect());
        let masses = vec_to_array::<&Array2<f64>, N>(masses.iter().collect());
        let is_diag = vec_to_array::<&bool, N>(is_diags.iter().collect());

        // Solver
        let solver = FdmaTensor::from_matrix(laplacians, masses, is_diag, 1.);

        // let solver = Box::new(solver);
        Self {
            solver: Box::new(solver),
            matvec,
            space,
        }
    }
}

#[allow(unused_variables, clippy::similar_names)]
impl<S, A> Solve<A, ndarray::Ix2> for HholtzMpi<f64, S, 2>
where
    A: SolverScalar
        + Div<f64, Output = A>
        + Mul<f64, Output = A>
        + Add<f64, Output = A>
        + From<f64>
        + Equivalence,
    S: BaseSpace<f64, 2, Physical = f64, Spectral = A>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = A>,
{
    /// # Example
    fn solve<S1, S2>(
        &self,
        input: &ArrayBase<S1, Ix2>,
        output: &mut ArrayBase<S2, Ix2>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = A>,
        S2: ndarray::Data<Elem = A> + ndarray::DataMut,
    {
        // Decomp 1 for mpi communication

        // Matvec axis 0
        let rhs = self.matvec[0]
            .as_ref()
            .map_or_else(|| input.to_owned(), |x| x.solve(input, 0));

        // Transpose data
        //let dcp = self.space.get_decomp_from_x_pencil(rhs.shape());
        let dcp = self.space.get_decomp_from_global_shape(&[
            self.space.shape_spectral()[0],
            self.space.shape_physical()[1],
        ]);
        let mut buf = Array2::zeros(dcp.y_pencil.sz);
        dcp.transpose_x_to_y(&rhs, &mut buf);

        // Matvec axis 1
        if let Some(x) = &self.matvec[1] {
            buf = x.solve(&buf, 1);
        };

        // Decomp 2 for mpi communication (array size  has changed)
        //let dcp = self.space.get_decomp_from_y_pencil(buf.shape());
        let dcp = self.space.get_decomp_from_global_shape(&[
            self.space.shape_spectral()[0],
            self.space.shape_spectral()[1],
        ]);
        let mut buf_ypen: Array2<A> = Array2::zeros(dcp.y_pencil.sz);
        let mut buf_xpen: Array2<A> = Array2::zeros(dcp.x_pencil.sz);
        dcp.transpose_y_to_x(&buf, &mut buf_xpen);

        // Solve fdma-tensor
        let solver = &self.solver;
        // Step 1: Forward Transform rhs along x
        if let Some(p) = &solver.fwd[0] {
            let p_cast: Array2<A> = p.mapv(Into::into);
            output.assign(&p_cast.dot(&buf_xpen));
        } else {
            output.assign(&buf_xpen);
        }
        dcp.transpose_x_to_y(output, &mut buf_ypen);
        let lam_local = solver.lam[0].slice(s![dcp.y_pencil.st[0]..=dcp.y_pencil.en[0]]);
        // Step 2: Solve along y (but iterate over all lanes in x)
        Zip::from(buf_ypen.outer_iter_mut())
            .and(lam_local.outer_iter())
            .for_each(|mut out, lam| {
                let l = lam.as_slice().unwrap()[0] + solver.alpha;
                let mut fdma = &solver.fdma[0] + &(&solver.fdma[1] * l);
                fdma.sweep();
                fdma.solve(&out.to_owned(), &mut out, 0);
            });

        // Step 3: Backward Transform solution along x
        dcp.transpose_y_to_x(&buf_ypen, output);
        if let Some(q) = &solver.bwd[0] {
            let q_cast: Array2<A> = q.mapv(Into::into);
            output.assign(&q_cast.dot(output));
        }
    }

    fn solve_par<S1, S2>(
        &self,
        input: &ArrayBase<S1, Ix2>,
        output: &mut ArrayBase<S2, Ix2>,
        axis: usize,
    ) where
        S1: ndarray::Data<Elem = A>,
        S2: ndarray::Data<Elem = A> + ndarray::DataMut,
    {
        unimplemented!("Parallel solve not implemented!");
    }
}
