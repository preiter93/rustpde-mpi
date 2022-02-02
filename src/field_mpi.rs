//! # Multidimensional field of basis functions
//! Wrapper around funspace `Space` fields, plus
//! let field store *n*-dimensional arrays which
//! belong the the physical (v) and spectral (vhat)
//! space.
//!
//! Supports mpi.
//pub mod average;
//pub mod read;
//pub mod write;
pub mod average;
pub mod io;
pub mod io_mpi;
use crate::bases::BaseKind;
use crate::bases::BaseSpace;
pub use crate::mpi::{BaseSpaceMpi, Decomp2d, Space2Mpi, Universe};
use crate::types::FloatNum;
use ndarray::{prelude::*, Data, DataMut, Ix, ScalarOperand};
use num_complex::Complex;
use std::convert::TryInto;

/// Two dimensional Field (Real in Physical space, Generic in Spectral Space)
pub type Field2Mpi<T2, S> = FieldBaseMpi<f64, f64, T2, S, 2>;

/// Field struct with mpi support
///
/// `v`: ndarray
///
///   Holds data in physical space
///
/// `vhat`: ndarray
///
///   Holds data in spectral space
///
/// `v_x_pen`: ndarray
///
///   Holds local data in physical space, distributed as x-pencil
///   across processors
///
/// `v_y_pen`: ndarray
///
///   Holds local data in physical space, distributed as y-pencil
///   across processors
///
/// `vhat_x_pen`: ndarray
///
///   Holds local data in spectral space, distributed as x-pencil
///   across processors
///
/// `vhat_y_pen`: ndarray
///
///   Holds local data in spectral space, distributed as y-pencil
///   across processors
///
/// `x`: list of ndarrays
///
///   Grid points (physical space)
///
/// `dx`: list of ndarrays
///
///   Grid points deltas (physical space)
///
/// `FieldBase` is derived from `SpaceBase` struct,
/// defined in the `funspace` crate.
/// It implements forward / backward transform from physical
/// to spectral space, differentation and casting
/// from an orthonormal space to its galerkin space (`from_ortho`
/// and `to_ortho`).
#[allow(clippy::similar_names)]
#[derive(Clone)]
pub struct FieldBaseMpi<A, T1, T2, S, const N: usize> {
    /// Number of dimensions
    pub ndim: usize,
    /// Space
    pub space: S,
    /// Field in physical space - x pencil
    pub v_x_pen: Array<T1, Dim<[Ix; N]>>,
    /// Field in physical space - y pencil
    pub v_y_pen: Array<T1, Dim<[Ix; N]>>,
    /// Field in spectral space - x pencil
    pub vhat_x_pen: Array<T2, Dim<[Ix; N]>>,
    /// Field in spectral space - y pencil
    pub vhat_y_pen: Array<T2, Dim<[Ix; N]>>,
    /// Grid coordinates
    pub x: [Array1<A>; N],
    /// Grid deltas
    pub dx: [Array1<A>; N],
}

impl<A, T1, T2, S, const N: usize> FieldBaseMpi<A, T1, T2, S, N>
where
    A: FloatNum,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, N, Physical = T1, Spectral = T2> + BaseSpaceMpi<A, N>,
{
    /// Return a new field from a given space
    #[allow(clippy::similar_names)]
    pub fn new(space: &S) -> Self {
        let v_x_pen = space.ndarray_physical_x_pen();
        let v_y_pen = space.ndarray_physical_y_pen();
        let vhat_x_pen = space.ndarray_spectral_x_pen();
        let vhat_y_pen = space.ndarray_spectral_y_pen();
        let x = space.coords();
        let dx = Self::get_dx(&space.coords(), Self::is_periodic(&space));
        Self {
            ndim: N,
            space: space.clone(),
            v_x_pen,
            v_y_pen,
            vhat_x_pen,
            vhat_y_pen,
            x,
            dx,
        }
    }

    /// Scale coordinates
    pub fn scale(&mut self, scale: [A; N]) {
        for (i, sc) in scale.iter().enumerate() {
            self.x[i] *= *sc;
            self.dx[i] *= *sc;
        }
    }

    /// Get coordinates that are local on processor
    pub fn get_coords_local(&self, axis: usize) -> ArrayView1<'_, A> {
        // Get mpi decomp
        let dcp = &self
            .space
            .get_decomp_from_global_shape(&self.space.shape_physical())
            .y_pencil;
        self.x[axis].slice(s![dcp.st[axis]..=dcp.en[axis]])
    }
}

// /// Conversion `FieldBase` -> `FieldBaseMpi`
// impl<A, T1, T2, S, const N: usize> From<&FieldBase<A, T1, T2, S, N>>
//     for FieldBaseMpi<A, T1, T2, S, N>
// where
//     A: FloatNum,
//     Complex<A>: ScalarOperand,
//     S: BaseSpace<A, N, Physical = T1, Spectral = T2> + BaseSpaceMpi<A, N>,
// {
//     fn from(item: &FieldBase<A, T1, T2, S, N>) -> Self {
//         let mut field_mpi = Self::new(&item.space);
//         field_mpi
//             .space
//             .scatter_to_y_pencil_phys(&item.v, &mut field_mpi.v_y_pen);
//         field_mpi
//             .space
//             .scatter_to_x_pencil_spec(&item.vhat, &mut field_mpi.vhat_y_pen);
//         field_mpi
//     }
// }
//
// /// Conversion `FieldBaseMpi` -> `FieldBase`
// impl<A, T1, T2, S, const N: usize> From<&FieldBaseMpi<A, T1, T2, S, N>>
//     for FieldBase<A, T1, T2, S, N>
// where
//     A: FloatNum,
//     Complex<A>: ScalarOperand,
//     S: BaseSpace<A, N, Physical = T1, Spectral = T2> + BaseSpaceMpi<A, N>,
// {
//     fn from(item: &FieldBaseMpi<A, T1, T2, S, N>) -> Self {
//         let mut field = Self::new(&item.space);
//         item.space
//             .gather_from_y_pencil_phys(&item.v_y_pen, &mut field.v);
//         item.space
//             .gather_from_x_pencil_spec(&item.vhat_x_pen, &mut field.vhat);
//         field
//     }
// }

impl<A, T1, T2, S, const N: usize> FieldBaseMpi<A, T1, T2, S, N>
where
    A: FloatNum,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, N, Physical = T1, Spectral = T2>,
{
    /// Generate grid deltas from coordinates
    ///
    /// ## Panics
    /// When vec to array convection fails
    fn get_dx(x_arr: &[Array1<A>; N], is_periodic: [bool; N]) -> [Array1<A>; N] {
        let mut dx_vec = Vec::new();
        let two = A::one() + A::one();
        for (x, periodic) in x_arr.iter().zip(is_periodic.iter()) {
            if *periodic {
                let dx = Array1::<A>::from_elem(x.len(), x[2] - x[1]);
                dx_vec.push(dx);
            } else {
                let mut dx = Array1::<A>::zeros(x.len());
                for (i, dxi) in dx.iter_mut().enumerate() {
                    let xs_left = if i == 0 {
                        x[0]
                    } else {
                        (x[i] + x[i - 1]) / two
                    };
                    let xs_right = if i == x.len() - 1 {
                        x[x.len() - 1]
                    } else {
                        (x[i + 1] + x[i]) / two
                    };
                    *dxi = xs_right - xs_left;
                }
                dx_vec.push(dx);
            }
        }
        dx_vec.try_into().unwrap_or_else(|v: Vec<Array1<A>>| {
            panic!("Expected Vec of length {} but got {}", N, v.len())
        })
    }

    /// Return true if base is periodic (used in calculating
    /// the grid spacing)
    fn is_periodic(space: &S) -> [bool; N] {
        let mut is_periodic: Vec<bool> = Vec::new();
        for axis in 0..N {
            let kind = space.base_kind(axis);

            let is_periodic_axis = match kind {
                BaseKind::Chebyshev
                | BaseKind::ChebDirichlet
                | BaseKind::ChebNeumann
                | BaseKind::ChebDirichletBc
                | BaseKind::ChebNeumannBc
                | BaseKind::ChebDirichletNeumann => false,
                BaseKind::FourierR2c | BaseKind::FourierC2c => true,
                // _ => panic!("Unknown Base kind: {}!", kind),
            };
            is_periodic.push(is_periodic_axis);
        }

        is_periodic.try_into().unwrap_or_else(|v: Vec<bool>| {
            panic!("Expected Vec of length {} but got {}", N, v.len())
        })
    }

    /// Hholtz equation: (I-c*D2) vhat = A f
    ///
    /// This function returns I (`mat_a`), D2 (`mat_b`) and
    /// the optional preconditionar A for a given base.
    ///
    /// # Panics
    /// If ingredients are not defined for a given base.
    pub fn ingredients_for_hholtz(&self, axis: usize) -> (Array2<A>, Array2<A>, Option<Array2<A>>) {
        let kind = self.space.base_kind(axis);
        let mass = self.space.mass(axis);
        let lap = self.space.laplace(axis);

        // Matrices and optional preconditioner
        match kind {
            BaseKind::Chebyshev => {
                let peye = self.space.laplace_inv_eye(axis);
                let pinv = peye.dot(&self.space.laplace_inv(axis));
                let mass_sliced = mass.slice(s![.., 2..]);
                (pinv.dot(&mass_sliced), peye.dot(&mass_sliced), Some(pinv))
            }
            BaseKind::ChebDirichlet | BaseKind::ChebNeumann | BaseKind::ChebDirichletNeumann => {
                let peye = self.space.laplace_inv_eye(axis);
                let pinv = peye.dot(&self.space.laplace_inv(axis));
                (pinv.dot(&mass), peye.dot(&mass), Some(pinv))
            }
            BaseKind::FourierR2c | BaseKind::FourierC2c => (mass, lap, None),
            _ => panic!("No ingredients found for Base kind: {}!", kind),
        }
    }

    /// Poisson equation: D2 vhat = A f
    ///
    /// This function returns I (`mat_a`), D2 (`mat_b`) and
    /// the optional preconditionar A for a given base.
    /// The mass matrix I is only used in multidimensional
    /// problems when D2 is not diagonal. This function
    /// also returns a hint, if D2 is diagonal.
    ///
    /// # Panics
    /// If ingredients are not defined for a given base.
    pub fn ingredients_for_poisson(
        &self,
        axis: usize,
    ) -> (Array2<A>, Array2<A>, Option<Array2<A>>, bool) {
        // Matrices and preconditioner
        let (mat_a, mat_b, precond) = self.ingredients_for_hholtz(axis);

        // Boolean, if laplacian is already diagonal
        // if not, a eigendecomposition will diagonalize mat a,
        // however, this is more expense.
        let kind = self.space.base_kind(axis);
        let is_diag = match kind {
            BaseKind::Chebyshev
            | BaseKind::ChebDirichlet
            | BaseKind::ChebNeumann
            | BaseKind::ChebDirichletNeumann => false,
            BaseKind::FourierR2c | BaseKind::FourierC2c => true,
            _ => panic!("No ingredients found for Base kind: {}!", kind),
        };
        (mat_a, mat_b, precond, is_diag)
    }
}

impl<A, T1, T2, S, const N: usize> FieldBaseMpi<A, T1, T2, S, N>
where
    A: FloatNum,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, N, Physical = T1, Spectral = T2> + BaseSpaceMpi<A, N>,
{
    /// Get my processor id
    pub fn universe(&self) -> &Universe {
        self.space.get_universe()
    }

    /// Get my processor id
    pub fn nrank(&self) -> usize {
        self.space.get_nrank()
    }

    /// Get total number of processors
    pub fn nprocs(&self) -> usize {
        self.space.get_nprocs()
    }

    /// Forward transformation with mpi
    pub fn forward_mpi(&mut self) {
        self.space
            .forward_inplace_mpi(&self.v_y_pen, &mut self.vhat_x_pen);
    }

    /// Backward transformation with mpi
    pub fn backward_mpi(&mut self) {
        self.space
            .backward_inplace_mpi(&self.vhat_x_pen, &mut self.v_y_pen);
    }

    /// Transform from composite to orthogonal space
    pub fn to_ortho_mpi(&self) -> Array<T2, Dim<[usize; N]>> {
        self.space.to_ortho_mpi(&self.vhat_x_pen)
    }

    /// Transform from orthogonal to composite space
    pub fn from_ortho_mpi<S1>(&mut self, input: &ArrayBase<S1, Dim<[usize; N]>>)
    where
        S1: Data<Elem = T2>,
    {
        self.space
            .from_ortho_inplace_mpi(input, &mut self.vhat_x_pen);
    }

    /// Gradient
    // #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn gradient_mpi(
        &self,
        deriv: [usize; N],
        scale: Option<[A; N]>,
    ) -> Array<T2, Dim<[usize; N]>> {
        self.space.gradient_mpi(&self.vhat_x_pen, deriv, scale)
    }

    /// Gather distributed data on root (x-pencil, spectral domain)
    ///
    /// # Info
    /// Must be called from mpi processor non-root
    pub fn gather_spectral(&self) {
        self.space.gather_from_x_pencil_spec(&self.vhat_x_pen);
    }

    /// Gather distributed data on root (x-pencil, spectral domain)
    ///
    /// # Info
    /// Must be called from mpi processor root
    pub fn gather_spectral_root<S1>(&self, vhat: &mut ArrayBase<S1, Dim<[usize; N]>>)
    where
        S1: DataMut<Elem = T2>,
    {
        self.space
            .gather_from_x_pencil_spec_root(&self.vhat_x_pen, vhat);
    }

    /// Gather distributed data on root (y-pencil, physical domain)
    ///
    /// # Info
    /// Must be called from mpi processor non-root
    pub fn gather_physical(&self) {
        self.space.gather_from_y_pencil_phys(&self.v_y_pen);
    }

    /// Gather distributed data on root (y-pencil, physical domain)
    ///
    /// # Info
    /// Must be called from mpi processor root
    pub fn gather_physical_root<S1>(&self, v: &mut ArrayBase<S1, Dim<[usize; N]>>)
    where
        S1: DataMut<Elem = T1>,
    {
        self.space.gather_from_y_pencil_phys_root(&self.v_y_pen, v);
    }

    /// Send data from root to all processors (y-pencil, physical domain)
    ///
    /// # Info
    /// Must be called from mpi processor non-root
    pub fn scatter_physical(&mut self) {
        self.space.scatter_to_y_pencil_phys(&mut self.v_y_pen);
    }

    /// Send data from root to all processors (y-pencil, physical domain)
    ///
    /// # Info
    /// Must be called from mpi processor root
    pub fn scatter_physical_root<S1>(&mut self, v: &ArrayBase<S1, Dim<[usize; N]>>)
    where
        S1: Data<Elem = T1>,
    {
        self.space
            .scatter_to_y_pencil_phys_root(v, &mut self.v_y_pen);
    }

    /// Send data from root to all processors (x-pencil, spectral domain)
    ///
    /// # Info
    /// Must be called from mpi processor non-root
    pub fn scatter_spectral(&mut self) {
        self.space.scatter_to_x_pencil_spec(&mut self.vhat_x_pen);
    }

    /// Send data from root to all processors (x-pencil, spectral domain)
    ///
    /// # Info
    /// Must be called from mpi processor root
    pub fn scatter_spectral_root<S1>(&mut self, vhat: &ArrayBase<S1, Dim<[usize; N]>>)
    where
        S1: Data<Elem = T2>,
    {
        self.space
            .scatter_to_x_pencil_spec_root(vhat, &mut self.vhat_x_pen);
    }

    /// Gather distributed data on all participating processors (y-pencil, physical domain)
    pub fn all_gather_physical<S1>(&mut self, v: &mut ArrayBase<S1, Dim<[usize; N]>>)
    where
        S1: DataMut<Elem = T1>,
    {
        self.space.all_gather_from_y_pencil_phys(&self.v_y_pen, v);
    }

    /// Gather distributed data on all participating processors (x-pencil, spectral domain)
    pub fn all_gather_spectral<S1>(&mut self, vhat: &mut ArrayBase<S1, Dim<[usize; N]>>)
    where
        S1: DataMut<Elem = T2>,
    {
        self.space
            .all_gather_from_x_pencil_spec(&self.vhat_x_pen, vhat);
    }

    /// Transpose physical data from x to y pencil
    pub fn transpose_x_to_y_phys(&mut self) {
        self.space
            .transpose_x_to_y_phys(&self.v_x_pen, &mut self.v_y_pen);
    }

    /// Transpose physical data from y to x pencil
    pub fn transpose_y_to_x_phys(&mut self) {
        self.space
            .transpose_y_to_x_phys(&self.v_y_pen, &mut self.v_x_pen);
    }

    /// Transpose spectral data from x to y pencil
    pub fn transpose_x_to_y_spec(&mut self) {
        self.space
            .transpose_x_to_y_spec(&self.vhat_x_pen, &mut self.vhat_y_pen);
    }

    /// Transpose spectral data from y to x pencil
    pub fn transpose_y_to_x_spec(&mut self) {
        self.space
            .transpose_y_to_x_spec(&self.vhat_y_pen, &mut self.vhat_x_pen);
    }
}
