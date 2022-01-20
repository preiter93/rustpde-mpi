//! # Multidimensional field of basis functions
//! Wrapper around funspace `Space` fields, plus
//! let field store *n*-dimensional arrays which
//! belong the the physical (v) and spectral (vhat)
//! space.
pub mod average;
pub mod io;
use crate::bases::BaseKind;
pub use crate::bases::{BaseSpace, Space1, Space2};
use crate::types::FloatNum;
use ndarray::{prelude::*, Data};
use ndarray::{Ix, ScalarOperand};
use num_complex::Complex;
use std::convert::TryInto;

/// One dimensional Field (Real in Physical space, Generic in Spectral Space)
pub type Field1<T2, S> = FieldBase<f64, f64, T2, S, 1>;

/// Two dimensional Field (Real in Physical space, Generic in Spectral Space)
pub type Field2<T2, S> = FieldBase<f64, f64, T2, S, 2>;

/// Field struct is rustpdes backbone
///
/// v: ndarray
///
///   Holds data in physical space
///
/// vhat: ndarray
///
///   Holds data in spectral space
///
/// x: list of ndarrays
///
///   Grid points (physical space)
///
/// dx: list of ndarrays
///
///   Grid points deltas (physical space)
///
/// `FieldBase` is derived from `SpaceBase` struct,
/// defined in the `funspace` crate.
/// It implements forward / backward transform from physical
/// to spectral space, differentation and casting
/// from an orthonormal space to its galerkin space (`from_ortho`
/// and `to_ortho`).
///
/// # Example
/// 2-D field in chebyshev space
///```
/// use rustpde::bases::cheb_dirichlet;
/// use rustpde::field::{Field2, Space2};
///
/// let cdx = cheb_dirichlet::<f64>(8);
/// let cdy = cheb_dirichlet::<f64>(6);
/// let space = Space2::new(&cdx, &cdy);
/// let field = Field2::new(&space);
///```
#[derive(Clone)]
pub struct FieldBase<A, T1, T2, S, const N: usize> {
    /// Number of dimensions
    pub ndim: usize,
    /// Space
    pub space: S,
    /// Field in physical space
    pub v: Array<T1, Dim<[Ix; N]>>,
    /// Field in spectral space
    pub vhat: Array<T2, Dim<[Ix; N]>>,
    /// Grid coordinates
    pub x: [Array1<A>; N],
    /// Grid deltas
    pub dx: [Array1<A>; N],
}

impl<A, T1, T2, S, const N: usize> FieldBase<A, T1, T2, S, N>
where
    A: FloatNum,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, N, Physical = T1, Spectral = T2>,
{
    /// Return a new field from a given space
    pub fn new(space: &S) -> Self {
        Self {
            ndim: N,
            space: space.clone(),
            v: space.ndarray_physical(),
            vhat: space.ndarray_spectral(),
            x: space.coords(),
            dx: Self::get_dx(&space.coords(), Self::is_periodic(space)),
        }
    }

    /// Scale coordinates
    pub fn scale(&mut self, scale: [A; N]) {
        for (i, sc) in scale.iter().enumerate() {
            self.x[i] *= *sc;
            self.dx[i] *= *sc;
        }
    }

    /// Forward transformation
    pub fn forward(&mut self) {
        self.space.forward_inplace_par(&self.v, &mut self.vhat);
    }

    /// Backward transformation
    pub fn backward(&mut self) {
        self.space.backward_inplace_par(&self.vhat, &mut self.v);
    }

    /// Transform from composite to orthogonal space
    pub fn to_ortho(&self) -> Array<T2, Dim<[usize; N]>> {
        self.space.to_ortho_par(&self.vhat)
    }

    /// Transform from orthogonal to composite space
    pub fn from_ortho<S1>(&mut self, input: &ArrayBase<S1, Dim<[usize; N]>>)
    where
        S1: Data<Elem = T2>,
    {
        self.space.from_ortho_inplace_par(input, &mut self.vhat);
    }

    /// Gradient
    // #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn gradient(&self, deriv: [usize; N], scale: Option<[A; N]>) -> Array<T2, Dim<[usize; N]>> {
        self.space.gradient_par(&self.vhat, deriv, scale)
    }

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
