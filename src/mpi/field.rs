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
pub use super::BaseSpaceMpi;
pub use super::Decomp2d;
pub use super::Universe;
use crate::bases::BaseSpace;
use crate::bases::LaplacianInverse;
use crate::bases::{BaseAll, BaseC2c, BaseR2c, BaseR2r, Basics};
use crate::field::read::broadcast_2d;
use crate::hdf5::Result;
use crate::types::FloatNum;
use crate::ReadField;
use crate::WriteField;
use hdf5_interface::H5Type;
use ndarray::{prelude::*, Data};
use ndarray::{Ix, ScalarOperand, Slice};
use num_complex::Complex;
use std::convert::TryInto;

/// Two dimensional Field (Real in Physical space, Generic in Spectral Space)
pub type Field2Mpi<T2, S> = FieldBaseMpi<f64, f64, T2, S, 2>;

/// Field struct with mpi support
///
/// v: ndarray
///
///   Holds data in physical space
///
/// vhat: ndarray
///
///   Holds data in spectral space
///
/// v_x_pen: ndarray
///
///   Holds local data in physical space, distributed as x-pencil
///   across processors
///
/// v_y_pen: ndarray
///
///   Holds local data in physical space, distributed as y-pencil
///   across processors
///
/// vhat_x_pen: ndarray
///
///   Holds local data in spectral space, distributed as x-pencil
///   across processors
///
/// vhat_y_pen: ndarray
///
///   Holds local data in spectral space, distributed as y-pencil
///   across processors
///
/// x: list of ndarrays
///
///   Grid points (physical space)
///
/// dx: list of ndarrays
///
///   Grid points deltas (physical space)
///
/// solvers: HashMap<String, `SolverField`>
///
///  Add plans for various equations
///
/// `FieldBase` is derived from `SpaceBase` struct,
/// defined in the `funspace` crate.
/// It implements forward / backward transform from physical
/// to spectral space, differentation and casting
/// from an orthonormal space to its galerkin space (`from_ortho`
/// and `to_ortho`).
///
// / # Example
// / 2-D field in chebyshev space
// /```
// / use rustpde::cheb_dirichlet;
// / use rustpde::{Space2, Field2};
// /
// / let cdx = cheb_dirichlet::<f64>(8);
// / let cdy = cheb_dirichlet::<f64>(6);
// / let space = Space2::new(&cdx, &cdy);
// / let field = Field2::new(&space);
// /```
#[derive(Clone)]
pub struct FieldBaseMpi<A, T1, T2, S, const N: usize> {
    /// Number of dimensions
    pub ndim: usize,
    /// Space
    pub space: S,
    /// Field in physical space
    pub v: Array<T1, Dim<[Ix; N]>>,
    /// Field in spectral space
    pub vhat: Array<T2, Dim<[Ix; N]>>,
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
    // /// Collection of numerical solvers (Poisson, Hholtz, ...)
    // pub solvers: HashMap<String, SolverField<T, N>>,
}

impl<A, T1, T2, S, const N: usize> FieldBaseMpi<A, T1, T2, S, N>
where
    A: FloatNum,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, N, Physical = T1, Spectral = T2> + BaseSpaceMpi<A, N>,
{
    /// Return a new field from a given space
    pub fn new(space: &S) -> Self {
        let v = space.ndarray_physical();
        let vhat = space.ndarray_spectral();
        let v_x_pen = space.ndarray_physical_x_pen();
        let v_y_pen = space.ndarray_physical_y_pen();
        let vhat_x_pen = space.ndarray_spectral_x_pen();
        let vhat_y_pen = space.ndarray_spectral_y_pen();
        let x = space.coords();
        let dx = Self::get_dx(&space.coords(), Self::is_periodic(&space));
        Self {
            ndim: N,
            space: space.clone(),
            v,
            vhat,
            v_x_pen,
            v_y_pen,
            vhat_x_pen,
            vhat_y_pen,
            x,
            dx,
        }
    }
}

impl<A, T1, T2, S, const N: usize> FieldBaseMpi<A, T1, T2, S, N>
where
    A: FloatNum,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, N, Physical = T1, Spectral = T2>,
{
    /// Forward transformation
    pub fn forward(&mut self) {
        self.space.forward_inplace(&self.v, &mut self.vhat);
    }

    /// Backward transformation
    pub fn backward(&mut self) {
        self.space.backward_inplace(&self.vhat, &mut self.v);
    }

    /// Transform from composite to orthogonal space
    pub fn to_ortho(&self) -> Array<T2, Dim<[usize; N]>> {
        self.space.to_ortho(&self.vhat)
    }

    /// Transform from orthogonal to composite space
    pub fn from_ortho<S1>(&mut self, input: &ArrayBase<S1, Dim<[usize; N]>>)
    where
        S1: Data<Elem = T2>,
    {
        self.space.from_ortho_inplace(input, &mut self.vhat);
    }

    /// Gradient
    // #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    pub fn gradient(&self, deriv: [usize; N], scale: Option<[A; N]>) -> Array<T2, Dim<[usize; N]>> {
        self.space.gradient(&self.vhat, deriv, scale)
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
            let x = &space.base_all()[axis];
            let is_periodic_axis = match x {
                BaseAll::BaseR2r(ref b) => match b {
                    BaseR2r::Chebyshev(_) | BaseR2r::CompositeChebyshev(_) => false,
                },
                BaseAll::BaseR2c(ref b) => match b {
                    BaseR2c::FourierR2c(_) => true,
                },
                BaseAll::BaseC2c(ref b) => match b {
                    BaseC2c::FourierC2c(_) => true,
                },
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
    pub fn ingredients_for_hholtz(&self, axis: usize) -> (Array2<A>, Array2<A>, Option<Array2<A>>) {
        let x = &self.space.base_all()[axis];
        let mass = x.mass();
        let lap = x.laplace();
        let peye = x.laplace_inv_eye();
        let pinv = peye.dot(&x.laplace_inv());

        // Matrices
        let (mat_a, mat_b) = match x {
            BaseAll::BaseR2r(ref b) => match b {
                BaseR2r::Chebyshev(_) => {
                    let mass_sliced = mass.slice_axis(Axis(1), Slice::from(2..));
                    (pinv.dot(&mass_sliced), peye.dot(&mass_sliced))
                }
                BaseR2r::CompositeChebyshev(_) => (pinv.dot(&mass), peye.dot(&mass)),
            },
            BaseAll::BaseR2c(ref b) => match b {
                BaseR2c::FourierR2c(_) => (mass, lap),
            },
            BaseAll::BaseC2c(ref b) => match b {
                BaseC2c::FourierC2c(_) => (mass, lap),
            },
        };
        // Preconditioner (optional)
        let precond = match x {
            BaseAll::BaseR2r(ref b) => match b {
                BaseR2r::Chebyshev(_) | BaseR2r::CompositeChebyshev(_) => Some(pinv),
            },
            BaseAll::BaseR2c(_) | BaseAll::BaseC2c(_) => None,
        };
        (mat_a, mat_b, precond)
    }

    /// Poisson equation: D2 vhat = A f
    ///
    /// This function returns I (`mat_a`), D2 (`mat_b`) and
    /// the optional preconditionar A for a given base.
    /// The mass matrix I is only used in multidimensional
    /// problems when D2 is not diagonal. This function
    /// also returns a hint, if D2 is diagonal.
    pub fn ingredients_for_poisson(
        &self,
        axis: usize,
    ) -> (Array2<A>, Array2<A>, Option<Array2<A>>, bool) {
        let x = &self.space.base_all()[axis];

        // Matrices and preconditioner
        let (mat_a, mat_b, precond) = self.ingredients_for_hholtz(axis);

        // Boolean, if laplacian is already diagonal
        // if not, a eigendecomposition will diagonalize mat a,
        // however, this is more expense.
        let is_diag = match x {
            BaseAll::BaseR2r(_) => false,
            BaseAll::BaseR2c(_) | BaseAll::BaseC2c(_) => true,
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

    // /// Get pencil decomposition (physical domain)
    // pub fn decomp_physical(&self) -> &Decomp2d {
    //     self.space.get_decomp_phys()
    // }
    //
    // /// Get pencil decomposition (spectral domain)
    // pub fn decomp_spectral(&self) -> &Decomp2d {
    //     self.space.get_decomp_spec()
    // }

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

    /// Gather distributed data on root (y-pencil, physical domain)
    pub fn gather_physical(&mut self) {
        self.space
            .gather_from_y_pencil_phys(&self.v_y_pen, &mut self.v);
    }

    /// Gather distributed data on root (x-pencil, spectral domain)
    pub fn gather_spectral(&mut self) {
        self.space
            .gather_from_x_pencil_spec(&self.vhat_x_pen, &mut self.vhat);
    }

    /// Gather distributed data on all participating processors (y-pencil, physical domain)
    pub fn all_gather_physical(&mut self) {
        self.space
            .all_gather_from_y_pencil_phys(&self.v_y_pen, &mut self.v);
    }

    /// Gather distributed data on all participating processors (x-pencil, spectral domain)
    pub fn all_gather_spectral(&mut self) {
        self.space
            .all_gather_from_x_pencil_spec(&self.vhat_x_pen, &mut self.vhat);
    }

    /// Send data from root to all processors (y-pencil, physical domain)
    pub fn scatter_physical(&mut self) {
        self.space
            .scatter_to_y_pencil_phys(&self.v, &mut self.v_y_pen);
    }

    /// Send data from root to all processors   (x-pencil, spectral domain)
    pub fn scatter_spectral(&mut self) {
        self.space
            .scatter_to_x_pencil_spec(&self.vhat, &mut self.vhat_x_pen);
    }

    // /// Split global data to y pencil for physical domain
    // pub fn split_physical(&mut self) {
    //     self.space
    //         .split_to_y_pencil_phys(&self.v, &mut self.v_y_pen);
    // }
    //
    // /// Split global data to x pencil for spectral domain
    // pub fn split_spectral(&mut self) {
    //     self.space
    //         .split_to_x_pencil_spec(&self.vhat, &mut self.vhat_x_pen);
    // }

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

impl<A: FloatNum, T2, S> FieldBaseMpi<A, A, T2, S, 2>
where
    S: BaseSpace<A, 2, Physical = A, Spectral = T2>,
{
    /// Return volumetric weighted average along axis
    /// # Example
    ///```
    /// use ndarray::{array, Axis};
    /// use rustpde::{chebyshev, Field2, Space2};
    /// let (nx, ny) = (6, 5);
    /// let space = Space2::new(&chebyshev(nx), &chebyshev(ny));
    /// let mut field = Field2::new(&space);
    /// for mut lane in field.v.lanes_mut(Axis(1)) {
    ///     for (i, vi) in lane.iter_mut().enumerate() {
    ///         *vi = i as f64;
    ///     }
    /// }
    /// assert!(field.average_axis(0) == array![0.0, 1.0, 2.0, 3.0, 4.0]);
    ///```
    pub fn average_axis(&self, axis: usize) -> Array1<A> {
        let mut weighted_avg = Array2::<A>::zeros(self.v.raw_dim());
        let length: A = (self.x[axis][self.x[axis].len() - 1] - self.x[axis][0]).abs();
        ndarray::Zip::from(self.v.lanes(Axis(axis)))
            .and(weighted_avg.lanes_mut(Axis(axis)))
            .for_each(|ref v, mut s| {
                s.assign(&(v * &self.dx[axis] / length));
            });
        weighted_avg.sum_axis(Axis(axis))
    }

    /// Return volumetric weighted average
    /// # Example
    ///```
    /// use ndarray::{array, Axis};
    /// use rustpde::{chebyshev, Space2, Field2};
    /// let (nx, ny) = (6, 5);
    /// let space = Space2::new(&chebyshev(nx), &chebyshev(ny));
    /// let mut field = Field2::new(&space);
    /// for mut lane in field.v.lanes_mut(Axis(1)) {
    ///     for (i, vi) in lane.iter_mut().enumerate() {
    ///         *vi = i as f64;
    ///     }
    /// }
    /// assert!(field.average() == 2.);
    ///```
    pub fn average(&self) -> A {
        let mut avg_x = Array1::<A>::zeros(self.dx[1].raw_dim());
        let length = (self.x[1][self.x[1].len() - 1] - self.x[1][0]).abs();
        avg_x.assign(&(self.average_axis(0) * &self.dx[1] / length));
        let avg = avg_x.sum_axis(Axis(0));
        avg[[]]
    }
}

impl<A, S> ReadField for FieldBaseMpi<A, A, A, S, 2>
where
    A: FloatNum + H5Type,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 2, Physical = A, Spectral = A>,
{
    fn read(&mut self, filename: &str, group: Option<&str>) {
        use crate::hdf5::read_from_hdf5;
        let result = read_from_hdf5::<A, Ix2>(filename, "vhat", group);
        match result {
            Ok(x) => {
                if x.shape() == self.vhat.shape() {
                    self.vhat.assign(&x);
                } else {
                    println!(
                        "Attention! Broadcast from shape {:?} to shape {:?}.",
                        x.shape(),
                        self.vhat.shape()
                    );
                    broadcast_2d(&x, &mut self.vhat);
                }
                self.backward();
                println!("Reading file {:?} was successfull.", filename);
            }
            Err(_) => println!("Error while reading file {:?}.", filename),
        }
    }
}

impl<A, S> ReadField for FieldBaseMpi<A, A, Complex<A>, S, 2>
where
    A: FloatNum + H5Type,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 2, Physical = A, Spectral = Complex<A>>,
{
    #[allow(clippy::cast_precision_loss)]
    fn read(&mut self, filename: &str, group: Option<&str>) {
        use crate::hdf5::read_from_hdf5_complex;
        let result = read_from_hdf5_complex::<A, Ix2>(filename, "vhat", group);
        match result {
            Ok(x) => {
                if x.shape() == self.vhat.shape() {
                    self.vhat.assign(&x);
                } else {
                    println!(
                        "Attention! Broadcast from shape {:?} to shape {:?}.",
                        x.shape(),
                        self.vhat.shape()
                    );
                    broadcast_2d(&x, &mut self.vhat);
                    // Renormalize Fourier base
                    let base = &self.space.base_all()[0];
                    match base {
                        BaseAll::BaseR2c(b) => match b {
                            BaseR2c::FourierR2c(_) => {
                                let norm = A::from(
                                    (self.vhat.shape()[0] - 1) as f64 / (x.shape()[0] - 1) as f64,
                                )
                                .unwrap();
                                for v in self.vhat.iter_mut() {
                                    v.re *= norm;
                                    v.im *= norm;
                                }
                            }
                        },
                        _ => (),
                    };
                }
                self.backward();
                println!("Reading file {:?} was successfull.", filename);
            }
            Err(_) => println!("Error while reading file {:?}.", filename),
        }
    }
}

impl<A, S> WriteField for FieldBaseMpi<A, A, A, S, 2>
where
    A: FloatNum + H5Type,
    S: BaseSpace<A, 2, Physical = A, Spectral = A>,
{
    /// Write Field data to hdf5 file
    fn write(&mut self, filename: &str, group: Option<&str>) {
        let result = self.write_return_result(filename, group);
        match result {
            Ok(_) => (),
            Err(_) => println!("Error while writing file {:?}.", filename),
        }
    }

    fn write_return_result(&mut self, filename: &str, group: Option<&str>) -> Result<()> {
        use hdf5_interface::write_to_hdf5;
        write_to_hdf5(filename, "v", group, &self.v)?;
        write_to_hdf5(filename, "vhat", group, &self.vhat)?;
        write_to_hdf5(filename, "x", None, &self.x[0])?;
        write_to_hdf5(filename, "dx", None, &self.dx[0])?;
        write_to_hdf5(filename, "y", None, &self.x[1])?;
        write_to_hdf5(filename, "dy", None, &self.dx[1])?;
        Ok(())
    }
}

impl<A, S> WriteField for FieldBaseMpi<A, A, Complex<A>, S, 2>
where
    A: FloatNum + H5Type,
    S: BaseSpace<A, 2, Physical = A, Spectral = Complex<A>>,
{
    /// Write Field data to hdf5 file
    fn write(&mut self, filename: &str, group: Option<&str>) {
        let result = self.write_return_result(filename, group);
        match result {
            Ok(_) => (),
            Err(_) => println!("Error while writing file {:?}.", filename),
        }
    }

    fn write_return_result(&mut self, filename: &str, group: Option<&str>) -> Result<()> {
        use hdf5_interface::write_to_hdf5;
        use hdf5_interface::write_to_hdf5_complex;
        write_to_hdf5(filename, "v", group, &self.v)?;
        write_to_hdf5_complex(filename, "vhat", group, &self.vhat)?;
        write_to_hdf5(filename, "x", None, &self.x[0])?;
        write_to_hdf5(filename, "dx", None, &self.dx[0])?;
        write_to_hdf5(filename, "y", None, &self.x[1])?;
        write_to_hdf5(filename, "dy", None, &self.dx[1])?;
        Ok(())
    }
}
