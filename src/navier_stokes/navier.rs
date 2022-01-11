//! # Direct numerical simulation
//! Solver for 2-dimensional Navier-Stokes momentum equations
//! coupled with temperature equation.
//!
//! # Example
//! Solve 2-D Rayleigh Benard Convection
//! ```ignore
//! use rustpde::{Integrate, integrate};
//! use rustpde::navier_stokes::Navier2D;
//!
//! fn main() {
//!     // Parameters
//!     let (nx, ny) = (64, 64);
//!     let ra = 1e5;
//!     let pr = 1.;
//!     let adiabatic = true;
//!     let aspect = 1.0;
//!     let dt = 0.02;
//!     let mut navier = Navier2D::new(nx, ny, ra, pr, dt, aspect, adiabatic);
//!     // Set initial conditions
//!     navier.set_velocity(0.2, 1., 1.);
//!     // // Want to restart?
//!     // navier.read("data/flow100.000.h5"", None);
//!     // Write first field
//!     navier.callback();
//!     integrate(&mut navier, 100., Some(1.0));
//! }
//! ```
use super::boundary_conditions::{bc_rbc, bc_rbc_periodic};
use super::boundary_conditions::{pres_bc_rbc, pres_bc_rbc_periodic};
use super::conv_term::conv_term;
use super::functions::{apply_cos_sin, apply_random_disturbance, apply_sin_cos, dealias};
use super::functions::{get_ka, get_nu};
use super::statistics::Statistics;
use crate::bases::{cheb_dirichlet, cheb_neumann, chebyshev, fourier_r2c};
use crate::bases::{BaseR2c, BaseR2r};
use crate::field::{BaseSpace, Field2, ReadField, Space2, WriteField};
use crate::hdf5::{read_scalar_from_hdf5, write_scalar_to_hdf5, Result};
use crate::solver::{HholtzAdi, Poisson, Solve, SolverField};
use crate::types::Scalar;
use crate::Integrate;
use ndarray::Array2;
use num_complex::Complex;
use num_traits::Zero;
use std::collections::HashMap;
use std::ops::{Div, Mul};

type Space2R2r = Space2<BaseR2r<f64>, BaseR2r<f64>>;
type Space2R2c = Space2<BaseR2c<f64>, BaseR2r<f64>>;

/// Implement the ndividual terms of the Navier-Stokes equation
/// as a trait. This is necessary to support both real and complex
/// valued spectral spaces
pub trait NavierConvection {
    /// Type in physical space (ususally f64)
    type Physical;
    /// Type in spectral space (f64 or Complex<f64>)
    type Spectral;

    /// Convection term for temperature
    fn conv_temp(
        &mut self,
        ux: &Array2<Self::Physical>,
        uy: &Array2<Self::Physical>,
    ) -> Array2<Self::Spectral>;

    /// Convection term for velocity ux
    fn conv_ux(
        &mut self,
        ux: &Array2<Self::Physical>,
        uy: &Array2<Self::Physical>,
    ) -> Array2<Self::Spectral>;

    /// Convection term for velocity uy
    fn conv_uy(
        &mut self,
        ux: &Array2<Self::Physical>,
        uy: &Array2<Self::Physical>,
    ) -> Array2<Self::Spectral>;

    /// Solve horizontal momentum equation
    /// $$
    /// (1 - \delta t  \mathcal{D}) u\\_new = -dt*C(u) - \delta t grad(p) + \delta t f + u
    /// $$
    fn solve_ux(&mut self, ux: &Array2<Self::Physical>, uy: &Array2<Self::Physical>);

    /// Solve vertical momentum equation
    fn solve_uy(
        &mut self,
        ux: &Array2<Self::Physical>,
        uy: &Array2<Self::Physical>,
        buoy: &Array2<Self::Spectral>,
    );

    // Solve temperature equation:
    /// $$
    /// (1 - dt*D) temp\\_new = -dt*C(temp) + dt*fbc + temp
    /// $$
    fn solve_temp(&mut self, ux: &Array2<Self::Physical>, uy: &Array2<Self::Physical>);

    /// Correct velocity field.
    /// $$
    /// uxnew = ux - c*dpdx
    /// $$
    /// uynew = uy - c*dpdy
    /// $$
    fn project_velocity(&mut self, c: f64);

    /// Divergence: duxdx + duydy
    fn divergence(&mut self) -> Array2<Self::Spectral>;

    /// Solve pressure poisson equation
    /// $$
    /// D2 pres = f
    /// $$
    /// pseu: pseudo pressure ( in code it is pres\[1\] )
    fn solve_pres(&mut self, f: &Array2<Self::Spectral>);

    /// Update pressure term by divergence
    fn update_pres(&mut self, div: &Array2<Self::Spectral>);
}

/// Solve 2-dimensional Navier-Stokes equations
/// coupled with temperature equations
///
/// # Examples
///
/// ```
/// use rustpde::{integrate, Integrate};
/// use rustpde::navier_stokes::Navier2D;
/// let (nx, ny) = (33, 33);
/// let ra = 1e5;
/// let pr = 1.;
/// let adiabatic = true;
/// let aspect = 1.0;
/// let dt = 0.01;
/// let mut navier = Navier2D::new(nx, ny, ra, pr, dt, aspect, adiabatic);
/// // Read initial field from file
/// // navier.read("data/flow0.000.h5");
/// integrate(&mut navier, 0.2,  None);
/// ```
pub struct Navier2D<T, S> {
    /// Field for derivatives and transforms
    pub field: Field2<T, S>,
    /// Temperature
    pub temp: Field2<T, S>,
    /// Horizontal Velocity
    pub ux: Field2<T, S>,
    /// Vertical Velocity
    pub uy: Field2<T, S>,
    /// Pressure \[pres, pseudo pressure\]
    pub pres: [Field2<T, S>; 2],
    /// Collection of solvers \[ux, uy, temp, pres\]
    solver: [SolverField<f64, 2>; 4],
    /// Buffer
    rhs: Array2<T>,
    /// Field for temperature boundary condition
    pub tempbc: Option<Field2<T, S>>,
    /// Field for pressure boundary condition
    pub presbc: Option<Field2<T, S>>,
    /// Viscosity
    pub nu: f64,
    /// Thermal diffusivity
    pub ka: f64,
    /// Rayleigh number
    pub ra: f64,
    /// Prandtl number
    pub pr: f64,
    /// Time
    pub time: f64,
    /// Time step size
    pub dt: f64,
    /// Scale of phsical dimension \[scale_x, scale_y\]
    pub scale: [f64; 2],
    /// diagnostics like Nu, ...
    pub diagnostics: HashMap<String, Vec<f64>>,
    /// Time intervall for write fields
    /// If none, same intervall as diagnostics
    pub write_intervall: Option<f64>,
    /// Add a solid obstacle
    pub solid: Option<[Array2<f64>; 2]>,
    /// Set true and the fields will be dealiased
    pub dealias: bool,
    /// If set, collect statistics
    pub statistics: Option<Statistics<T, S>>,
}

impl Navier2D<f64, Space2R2r>
//where
//    S: BaseSpace<f64, 2, Physical = f64, Spectral = f64>,
{
    /// Bases: Chebyshev in x & y
    ///
    /// Struct must be mutable, to perform the
    /// update step, which advances the solution
    /// by 1 timestep.
    ///
    /// # Arguments
    ///
    /// * `nx,ny` - The number of modes in x and y -direction
    ///
    /// * `ra,pr` - Rayleigh and Prandtl number
    ///
    /// * `dt` - Timestep size
    ///
    /// * `aspect` - Aspect ratio L/H
    ///
    /// * `adiabatic` - Boolean, sidewall temperature boundary condition
    #[allow(clippy::similar_names)]
    pub fn new(
        nx: usize,
        ny: usize,
        ra: f64,
        pr: f64,
        dt: f64,
        aspect: f64,
        adiabatic: bool,
    ) -> Navier2D<f64, Space2R2r> {
        // geometry scales
        let scale = [aspect, 1.];
        // diffusivities
        let nu = get_nu(ra, pr, scale[1] * 2.0);
        let ka = get_ka(ra, pr, scale[1] * 2.0);
        // velocities
        let ux = Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny)));
        let uy = Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny)));
        // temperature
        let temp = if adiabatic {
            Field2::new(&Space2::new(&cheb_neumann(nx), &cheb_dirichlet(ny)))
        } else {
            Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny)))
        };
        // pressure
        let pres = [
            Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny))),
            Field2::new(&Space2::new(&cheb_neumann(nx), &cheb_neumann(ny))),
        ];
        // fields for derivatives
        let field = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));
        // define solver
        let solver_ux = SolverField::HholtzAdi(HholtzAdi::new(
            &ux,
            [dt * nu / scale[0].powf(2.), dt * nu / scale[1].powf(2.)],
        ));
        let solver_uy = SolverField::HholtzAdi(HholtzAdi::new(
            &uy,
            [dt * nu / scale[0].powf(2.), dt * nu / scale[1].powf(2.)],
        ));
        let solver_temp = SolverField::HholtzAdi(HholtzAdi::new(
            &temp,
            [dt * ka / scale[0].powf(2.), dt * ka / scale[1].powf(2.)],
        ));
        let solver_pres = SolverField::Poisson(Poisson::new(
            &pres[1],
            [1. / scale[0].powf(2.), 1. / scale[1].powf(2.)],
        ));
        let solver = [solver_ux, solver_uy, solver_temp, solver_pres];
        let rhs = Array2::zeros(temp.v.raw_dim());

        // Diagnostics
        let mut diagnostics = HashMap::new();
        diagnostics.insert("time".to_string(), Vec::<f64>::new());
        diagnostics.insert("Nu".to_string(), Vec::<f64>::new());
        diagnostics.insert("Nuvol".to_string(), Vec::<f64>::new());
        diagnostics.insert("Re".to_string(), Vec::<f64>::new());

        // Initialize
        let mut navier = Navier2D::<f64, Space2R2r> {
            field,
            temp,
            ux,
            uy,
            pres,
            solver,
            rhs,
            tempbc: None,
            presbc: None,
            nu,
            ka,
            ra,
            pr,
            time: 0.0,
            dt,
            scale,
            diagnostics,
            write_intervall: None,
            solid: None,
            dealias: true,
            statistics: None,
        };
        navier._scale();
        // Boundary conditions
        navier.set_temp_bc(bc_rbc(nx, ny));
        navier.set_pres_bc(pres_bc_rbc(nx, ny));
        // Initial condition
        // navier.set_velocity(0.2, 2., 1.);
        navier.random_disturbance(0.1);
        // Return
        navier
    }
}

impl Navier2D<Complex<f64>, Space2R2c>
//where
//    S: BaseSpace<f64, 2, Physical = f64, Spectral = f64>,
{
    /// Bases: Fourier in x and chebyshev in y
    ///
    /// Struct must be mutable, to perform the
    /// update step, which advances the solution
    /// by 1 timestep.
    ///
    /// # Arguments
    ///
    /// * `nx,ny` - The number of modes in x and y -direction
    ///
    /// * `ra,pr` - Rayleigh and Prandtl number
    ///
    /// * `dt` - Timestep size
    ///
    /// * `aspect` - Aspect ratio L/H (unity is assumed to be to 2pi)
    #[allow(clippy::similar_names)]
    pub fn new_periodic(
        nx: usize,
        ny: usize,
        ra: f64,
        pr: f64,
        dt: f64,
        aspect: f64,
    ) -> Navier2D<Complex<f64>, Space2R2c> {
        // geometry scales
        let scale = [aspect, 1.];
        // diffusivities
        let nu = get_nu(ra, pr, scale[1] * 2.0);
        let ka = get_ka(ra, pr, scale[1] * 2.0);
        // velocities
        let ux = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny)));
        let uy = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny)));
        // temperature
        let temp = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny)));
        // pressure
        let pres = [
            Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny))),
            Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_neumann(ny))),
        ];
        // fields for derivatives
        let field = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
        // define solver
        let solver_ux = SolverField::HholtzAdi(HholtzAdi::new(
            &ux,
            [dt * nu / scale[0].powf(2.), dt * nu / scale[1].powf(2.)],
        ));
        let solver_uy = SolverField::HholtzAdi(HholtzAdi::new(
            &uy,
            [dt * nu / scale[0].powf(2.), dt * nu / scale[1].powf(2.)],
        ));
        let solver_temp = SolverField::HholtzAdi(HholtzAdi::new(
            &temp,
            [dt * ka / scale[0].powf(2.), dt * ka / scale[1].powf(2.)],
        ));
        let solver_pres = SolverField::Poisson(Poisson::new(
            &pres[1],
            [1. / scale[0].powf(2.), 1. / scale[1].powf(2.)],
        ));
        let solver = [solver_ux, solver_uy, solver_temp, solver_pres];
        let rhs = Array2::zeros(field.vhat.raw_dim());

        // Diagnostics
        let mut diagnostics = HashMap::new();
        diagnostics.insert("time".to_string(), Vec::<f64>::new());
        diagnostics.insert("Nu".to_string(), Vec::<f64>::new());
        diagnostics.insert("Nuvol".to_string(), Vec::<f64>::new());
        diagnostics.insert("Re".to_string(), Vec::<f64>::new());

        // Initialize
        let mut navier = Navier2D::<Complex<f64>, Space2R2c> {
            field,
            temp,
            ux,
            uy,
            pres,
            solver,
            rhs,
            tempbc: None,
            presbc: None,
            nu,
            ka,
            ra,
            pr,
            time: 0.0,
            dt,
            scale,
            diagnostics,
            write_intervall: None,
            solid: None,
            dealias: true,
            statistics: None,
        };
        navier._scale();
        // Boundary condition
        navier.set_temp_bc(bc_rbc_periodic(nx, ny));
        navier.set_pres_bc(pres_bc_rbc_periodic(nx, ny));
        // Initial condition
        // navier.set_velocity(0.2, 2., 1.);
        navier.random_disturbance(0.1);
        // Return
        navier
    }
}

impl<T, S> Navier2D<T, S>
where
    T: num_traits::Zero,
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
{
    /// Rescale x & y coordinates of fields.
    /// Only affects output of files
    fn _scale(&mut self) {
        for field in &mut [
            &mut self.temp,
            &mut self.ux,
            &mut self.uy,
            &mut self.pres[0],
        ] {
            field.x[0] *= self.scale[0];
            field.x[1] *= self.scale[1];
            field.dx[0] *= self.scale[0];
            field.dx[1] *= self.scale[1];
        }
    }

    /// Set boundary condition field for temperature
    pub fn set_temp_bc(&mut self, fieldbc: Field2<T, S>) {
        self.tempbc = Some(fieldbc);
    }

    /// Set boundary condition field for pressure
    pub fn set_pres_bc(&mut self, presbc: Field2<T, S>) {
        self.presbc = Some(presbc);
    }

    fn zero_rhs(&mut self) {
        for r in self.rhs.iter_mut() {
            *r = T::zero();
        }
    }
}

macro_rules! impl_navier_convection {
    ($s: ty) => {
        impl<S> NavierConvection for Navier2D<$s, S>
        where
            S: BaseSpace<f64, 2, Physical = f64, Spectral = $s>,
        {
            type Physical = f64;
            type Spectral = $s;

            /// Convection term for temperature
            fn conv_temp(
                &mut self,
                ux: &Array2<Self::Physical>,
                uy: &Array2<Self::Physical>,
            ) -> Array2<Self::Spectral> {
                // + ux * dTdx + uy * dTdy
                let mut conv = conv_term(&self.temp, &mut self.field, ux, [1, 0], Some(self.scale));
                conv += &conv_term(&self.temp, &mut self.field, uy, [0, 1], Some(self.scale));
                // + bc contribution
                if let Some(field) = &self.tempbc {
                    conv += &conv_term(field, &mut self.field, ux, [1, 0], Some(self.scale));
                    conv += &conv_term(field, &mut self.field, uy, [0, 1], Some(self.scale));
                }
                // + solid interaction
                if let Some(solid) = &self.solid {
                    let eta = 1e-2;
                    self.temp.backward();
                    let damp = self.tempbc.as_ref().map_or_else(
                        || -1. / eta * &solid[0] * (&self.temp.v - &solid[1]),
                        |field| -1. / eta * &solid[0] * &(&self.temp.v + &field.v - &solid[1]),
                    );
                    conv -= &damp;
                }
                // -> spectral space
                self.field.v.assign(&conv);
                self.field.forward();
                if self.dealias {
                    dealias(&mut self.field);
                }
                self.field.vhat.to_owned()
            }

            /// Convection term for ux
            fn conv_ux(
                &mut self,
                ux: &Array2<Self::Physical>,
                uy: &Array2<Self::Physical>,
            ) -> Array2<Self::Spectral> {
                // + ux * dudx + uy * dudy
                let mut conv = conv_term(&self.ux, &mut self.field, ux, [1, 0], Some(self.scale));
                conv += &conv_term(&self.ux, &mut self.field, uy, [0, 1], Some(self.scale));
                // + solid interaction
                if let Some(solid) = &self.solid {
                    let eta = 1e-2;
                    let damp = -1. / eta * &solid[0] * ux;
                    conv -= &damp;
                }
                // -> spectral space
                self.field.v.assign(&conv);
                self.field.forward();
                if self.dealias {
                    dealias(&mut self.field);
                }
                self.field.vhat.to_owned()
            }

            /// Convection term for uy
            fn conv_uy(
                &mut self,
                ux: &Array2<Self::Physical>,
                uy: &Array2<Self::Physical>,
            ) -> Array2<Self::Spectral> {
                // + ux * dudx + uy * dudy
                let mut conv = conv_term(&self.uy, &mut self.field, ux, [1, 0], Some(self.scale));
                conv += &conv_term(&self.uy, &mut self.field, uy, [0, 1], Some(self.scale));
                // + solid interaction
                if let Some(solid) = &self.solid {
                    let eta = 1e-2;
                    let damp = -1. / eta * &solid[0] * uy;
                    conv -= &damp;
                }
                // -> spectral space
                self.field.v.assign(&conv);
                self.field.forward();
                if self.dealias {
                    dealias(&mut self.field);
                }
                self.field.vhat.to_owned()
            }

            /// Solve horizontal momentum equation
            /// $$
            /// (1 - \delta t  \mathcal{D}) u\\_new = -dt*C(u) - \delta t grad(p) + \delta t f + u
            /// $$
            fn solve_ux(&mut self, ux: &Array2<Self::Physical>, uy: &Array2<Self::Physical>) {
                self.zero_rhs();
                // + old field
                self.rhs += &self.ux.to_ortho();
                // + pres
                self.rhs -= &(self.pres[0].gradient([1, 0], Some(self.scale)) * self.dt);
                if let Some(field) = &self.presbc {
                    self.rhs -= &(field.gradient([1, 0], Some(self.scale)) * self.dt);
                }
                // + convection
                let conv = self.conv_ux(ux, uy);
                self.rhs -= &(conv * self.dt);
                // solve lhs
                self.solver[0].solve_par(&self.rhs, &mut self.ux.vhat, 0);
            }

            /// Solve vertical momentum equation
            fn solve_uy(
                &mut self,
                ux: &Array2<Self::Physical>,
                uy: &Array2<Self::Physical>,
                buoy: &Array2<Self::Spectral>,
            ) {
                self.zero_rhs();
                // + old field
                self.rhs += &self.uy.to_ortho();
                // + pres
                self.rhs -= &(self.pres[0].gradient([0, 1], Some(self.scale)) * self.dt);
                if let Some(field) = &self.presbc {
                    self.rhs -= &(field.gradient([0, 1], Some(self.scale)) * self.dt);
                }
                // + buoyancy
                self.rhs += &(buoy * self.dt);
                // + convection
                let conv = self.conv_uy(ux, uy);
                self.rhs -= &(conv * self.dt);
                // solve lhs
                self.solver[1].solve_par(&self.rhs, &mut self.uy.vhat, 0);
            }

            /// Solve temperature equation:
            /// $$
            /// (1 - dt*D) temp\\_new = -dt*C(temp) + dt*fbc + temp
            /// $$
            fn solve_temp(&mut self, ux: &Array2<Self::Physical>, uy: &Array2<Self::Physical>) {
                self.zero_rhs();
                // + old field
                self.rhs += &self.temp.to_ortho();
                // + diffusion bc contribution
                if let Some(field) = &self.tempbc {
                    self.rhs += &(field.gradient([2, 0], Some(self.scale)) * self.dt * self.ka);
                    self.rhs += &(field.gradient([0, 2], Some(self.scale)) * self.dt * self.ka);
                }
                // + convection
                let conv = self.conv_temp(ux, uy);
                self.rhs -= &(conv * self.dt);
                // solve lhs
                self.solver[2].solve_par(&self.rhs, &mut self.temp.vhat, 0);
            }

            /// Correct velocity field.
            /// $$
            /// uxnew = ux - c*dpdx
            /// $$
            /// uynew = uy - c*dpdy
            /// $$
            #[allow(clippy::similar_names)]
            fn project_velocity(&mut self, c: f64) {
                let dpdx = self.pres[1].gradient([1, 0], Some(self.scale));
                let dpdy = self.pres[1].gradient([0, 1], Some(self.scale));
                let ux_old = self.ux.vhat.clone();
                let uy_old = self.uy.vhat.clone();
                self.ux.from_ortho(&dpdx);
                self.uy.from_ortho(&dpdy);
                let cinto: Self::Spectral = (-c).into();
                self.ux.vhat *= cinto;
                self.uy.vhat *= cinto;
                self.ux.vhat += &ux_old;
                self.uy.vhat += &uy_old;
            }

            /// Divergence: duxdx + duydy
            fn divergence(&mut self) -> Array2<Self::Spectral> {
                self.zero_rhs();
                self.rhs += &self.ux.gradient([1, 0], Some(self.scale));
                self.rhs += &self.uy.gradient([0, 1], Some(self.scale));
                self.rhs.to_owned()
            }

            /// Solve pressure poisson equation
            /// $$
            /// D2 pres = f
            /// $$
            /// pseu: pseudo pressure ( in code it is pres\[1\] )
            fn solve_pres(&mut self, f: &Array2<Self::Spectral>) {
                //self.pres[1].vhat.assign(&self.solver[3].solve(&f));
                self.solver[3].solve_par(&f, &mut self.pres[1].vhat, 0);
                // Singularity
                self.pres[1].vhat[[0, 0]] = Self::Spectral::zero();
            }

            fn update_pres(&mut self, div: &Array2<Self::Spectral>) {
                self.pres[0].vhat = &self.pres[0].vhat - &(div * self.nu);
                let inv_dt: Self::Spectral = (1. / self.dt).into();
                self.pres[0].vhat = &self.pres[0].vhat + &(&self.pres[1].to_ortho() * inv_dt);
            }
        }
    };
}

impl_navier_convection!(f64);
impl_navier_convection!(Complex<f64>);

macro_rules! impl_integrate_for_navier {
    ($s: ty, $norm: ident) => {

        impl<S> Integrate for Navier2D<$s, S>
        where
            S: BaseSpace<f64, 2, Physical = f64, Spectral = $s>,
        {
            /// Update 1 timestep
            fn update(&mut self) {
                // Buoyancy
                let mut that = self.temp.to_ortho();
                if let Some(field) = &self.tempbc {
                    that = &that + &field.to_ortho();
                }

                // Convection Veclocity
                self.ux.backward();
                self.uy.backward();
                let ux = self.ux.v.to_owned();
                let uy = self.uy.v.to_owned();

                // Solve Velocity
                self.solve_ux(&ux, &uy);
                self.solve_uy(&ux, &uy, &that);

                // Projection
                let div = self.divergence();
                self.solve_pres(&div);
                self.project_velocity(1.0);
                self.update_pres(&div);

                // Solve Temperature
                self.solve_temp(&ux, &uy);

                // update time
                self.time += self.dt;
            }

            fn get_time(&self) -> f64 {
                self.time
            }

            fn get_dt(&self) -> f64 {
                self.dt
            }

            fn callback(&mut self) {
                use std::io::Write;

                // Write hdf5 file
                std::fs::create_dir_all("data").unwrap();

                // Write flow field
                //let fname = format!("data/flow{:.*}.h5", 3, self.time);
                let fname = format!("data/flow{:0>8.2}.h5", self.time);
                if let Some(dt_save) = &self.write_intervall {
                    if (self.time % dt_save) < self.dt / 2.
                        || (self.time % dt_save) > dt_save - self.dt / 2.
                    {
                        self.write(&fname);
                    }
                } else {
                    self.write(&fname);
                }

                // Write statistics
                let statname = "data/statistics.h5";
                if let Some(ref mut statistics) = self.statistics {
                    // Update
                    if (self.time % &statistics.save_stat) < self.dt / 2.
                        || (self.time % &statistics.save_stat) > &statistics.save_stat - self.dt / 2.
                    {
                        let that = if let Some(x) = &self.tempbc {
                            (&self.temp.to_ortho() + &x.to_ortho()).to_owned()
                        } else {
                            self.temp.to_ortho()
                        };
                        statistics.update(&that, &self.ux.to_ortho(), &self.uy.to_ortho(), self.time);
                    }
                    // Write
                    if (self.time % &statistics.write_stat) < self.dt / 2.
                        || (self.time % &statistics.write_stat) > &statistics.write_stat - self.dt / 2.
                    {
                        statistics.write(&statname);
                    }
                }

                // I/O
                let div = self.divergence();
                let nu = self.eval_nu();
                let nuvol = self.eval_nuvol();
                let re = self.eval_re();
                println!(
                    "time = {:4.2}      |div| = {:4.2e}     Nu = {:5.3e}     Nuv = {:5.3e}    Re = {:5.3e}",
                    self.time,
                    $norm(&div),
                    nu,
                    nuvol,
                    re,
                );

                // diagnostics
                if let Some(d) = self.diagnostics.get_mut("time") {
                    d.push(self.time);
                }
                if let Some(d) = self.diagnostics.get_mut("Nu") {
                    d.push(nu);
                }
                if let Some(d) = self.diagnostics.get_mut("Nuvol") {
                    d.push(nuvol);
                }
                if let Some(d) = self.diagnostics.get_mut("Re") {
                    d.push(re);
                }
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .append(true)
                    .create(true)
                    .open("data/info.txt")
                    .unwrap();
                //write!(file, "{} {}", time, nu);
                if let Err(e) = writeln!(file, "{} {} {} {}", self.time, nu, nuvol, re) {
                    eprintln!("Couldn't write to file: {}", e);
                }
            }

            fn exit(&mut self) -> bool {
                // Break if divergence is nan
                let div = self.divergence();
                if $norm(&div).is_nan() {
                    return true;
                }
                false
            }
        }
    };
}
impl_integrate_for_navier!(f64, norm_l2_f64);
impl_integrate_for_navier!(Complex<f64>, norm_l2_c64);

fn norm_l2_f64(array: &Array2<f64>) -> f64 {
    array.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
}

fn norm_l2_c64(array: &Array2<Complex<f64>>) -> f64 {
    array
        .iter()
        .map(|x| x.re.powi(2) + x.im.powi(2))
        .sum::<f64>()
        .sqrt()
}

impl<T, S> Navier2D<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar + Mul<f64, Output = T> + Div<f64, Output = T>,
{
    /// Returns Nusselt number (heat flux at the plates)
    /// $$
    /// Nu = \langle - dTdz \rangle\\_x (0/H))
    /// $$
    pub fn eval_nu(&mut self) -> f64 {
        use super::functions::eval_nu;
        eval_nu(&mut self.temp, &mut self.field, &self.tempbc, &self.scale)
    }

    /// Returns volumetric Nusselt number
    /// $$
    /// Nuvol = \langle uy*T/kappa - dTdz \rangle\\_V
    /// $$
    pub fn eval_nuvol(&mut self) -> f64 {
        use super::functions::eval_nuvol;
        eval_nuvol(
            &mut self.temp,
            &mut self.uy,
            &mut self.field,
            &self.tempbc,
            self.ka,
            &self.scale,
        )
    }

    /// Returns Reynolds number based on kinetic energy
    pub fn eval_re(&mut self) -> f64 {
        use super::functions::eval_re;
        eval_re(
            &mut self.ux,
            &mut self.uy,
            &mut self.field,
            self.nu,
            &self.scale,
        )
    }

    /// Initialize velocity with fourier modes
    ///
    /// ux = amp \* sin(mx)cos(nx)
    /// uy = -amp \* cos(mx)sin(nx)
    pub fn set_velocity(&mut self, amp: f64, m: f64, n: f64) {
        apply_sin_cos(&mut self.ux, amp, m, n);
        apply_cos_sin(&mut self.uy, -amp, m, n);
    }
    /// Initialize temperature with fourier modes
    ///
    /// temp = -amp \* cos(mx)sin(ny)
    pub fn set_temperature(&mut self, amp: f64, m: f64, n: f64) {
        apply_cos_sin(&mut self.temp, -amp, m, n);
    }

    /// Initialize all fields with random disturbances
    pub fn random_disturbance(&mut self, amp: f64) {
        apply_random_disturbance(&mut self.temp, amp);
        apply_random_disturbance(&mut self.ux, amp);
        apply_random_disturbance(&mut self.uy, amp);
        // Remove bc base from temp
        if let Some(x) = &self.tempbc {
            self.temp.v = &self.temp.v - &x.v;
            self.temp.forward();
        }
    }

    /// Reset time
    pub fn reset_time(&mut self) {
        self.time = 0.;
    }
}

macro_rules! impl_read_write_navier {
    ($s: ty) => {
        impl<S> Navier2D<$s, S>
        where
            S: BaseSpace<f64, 2, Physical = f64, Spectral = $s>,
        {
            /// Restart from file
            pub fn read(&mut self, filename: &str) {
                // Field
                self.temp.read(&filename, Some("temp"));
                self.ux.read(&filename, Some("ux"));
                self.uy.read(&filename, Some("uy"));
                self.pres[0].read(&filename, Some("pres"));
                // Read scalars
                self.time = read_scalar_from_hdf5::<f64>(&filename, "time", None).unwrap();
                println!(" <== {:?}", filename);
            }

            /// Write Field data to hdf5 file
            pub fn write(&mut self, filename: &str) {
                let result = self.write_return_result(filename);
                match result {
                    Ok(_) => println!(" ==> {:?}", filename),
                    Err(_) => println!("Error while writing file {:?}.", filename),
                }
            }

            fn write_return_result(&mut self, filename: &str) -> Result<()> {
                use crate::hdf5::write_to_hdf5;
                self.temp.backward();
                self.ux.backward();
                self.uy.backward();
                self.pres[0].backward();
                // Add boundary contribution
                if let Some(x) = &self.tempbc {
                    self.temp.v = &self.temp.v + &x.v;
                }
                // Field
                self.temp.write(&filename, Some("temp"));
                self.ux.write(&filename, Some("ux"));
                self.uy.write(&filename, Some("uy"));
                self.pres[0].write(&filename, Some("pres"));
                // Write solid mask
                if let Some(x) = &self.solid {
                    write_to_hdf5(&filename, "mask", Some("solid"), &x[0])?;
                }
                // Write scalars
                write_scalar_to_hdf5(&filename, "time", None, self.time)?;
                write_scalar_to_hdf5(&filename, "ra", None, self.ra)?;
                write_scalar_to_hdf5(&filename, "pr", None, self.pr)?;
                write_scalar_to_hdf5(&filename, "nu", None, self.nu)?;
                write_scalar_to_hdf5(&filename, "kappa", None, self.ka)?;
                // Undo addition of bc
                if self.tempbc.is_some() {
                    self.temp.backward();
                }
                Ok(())
            }
        }
    };
}

impl_read_write_navier!(f64);
impl_read_write_navier!(Complex<f64>);
