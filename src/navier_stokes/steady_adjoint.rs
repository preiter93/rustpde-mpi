//! # Adjoint descent method for steady state solutions
//! Solve adjoint 2-dimensional Navier-Stokes equations
//! coupled with temperature equations to obtain steady
//! state solutions
//!
//! # Example
//! Find steady state solution of large scale circulation
//! ```ignore
//! use rustpde::{Integrate, integrate};
//! use rustpde::navier_stokes::steady_adjoint::Navier2DAdjoint;
//!
//! fn main() {
//!     // Parameters
//!     let (nx, ny) = (64, 64);
//!     let ra = 1e5;
//!     let pr = 1.;
//!     let adiabatic = true;
//!     let aspect = 1.0;
//!     let dt = 0.02;
//!     let mut navier_adjoint = Navier2DAdjoint::new(nx, ny, ra, pr, dt, aspect, "rbc");
//!     // Set initial conditions
//!     navier_adjoint.set_temperature(0.5, 1., 1.);
//!     navier_adjoint.set_velocity(0.5, 1., 1.);
//!     // // Want to restart?
//!     // navier_adjoint.read("data/flow100.000.h5", None);
//!     // Write first field
//!     navier_adjoint.callback();
//!     integrate(&mut navier_adjoint, 100., Some(1.0));
//! }
//! ```
//!
//! ## References
//! <a id="1">\[1\]</a>
//! M. Farazmand (2016).
//! An adjoint-based approach for finding invariant solutions of Navier--Stokes equations
//! J. Fluid Mech., 795, 278-312.
//!
//! <a id="1">\[2\]</a>
//! P. Reiter et al. (2022)
//! Flow states and heat transport in Rayleigh--B\'enard convection with different sidewall
//! boundary conditions
//! J. Fluid Mech., in print.
use super::boundary_conditions::{bc_hc, bc_hc_periodic, bc_rbc, bc_rbc_periodic};
use super::functions::{apply_cos_sin, apply_sin_cos, random_field};
use super::functions::{get_ka, get_nu, norm_l2_c64, norm_l2_f64};
use super::navier::{Navier2D, Space2R2c, Space2R2r};
use crate::bases::{cheb_dirichlet, cheb_dirichlet_neumann, cheb_neumann, chebyshev, fourier_r2c};
use crate::bases::{BaseR2c, BaseR2r};
use crate::field::{BaseSpace, Field2, Space2};
use crate::solver::{Hholtz, Poisson, Solve};
use crate::types::Scalar;
use crate::Integrate;
use ndarray::Array2;
use num_complex::Complex;
use num_traits::Zero;
use std::collections::HashMap;
use std::ops::{Div, Mul};

/// Tolerance criteria for residual
const RES_TOL: f64 = 1e-8;
/// Laplacian weight in norm
const WEIGHT_LAPLACIAN: f64 = 1e-1;
/// Timestep of forward navier integration
const DT_NAVIER: f64 = 1e-2;

// /// Implement the ndividual terms of the Navier-Stokes equation
// /// as a trait. This is necessary to support both real and complex
// /// valued spectral spaces
// pub trait NavierConvectionAdjoint {
//     /// Type in physical space (ususally f64)
//     type Physical;
//     /// Type in spectral space (f64 or Complex<f64>)
//     type Spectral;
//
//     /// Convection term for velocity ux
//     fn conv_ux(
//         &mut self,
//         ux: &Array2<Self::Physical>,
//         uy: &Array2<Self::Physical>,
//         t: &Array2<Self::Physical>,
//     ) -> Array2<Self::Spectral>;
//
//     /// Convection term for velocity uy
//     fn conv_uy(
//         &mut self,
//         ux: &Array2<Self::Physical>,
//         uy: &Array2<Self::Physical>,
//         t: &Array2<Self::Physical>,
//     ) -> Array2<Self::Spectral>;
//
//     /// Convection term for temperature
//     fn conv_temp(
//         &mut self,
//         ux: &Array2<Self::Physical>,
//         uy: &Array2<Self::Physical>,
//     ) -> Array2<Self::Spectral>;
//
//     /// Solve horizontal momentum equation
//     fn solve_ux(
//         &mut self,
//         ux: &Array2<Self::Physical>,
//         uy: &Array2<Self::Physical>,
//         temp: &Array2<Self::Physical>,
//     );
//
//     /// Solve vertical momentum equation
//     fn solve_uy(
//         &mut self,
//         ux: &Array2<Self::Physical>,
//         uy: &Array2<Self::Physical>,
//         temp: &Array2<Self::Physical>,
//     );
//
//     /// Solve temperature equation:
//     fn solve_temp(&mut self, ux: &Array2<Self::Physical>, uy: &Array2<Self::Physical>);
//
//     /// Correct velocity field.
//     fn project_velocity(&mut self, c: f64);
//
//     /// Divergence: duxdx + duydy
//     fn divergence(&mut self) -> Array2<Self::Spectral>;
//
//     /// Solve pressure poisson equation
//     /// pseu: pseudo pressure ( in code it is pres\[1\] )
//     fn solve_pres(&mut self, f: &Array2<Self::Spectral>);
//
//     /// Update pressure term by divergence
//     fn update_pres(&mut self, div: &Array2<Self::Spectral>);
//
//     /// Update navier stokes residual
//     fn update_residual(&mut self);
//
//     /// Update pressure bc
//     fn update_pres_bc(&mut self);
// }

/// Container for Adjoint Navier-Stokes solver
pub struct Navier2DAdjoint<T, S> {
    /// Navier Stokes solver
    pub(crate) navier: Navier2D<T, S>,
    /// Field for derivatives and transforms
    pub field: Field2<T, S>,
    /// Temperature \[Adjoint Field, NS Residual\]
    pub temp: Field2<T, S>,
    /// Horizontal Velocity \[Adjoint Field, NS Residual\]
    pub velx: Field2<T, S>,
    /// Vertical Velocity \[Adjoint Field, NS Residual\]
    pub vely: Field2<T, S>,
    /// Pressure
    pub pres: Field2<T, S>,
    /// Pseudo Pressure
    pub pseu: Field2<T, S>,
    /// Field for temperature boundary condition
    pub tempbc: Option<Field2<T, S>>,
    /// Field for pressure boundary condition
    pub presbc: Option<Field2<T, S>>,
    /// Buffer
    pub(crate) rhs: Array2<T>,
    /// Parameter (e.g. diffusivities, ra, pr, ...)
    pub params: HashMap<&'static str, f64>,
    /// Time
    pub time: f64,
    /// Time step size
    pub dt: f64,
    /// Scale of physical dimension [scale_x, scale_y]
    pub scale: [f64; 2],
    /// Poisson solver for pressure
    pub(crate) solver_pres: Poisson<f64, 2>,
    /// Evaluate steady state in different norm for better convergence \[ux, uy, temp\]
    pub(crate) solver_norm: [Hholtz<f64, 2>; 3],
    /// Fields unsmoothed (for diffusion) \[ux, uy, temp\]
    pub(crate) fields_unsmoothed: [Array2<T>; 3],
    /// diagnostics like Nu, ...
    pub diagnostics: HashMap<String, Vec<f64>>,
    /// Time intervall for write fields
    /// If none, same intervall as diagnostics
    pub write_intervall: Option<f64>,
}

impl<T, S> Navier2DAdjoint<T, S>
where
    T: Zero,
{
    pub(crate) fn zero_rhs(&mut self) {
        for r in self.rhs.iter_mut() {
            *r = T::zero();
        }
    }
}

impl<T, S> Navier2DAdjoint<T, S>
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
        eval_nu(
            &mut self.navier.temp,
            &mut self.field,
            &self.tempbc,
            &self.scale,
        )
    }

    /// Returns volumetric Nusselt number
    /// $$
    /// Nuvol = \langle vely*T/kappa - dTdz \rangle\\_V
    /// $$
    ///
    /// # Panics
    /// If *ka* is not in params
    pub fn eval_nuvol(&mut self) -> f64 {
        use super::functions::eval_nuvol;
        let ka = self.params.get("ka").unwrap();
        eval_nuvol(
            &mut self.navier.temp,
            &mut self.navier.vely,
            &mut self.field,
            &self.tempbc,
            *ka,
            &self.scale,
        )
    }

    /// Returns Reynolds number based on kinetic energy
    ///
    /// # Panics
    /// If *nu* is not in params
    pub fn eval_re(&mut self) -> f64 {
        use super::functions::eval_re;
        let nu = self.params.get("nu").unwrap();
        eval_re(
            &mut self.navier.velx,
            &mut self.navier.vely,
            &mut self.field,
            *nu,
            &self.scale,
        )
    }
}
impl<T, S> Navier2DAdjoint<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Zero,
{
    /// Initialize velocity with fourier modes
    ///
    /// velx = amp \* sin(mx)cos(nx)
    /// vely = -amp \* cos(mx)sin(nx)
    pub fn set_velocity(&mut self, amp: f64, m: f64, n: f64) {
        apply_sin_cos(&mut self.velx, amp, m, n);
        apply_cos_sin(&mut self.vely, -amp, m, n);
    }
    /// Initialize temperature with fourier modes
    ///
    /// temp = -amp \* cos(mx)sin(ny)
    pub fn set_temperature(&mut self, amp: f64, m: f64, n: f64) {
        apply_cos_sin(&mut self.temp, -amp, m, n);
    }

    /// Initialize all fields with random disturbances
    pub fn init_random(&mut self, amp: f64) {
        random_field(&mut self.temp, amp);
        random_field(&mut self.velx, amp);
        random_field(&mut self.vely, amp);
        // // Remove bc base from temp
        // if let Some(x) = &self.tempbc {
        //     self.temp.v = &self.temp.v - &x.v;
        //     self.temp.forward();
        // }
    }

    /// Reset time
    pub fn reset_time(&mut self) {
        self.time = 0.;
    }
}

impl Navier2DAdjoint<f64, Space2R2r> {
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
    /// * `bc` - str for boundary conditions: "rbc", "hc"
    ///
    /// # Panics
    /// 'bc' type not recognized, see Arguments
    #[allow(clippy::similar_names)]
    pub fn new_confined(
        nx: usize,
        ny: usize,
        ra: f64,
        pr: f64,
        dt: f64,
        aspect: f64,
        bc: &str,
    ) -> Navier2DAdjoint<f64, Space2R2r> {
        let scale = [aspect, 1.];
        // diffusivities
        let nu = get_nu(ra, pr, scale[1] * 2.0);
        let ka = get_ka(ra, pr, scale[1] * 2.0);
        let mut params = HashMap::new();
        params.insert("ra", ra);
        params.insert("pr", pr);
        params.insert("nu", nu);
        params.insert("ka", ka);
        // fields
        let mut velx = Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny)));
        let mut vely = Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny)));
        let (mut temp, tempbc, presbc) = match bc {
            "rbc" => {
                let temp = Field2::new(&Space2::new(&cheb_neumann(nx), &cheb_dirichlet(ny)));
                let tempbc = bc_rbc(nx, ny);
                (temp, Some(tempbc), None)
            }
            "hc" => {
                let temp =
                    Field2::new(&Space2::new(&cheb_neumann(nx), &cheb_dirichlet_neumann(ny)));
                let tempbc = bc_hc(nx, ny);
                (temp, Some(tempbc), None)
            }
            _ => panic!("Boundary condition type {:?} not recognized!", bc),
        };
        // define underlying naver-stokes solver
        let navier = Navier2D::new_confined(nx, ny, ra, pr, DT_NAVIER, aspect, bc);
        // pressure
        let mut pres = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));
        let pseu = Field2::new(&Space2::new(&cheb_neumann(nx), &cheb_neumann(ny)));
        let field = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));

        // scale coordinates
        velx.scale(scale);
        vely.scale(scale);
        temp.scale(scale);
        pres.scale(scale);

        // pressure solver
        let solver_pres = Poisson::new(&pseu, [1.0 / scale[0].powi(2), 1.0 / scale[1].powi(2)]);

        // define smoother (hholtz type) (1-weight*D2)
        let smooth_ux = Hholtz::new(
            &velx,
            [
                WEIGHT_LAPLACIAN / scale[0].powi(2),
                WEIGHT_LAPLACIAN / scale[1].powi(2),
            ],
        );
        let smooth_uy = Hholtz::new(
            &vely,
            [
                WEIGHT_LAPLACIAN / scale[0].powi(2),
                WEIGHT_LAPLACIAN / scale[1].powi(2),
            ],
        );
        let smooth_temp = Hholtz::new(
            &temp,
            [
                WEIGHT_LAPLACIAN / scale[0].powi(2),
                WEIGHT_LAPLACIAN / scale[1].powi(2),
            ],
        );

        let solver_norm = [smooth_ux, smooth_uy, smooth_temp];
        let fields_unsmoothed = [
            Array2::zeros(field.vhat.raw_dim()),
            Array2::zeros(field.vhat.raw_dim()),
            Array2::zeros(field.vhat.raw_dim()),
        ];

        // Buffer for rhs
        let rhs = Array2::zeros(field.vhat.raw_dim());

        // Diagnostics
        let diagnostics = HashMap::new();

        // Return
        Navier2DAdjoint::<f64, Space2R2r> {
            navier,
            field,
            temp,
            velx,
            vely,
            pres,
            pseu,
            tempbc,
            presbc,
            rhs,
            params,
            time: 0.,
            dt,
            scale,
            solver_pres,
            solver_norm,
            fields_unsmoothed,
            diagnostics,
            write_intervall: None,
        }
    }
}

impl Navier2DAdjoint<Complex<f64>, Space2R2c> {
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
    ///
    /// * `bc` - str for boundary conditions: "rbc", "hc"
    ///
    /// # Panics
    /// 'bc' type not recognized, see Arguments
    #[allow(clippy::similar_names)]
    pub fn new_periodic(
        nx: usize,
        ny: usize,
        ra: f64,
        pr: f64,
        dt: f64,
        aspect: f64,
        bc: &str,
    ) -> Navier2DAdjoint<Complex<f64>, Space2R2c> {
        let scale = [aspect, 1.];
        // diffusivities
        let nu = get_nu(ra, pr, scale[1] * 2.0);
        let ka = get_ka(ra, pr, scale[1] * 2.0);
        let mut params = HashMap::new();
        params.insert("ra", ra);
        params.insert("pr", pr);
        params.insert("nu", nu);
        params.insert("ka", ka);
        // fields
        let mut velx = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny)));
        let mut vely = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny)));
        let (mut temp, tempbc, presbc) = match bc {
            "rbc" => {
                let temp = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny)));
                let tempbc = bc_rbc_periodic(nx, ny);
                (temp, Some(tempbc), None)
            }
            "hc" => {
                let temp = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet_neumann(ny)));
                let tempbc = bc_hc_periodic(nx, ny);
                (temp, Some(tempbc), None)
            }
            _ => panic!("Boundary condition type {:?} not recognized!", bc),
        };

        // define underlying naver-stokes solver
        let navier = Navier2D::new_periodic(nx, ny, ra, pr, DT_NAVIER, aspect, bc);
        // pressure
        let mut pres = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
        let pseu = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_neumann(ny)));
        // fields for derivatives
        let field = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));

        // scale coordinates
        velx.scale(scale);
        vely.scale(scale);
        temp.scale(scale);
        pres.scale(scale);

        // pressure solver
        let solver_pres = Poisson::new(&pseu, [1.0 / scale[0].powi(2), 1.0 / scale[1].powi(2)]);

        // define smoother (hholtz type) (1-weight*D2)
        let smooth_ux = Hholtz::new(
            &velx,
            [
                WEIGHT_LAPLACIAN / scale[0].powi(2),
                WEIGHT_LAPLACIAN / scale[1].powi(2),
            ],
        );
        let smooth_uy = Hholtz::new(
            &vely,
            [
                WEIGHT_LAPLACIAN / scale[0].powi(2),
                WEIGHT_LAPLACIAN / scale[1].powi(2),
            ],
        );
        let smooth_temp = Hholtz::new(
            &temp,
            [
                WEIGHT_LAPLACIAN / scale[0].powi(2),
                WEIGHT_LAPLACIAN / scale[1].powi(2),
            ],
        );

        let solver_norm = [smooth_ux, smooth_uy, smooth_temp];
        let fields_unsmoothed = [
            Array2::zeros(field.vhat.raw_dim()),
            Array2::zeros(field.vhat.raw_dim()),
            Array2::zeros(field.vhat.raw_dim()),
        ];

        // buffer for rhs
        let rhs = Array2::zeros(field.vhat.raw_dim());

        // Diagnostics
        let diagnostics = HashMap::new();

        // Return
        Navier2DAdjoint::<Complex<f64>, Space2R2c> {
            navier,
            field,
            temp,
            velx,
            vely,
            pres,
            pseu,
            tempbc,
            presbc,
            rhs,
            params,
            time: 0.,
            dt,
            scale,
            solver_pres,
            solver_norm,
            fields_unsmoothed,
            diagnostics,
            write_intervall: None,
        }
    }
}

impl<S> Integrate for Navier2DAdjoint<f64, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = f64>,
{
    /// Update 1 timestep
    fn update(&mut self) {

        // Calculate fields in physical space for convolution in convection terms
        let velx = self.navier.velx.space.backward_par(&self.navier.velx.vhat);
        let vely = self.navier.vely.space.backward_par(&self.navier.vely.vhat);
        let velx_adj = self.velx.space.backward_par(&self.velx.vhat);
        let vely_adj = self.vely.space.backward_par(&self.vely.vhat);
        let temp_adj = self.temp.space.backward_par(&self.temp.vhat);

        // // Forward step for residual

        // Get residual
        let velx_old = self.navier.velx.to_ortho();
        let vely_old = self.navier.vely.to_ortho();
        let temp_old = self.navier.temp.to_ortho();
        self.navier.update();
        let res_velx = (&self.navier.velx.to_ortho() - &velx_old) / self.navier.dt;
        let res_vely = (&self.navier.vely.to_ortho() - &vely_old) / self.navier.dt;
        let res_temp = (&self.navier.temp.to_ortho() - &temp_old) / self.navier.dt;

        self.navier.velx.from_ortho(&velx_old);
        self.navier.vely.from_ortho(&vely_old);
        self.navier.temp.from_ortho(&temp_old);

        // apply norm
        self.solver_norm[0].solve(&res_velx, &mut self.velx.vhat, 0);
        self.solver_norm[1].solve(&res_vely, &mut self.vely.vhat, 0);
        self.solver_norm[2].solve(&res_temp, &mut self.temp.vhat, 0);
        let rescale: f64 = (-1.0).into();
        self.velx.vhat *= rescale;
        self.vely.vhat *= rescale;
        self.temp.vhat *= rescale;

        // // Adjoint step

        // Calculate fields in physical space for convolution in convection terms
        // let velx = self.navier.velx.space.backward_par(&self.navier.velx.vhat);
        // let vely = self.navier.vely.space.backward_par(&self.navier.vely.vhat);
        // let velx_adj = self.velx.space.backward_par(&self.velx.vhat);
        // let vely_adj = self.vely.space.backward_par(&self.vely.vhat);
        // let temp_adj = self.temp.space.backward_par(&self.temp.vhat);

        // Solve Velocity
        self.solve_velx_adj(&velx, &vely, &velx_adj, &vely_adj, &temp_adj);
        self.solve_vely_adj(&velx, &vely, &velx_adj, &vely_adj, &temp_adj);

        // Projection step
        let div = self.div();
        self.solve_pres(&div);
        self.correct_velocity(1.0);
        self.update_pres(&div);

        // Solve Temperature
        self.solve_temp_adj(&velx, &vely, &velx_adj, &vely_adj, &temp_adj);

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
        let flowname = format!("data/adjoint{:0>8.2}.h5", self.time);
        let io_name = "data/info_adjoint.txt";
        self.callback_from_filename(&flowname, io_name, false, self.write_intervall);
    }

    fn exit(&mut self) -> bool {
        // Break if divergence is nan
        let div = self.div();
        if norm_l2_f64(&div).is_nan() {
            return true;
        }
        false
    }
}

//
// impl<T, S> Navier2DAdjoint<T, S>
// where
//     T: num_traits::Zero,
//     S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
// {
//     /// Rescale x & y coordinates of fields.
//     /// Only affects output of files
//     fn _scale(&mut self) {
//         for field in &mut [
//             &mut self.temp[0],
//             &mut self.ux[0],
//             &mut self.uy[0],
//             &mut self.pres,
//         ] {
//             field.x[0] *= self.scale[0];
//             field.x[1] *= self.scale[1];
//             field.dx[0] *= self.scale[0];
//             field.dx[1] *= self.scale[1];
//         }
//         for field in &mut [&mut self.temp[1], &mut self.ux[1], &mut self.uy[1]] {
//             field.x[0] *= self.scale[0];
//             field.x[1] *= self.scale[1];
//             field.dx[0] *= self.scale[0];
//             field.dx[1] *= self.scale[1];
//         }
//     }

//     /// Set boundary condition field for temperature
//     pub fn set_temp_bc(&mut self, fieldbc: Field2<T, S>)
//     where
//         T: Clone,
//     {
//         self.tempbc = Some(fieldbc.clone());
//         self.navier.tempbc = Some(fieldbc);
//     }
//
//     // /// Set boundary condition field for pressure
//     // pub fn set_pres_bc(&mut self, fieldbc: Field2<T, S>) {
//     //     self.presbc = Some(fieldbc);
//     // }
//
//     fn zero_rhs(&mut self) {
//         for r in self.rhs.iter_mut() {
//             *r = T::zero();
//         }
//     }
// }
//
// macro_rules! impl_navier_convection {
//     ($s: ty) => {
//         impl<S> NavierConvectionAdjoint for Navier2DAdjoint<$s, S>
//         where
//             S: BaseSpace<f64, 2, Physical = f64, Spectral = $s>,
//         {
//             type Physical = f64;
//             type Spectral = $s;
//
//             /// Convection term for ux
//             fn conv_ux(
//                 &mut self,
//                 ux: &Array2<Self::Physical>,
//                 uy: &Array2<Self::Physical>,
//                 t: &Array2<Self::Physical>,
//             ) -> Array2<Self::Spectral> {
//                 // + ux * dudx + uy * dudy
//                 let mut conv =
//                     conv_term(&self.ux[1], &mut self.field, ux, [1, 0], Some(self.scale));
//                 conv += &conv_term(&self.ux[1], &mut self.field, uy, [0, 1], Some(self.scale));
//                 // + adjoint contributions
//                 conv += &conv_term(&self.ux[1], &mut self.field, ux, [1, 0], Some(self.scale));
//                 conv += &conv_term(&self.uy[1], &mut self.field, uy, [1, 0], Some(self.scale));
//                 conv -= &conv_term(&self.temp[0], &mut self.field, t, [1, 0], Some(self.scale));
//                 if let Some(field) = &self.tempbc {
//                     conv -= &conv_term(&field, &mut self.field, t, [1, 0], Some(self.scale));
//                 }
//                 // if let Some(x) = &self.tempbc {
//                 //     conv += &conv_term(
//                 //         &self.temp[1],
//                 //         &mut self.field,
//                 //         &x.v,
//                 //         [1, 0],
//                 //         Some(self.scale),
//                 //     );
//                 // }
//                 // -> spectral space
//                 self.field.v.assign(&conv);
//                 self.field.forward();
//                 if self.dealias {
//                     dealias(&mut self.field);
//                 }
//                 self.field.vhat.to_owned()
//             }
//
//             /// Convection term for uy
//             fn conv_uy(
//                 &mut self,
//                 ux: &Array2<Self::Physical>,
//                 uy: &Array2<Self::Physical>,
//                 t: &Array2<Self::Physical>,
//             ) -> Array2<Self::Spectral> {
//                 // + ux * dudx + uy * dudy
//                 let mut conv =
//                     conv_term(&self.uy[1], &mut self.field, ux, [1, 0], Some(self.scale));
//                 conv += &conv_term(&self.uy[1], &mut self.field, uy, [0, 1], Some(self.scale));
//                 // + adjoint contributions
//                 conv += &conv_term(&self.ux[1], &mut self.field, ux, [0, 1], Some(self.scale));
//                 conv += &conv_term(&self.uy[1], &mut self.field, uy, [0, 1], Some(self.scale));
//                 conv -= &conv_term(&self.temp[0], &mut self.field, t, [0, 1], Some(self.scale));
//                 if let Some(field) = &self.tempbc {
//                     conv -= &conv_term(&field, &mut self.field, t, [0, 1], Some(self.scale));
//                 }
//                 // if let Some(x) = &self.tempbc {
//                 //     conv += &conv_term(
//                 //         &self.temp[1],
//                 //         &mut self.field,
//                 //         &x.v,
//                 //         [0, 1],
//                 //         Some(self.scale),
//                 //     );
//                 // }
//                 // -> spectral space
//                 self.field.v.assign(&conv);
//                 self.field.forward();
//                 if self.dealias {
//                     dealias(&mut self.field);
//                 }
//                 self.field.vhat.to_owned()
//             }
//
//             /// Convection term for temperature
//             fn conv_temp(
//                 &mut self,
//                 ux: &Array2<Self::Physical>,
//                 uy: &Array2<Self::Physical>,
//             ) -> Array2<Self::Spectral> {
//                 // + ux * dTdx + uy * dTdy
//                 let mut conv =
//                     conv_term(&self.temp[1], &mut self.field, ux, [1, 0], Some(self.scale));
//                 conv += &conv_term(&self.temp[1], &mut self.field, uy, [0, 1], Some(self.scale));
//                 // -> spectral space
//                 self.field.v.assign(&conv);
//                 self.field.forward();
//                 if self.dealias {
//                     dealias(&mut self.field);
//                 }
//                 self.field.vhat.to_owned()
//             }
//
//             /// Solve adjoint horizontal momentum equation
//             fn solve_ux(
//                 &mut self,
//                 ux: &Array2<Self::Physical>,
//                 uy: &Array2<Self::Physical>,
//                 temp: &Array2<Self::Physical>,
//             ) {
//                 self.zero_rhs();
//                 // + old field
//                 self.rhs += &self.ux[0].to_ortho();
//                 // + pres
//                 self.rhs -= &(self.pres.gradient([1, 0], Some(self.scale)) * self.dt);
//                 // if let Some(field) = &self.presbc {
//                 //     self.rhs -= &(field.gradient([1, 0], Some(self.scale)) * self.dt);
//                 // }
//                 // + convection
//                 let conv = self.conv_ux(ux, uy, temp);
//                 self.rhs += &(conv * self.dt);
//                 // + diffusion
//                 self.rhs += &(self.ux[1].gradient([2, 0], Some(self.scale)) * self.dt * self.nu);
//                 self.rhs += &(self.ux[1].gradient([0, 2], Some(self.scale)) * self.dt * self.nu);
//                 // update ux
//                 self.ux[0].from_ortho(&self.rhs);
//             }
//
//             /// Solve adjoint vertical momentum equation
//             fn solve_uy(
//                 &mut self,
//                 ux: &Array2<Self::Physical>,
//                 uy: &Array2<Self::Physical>,
//                 temp: &Array2<Self::Physical>,
//             ) {
//                 self.zero_rhs();
//                 // + old field
//                 self.rhs += &self.uy[0].to_ortho();
//                 // + pres
//                 self.rhs -= &(self.pres.gradient([0, 1], Some(self.scale)) * self.dt);
//                 // if let Some(field) = &self.presbc {
//                 //     self.rhs -= &(field.gradient([0, 1], Some(self.scale)) * self.dt);
//                 // }
//                 // + convection
//                 let conv = self.conv_uy(ux, uy, temp);
//                 self.rhs += &(conv * self.dt);
//                 // + temp bc (Rayleigh--Benard type)
//                 // self.rhs += &(self.temp[1].to_ortho() * 0.5 * self.dt);
//                 // + diffusion
//                 self.rhs += &(self.uy[1].gradient([2, 0], Some(self.scale)) * self.dt * self.nu);
//                 self.rhs += &(self.uy[1].gradient([0, 2], Some(self.scale)) * self.dt * self.nu);
//                 // update uy
//                 self.uy[0].from_ortho(&self.rhs);
//             }
//
//             /// Solve adjoint temperature equation
//             fn solve_temp(&mut self, ux: &Array2<Self::Physical>, uy: &Array2<Self::Physical>) {
//                 self.zero_rhs();
//                 // + old field
//                 self.rhs += &self.temp[0].to_ortho();
//                 // + convection
//                 let conv = self.conv_temp(ux, uy);
//                 self.rhs += &(conv * self.dt);
//                 // + buoyancy (adjoint)
//                 let buoy = self.uy[1].to_ortho();
//                 self.rhs += &(buoy * self.dt);
//                 // + diffusion
//                 self.rhs += &(self.temp[1].gradient([2, 0], Some(self.scale)) * self.dt * self.ka);
//                 self.rhs += &(self.temp[1].gradient([0, 2], Some(self.scale)) * self.dt * self.ka);
//                 // update temp
//                 self.temp[0].from_ortho(&self.rhs);
//             }
//
//             /// Correct velocity field.
//             ///
//             /// uxnew = ux - c*dpdx
//             ///
//             /// uynew = uy - c*dpdy
//             #[allow(clippy::similar_names)]
//             fn project_velocity(&mut self, c: f64) {
//                 let dpdx = self.pseu.gradient([1, 0], Some(self.scale));
//                 let dpdy = self.pseu.gradient([0, 1], Some(self.scale));
//                 let old_ux = self.ux[0].vhat.clone();
//                 let old_uy = self.uy[0].vhat.clone();
//                 self.ux[0].from_ortho(&dpdx);
//                 self.uy[0].from_ortho(&dpdy);
//                 let cinto: Self::Spectral = (-c).into();
//                 self.ux[0].vhat *= cinto;
//                 self.uy[0].vhat *= cinto;
//                 self.ux[0].vhat += &old_ux;
//                 self.uy[0].vhat += &old_uy;
//             }
//
//             /// Divergence: duxdx + duydy
//             fn divergence(&mut self) -> Array2<Self::Spectral> {
//                 self.zero_rhs();
//                 self.rhs += &self.ux[0].gradient([1, 0], Some(self.scale));
//                 self.rhs += &self.uy[0].gradient([0, 1], Some(self.scale));
//                 self.rhs.to_owned()
//             }
//
//             /// Solve pressure poisson equation
//             ///
//             /// D2 pres = f
//             ///
//             /// pseu: pseudo pressure ( in code it is pres\[1\] )
//             fn solve_pres(&mut self, f: &Array2<Self::Spectral>) {
//                 self.solver[0].solve(&f, &mut self.pseu.vhat, 0);
//                 // Singularity
//                 self.pseu.vhat[[0, 0]] = Self::Spectral::zero();
//             }
//
//             fn update_pres(&mut self, _div: &Array2<Self::Spectral>) {
//                 // self.pres.vhat = &self.pres.vhat - &(div * self.nu);
//                 let inv_dt: Self::Spectral = (1. / self.dt).into();
//                 self.pres.vhat += &(&self.pseu.to_ortho() * inv_dt);
//             }
//
//             /// Update navier stokes residual
//             fn update_residual(&mut self) {
//                 // Update residual
//                 self.navier.ux.vhat.assign(&self.ux[0].vhat);
//                 self.navier.uy.vhat.assign(&self.uy[0].vhat);
//                 self.navier.temp.vhat.assign(&self.temp[0].vhat);
//                 self.navier.update();
//                 let res = (&self.navier.ux.vhat - &self.ux[0].vhat) / self.navier.dt;
//                 self.navier.ux.vhat.assign(&res);
//                 let res = (&self.navier.uy.vhat - &self.uy[0].vhat) / self.navier.dt;
//                 self.navier.uy.vhat.assign(&res);
//                 let res = (&self.navier.temp.vhat - &self.temp[0].vhat) / self.navier.dt;
//                 self.navier.temp.vhat.assign(&res);
//                 // Save "unsmoothed" residual fields
//                 self.fields_unsmoothed[0].assign(&self.navier.ux.to_ortho());
//                 self.fields_unsmoothed[1].assign(&self.navier.uy.to_ortho());
//                 self.fields_unsmoothed[2].assign(&self.navier.temp.to_ortho());
//                 // Smooth fields
//                 self.smoother[0].solve(&self.fields_unsmoothed[0], &mut self.ux[1].vhat, 0);
//                 self.smoother[1].solve(&self.fields_unsmoothed[1], &mut self.uy[1].vhat, 0);
//                 self.smoother[2].solve(&self.fields_unsmoothed[2], &mut self.temp[1].vhat, 0);
//                 let rescale: Self::Spectral = (-1.0).into();
//                 self.ux[1].vhat *= rescale;
//                 self.uy[1].vhat *= rescale;
//                 self.temp[1].vhat *= rescale;
//             }
//
//             fn update_pres_bc(&mut self) {
//                 use ndarray::Array1;
//                 use ndarray::Axis;
//                 use num_traits::Pow;
//
//                 /// Return a, b of a*x**2 + b*x
//                 /// from derivatives at the boundaries
//                 fn parabola_coeff(df_l: f64, df_r: f64, x: &Array1<f64>) -> (f64, f64) {
//                     let x_l = x[0];
//                     let x_r = x[x.len() - 1];
//                     let a = 0.5 * (df_r - df_l) / (x_r - x_l);
//                     let b = df_l - 2. * a * x_l;
//                     (a, b)
//                 }
//
//                 if let Some(ref mut fieldbc) = self.presbc {
//                     // Create base and field
//                     self.field
//                         .vhat
//                         .assign(&self.temp[1].gradient([0, 1], Some(self.scale)));
//                     self.field.backward();
//                     let y = &fieldbc.x[1];
//                     for (i, mut axis) in fieldbc.v.axis_iter_mut(Axis(0)).enumerate() {
//                         // let bot = self.tempbc.as_ref().unwrap().v[[i, 0]] * self.field.v[[i, 0]];
//                         // let top = self.tempbc.as_ref().unwrap().v[[i, y.len() - 1]]
//                         //     * self.field.v[[i, y.len() - 1]];
//
//                         let bot = 0.5 * self.field.v[[i, 0]];
//                         let top = -0.5 * self.field.v[[i, y.len() - 1]];
//                         let (a, b) = parabola_coeff(bot, top, y);
//                         let parabola = a * y.mapv(|y| y.pow(2)) + b * y;
//                         axis.assign(&parabola);
//                     }
//
//                     // Transform
//                     fieldbc.forward();
//                     fieldbc.backward();
//                 }
//             }
//         }
//     };
// }
//
// impl_navier_convection!(f64);
// impl_navier_convection!(Complex<f64>);
//
// macro_rules! impl_integrate {
//     ($s: ty, $norm: ident) => {
//         impl<S> Integrate for Navier2DAdjoint<$s, S>
//         where
//             S: BaseSpace<f64, 2, Physical = f64, Spectral = $s>,
//         {
//             /// Update of adjoint Navier Stokes
//             fn update(&mut self) {
//                 // Convection fields
//                 self.ux[0].backward();
//                 self.uy[0].backward();
//                 self.temp[1].backward();
//                 let ux = self.ux[0].v.to_owned();
//                 let uy = self.uy[0].v.to_owned();
//                 let temp = self.temp[1].v.to_owned();
//
//                 // Update residual
//                 self.update_residual();
//
//                 // Update pressure bc
//                 self.update_pres_bc();
//
//                 // Solve Velocity
//                 self.solve_ux(&ux, &uy, &temp);
//                 self.solve_uy(&ux, &uy, &temp);
//
//                 // Projection
//                 let div = self.divergence();
//                 self.solve_pres(&div);
//                 self.project_velocity(1.0);
//                 self.update_pres(&div);
//
//                 // Solve Temperature
//                 self.solve_temp(&ux, &uy);
//
//                 // update time
//                 self.time += self.dt;
//             }
//
//             fn get_time(&self) -> f64 {
//                 self.time
//             }
//
//             fn get_dt(&self) -> f64 {
//                 self.dt
//             }
//
//             fn callback(&mut self) {
//                 use std::io::Write;
//                 std::fs::create_dir_all("data").unwrap();
//
//                 // Write flow field
//                 //let fname = format!("data/adjoint{:.*}.h5", 3, self.time);
//                 let fname = format!("data/adjoint{:0>8.2}.h5", self.time);
//                 if let Some(dt_save) = &self.write_intervall {
//                     if (self.time % dt_save) < self.dt / 2.
//                         || (self.time % dt_save) > dt_save - self.dt / 2.
//                     {
//                         self.write(&fname);
//                     }
//                 } else {
//                     self.write(&fname);
//                 }
//
//                 // I/O
//                 let div = self.divergence();
//                 let nu = self.eval_nu();
//                 let nuvol = self.eval_nuvol();
//                 let re = self.eval_re();
//                 //println!("Diagnostics:");
//                 println!(
//                     "time = {:4.2}      |div| = {:4.2e}     Nu = {:5.3e}     Nuv = {:5.3e}    Re = {:5.3e}",
//                     self.time,
//                     $norm(&div),
//                     nu,
//                     nuvol,
//                     re,
//                 );
//
//                 let mut file = std::fs::OpenOptions::new()
//                     .write(true)
//                     .append(true)
//                     .create(true)
//                     .open("data/adjoint_info.txt")
//                     .unwrap();
//                 if let Err(e) = writeln!(file, "{} {} {} {}", self.time, nu, nuvol, re) {
//                     eprintln!("Couldn't write to file: {}", e);
//                 }
//                 // Write residual
//                 let res_u = $norm(&self.fields_unsmoothed[0]);
//                 let res_v = $norm(&self.fields_unsmoothed[1]);
//                 let res_t = $norm(&self.fields_unsmoothed[2]);
//                 let res_total = res_u + res_v + res_t;
//                 let res_u2 = $norm(&self.ux[1].vhat);
//                 let res_v2 = $norm(&self.uy[1].vhat);
//                 let res_t2 = $norm(&self.temp[1].vhat);
//                 let res_total2 = (res_u2 + res_v2 + res_t2);
//                 println!("|U| = {:10.2e}", res_u2,);
//                 println!("|V| = {:10.2e}", res_v2,);
//                 println!("|T| = {:10.2e}", res_t2,);
//                 let mut residual = std::fs::OpenOptions::new()
//                     .write(true)
//                     .append(true)
//                     .create(true)
//                     .open("data/residual.txt")
//                     .unwrap();
//                 if let Err(e) = writeln!(residual, "{} {} {}", self.time, res_total, res_total2) {
//                     eprintln!("Couldn't write to file: {}", e);
//                 }
//             }
//
//             fn exit(&mut self) -> bool {
//                 // Break if divergence is nan
//                 let div = self.divergence();
//                 if $norm(&div).is_nan() {
//                     println!("Divergence is nan!");
//                     return true;
//                 }
//                 // Break if residual is small enough
//                 let res_u = $norm(&self.ux[1].vhat);
//                 let res_v = $norm(&self.uy[1].vhat);
//                 let res_t = $norm(&self.temp[1].vhat);
//                 if res_u + res_v + res_t < self.res_tol {
//                     println!("Residual reached!");
//                     return true;
//                 }
//                 false
//             }
//         }
//
//     };
// }
// impl_integrate!(f64, norm_l2_f64);
// impl_integrate!(Complex<f64>, norm_l2_c64);
//
// fn norm_l2_f64(array: &Array2<f64>) -> f64 {
//     array.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
// }
//
// fn norm_l2_c64(array: &Array2<Complex<f64>>) -> f64 {
//     array
//         .iter()
//         .map(|x| x.re.powi(2) + x.im.powi(2))
//         .sum::<f64>()
//         .sqrt()
// }

// fn norm_l2_f64(q1: &Array2<f64>, q2: &Array2<f64>) -> f64 {
//     q1.iter().zip(q2).map(|(x, y)| x * y).sum::<f64>().sqrt()
// }
//
// fn norm_l2_c64(q1: &Array2<Complex<f64>>, q2: &Array2<Complex<f64>>) -> f64 {
//     q1.iter()
//         .zip(q2)
//         .map(|(x, y)| {
//             let z = x * y;
//             z.re.powi(2) + z.im.powi(2)
//         })
//         .sum::<f64>()
//         .sqrt()
// }
//
// impl<T, S> Navier2DAdjoint<T, S>
// where
//     S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
//     T: crate::types::Scalar + Mul<f64, Output = T> + Div<f64, Output = T>,
// {
//     /// Returns Nusselt number (heat flux at the plates)
//     /// $$
//     /// Nu = \langle - dTdz \rangle\\_x (0/H))
//     /// $$
//     pub fn eval_nu(&mut self) -> f64 {
//         use super::functions::eval_nu;
//         eval_nu(
//             &mut self.temp[0],
//             &mut self.field,
//             &self.tempbc,
//             &self.scale,
//         )
//     }
//
//     /// Returns volumetric Nusselt number
//     /// $$
//     /// Nuvol = \langle uy*T/kappa - dTdz \rangle\\_V
//     /// $$
//     pub fn eval_nuvol(&mut self) -> f64 {
//         use super::functions::eval_nuvol;
//         eval_nuvol(
//             &mut self.temp[0],
//             &mut self.uy[0],
//             &mut self.field,
//             &self.tempbc,
//             self.ka,
//             &self.scale,
//         )
//     }
//
//     /// Returns Reynolds number based on kinetic energy
//     pub fn eval_re(&mut self) -> f64 {
//         use super::functions::eval_re;
//         eval_re(
//             &mut self.ux[0],
//             &mut self.uy[0],
//             &mut self.field,
//             self.nu,
//             &self.scale,
//         )
//     }
//
//     /// Initialize velocity with fourier modes
//     ///
//     /// ux = amp \* sin(mx)cos(nx)
//     /// uy = -amp \* cos(mx)sin(nx)
//     pub fn set_velocity(&mut self, amp: f64, m: f64, n: f64) {
//         apply_sin_cos(&mut self.ux[0], amp, m, n);
//         apply_cos_sin(&mut self.uy[0], -amp, m, n);
//     }
//     /// Initialize temperature with fourier modes
//     ///
//     /// temp = -amp \* cos(mx)sin(ny)
//     pub fn set_temperature(&mut self, amp: f64, m: f64, n: f64) {
//         apply_cos_sin(&mut self.temp[0], -amp, m, n);
//     }
//
//     /// Reset time
//     pub fn reset_time(&mut self) {
//         self.time = 0.;
//         self.navier.time = 0.;
//     }
// }
//
// macro_rules! impl_read_write_navier {
//     ($s: ty) => {
//         impl<S> Navier2DAdjoint<$s, S>
//         where
//             S: BaseSpace<f64, 2, Physical = f64, Spectral = $s>,
//         {
//             /// Restart from file
//             pub fn read(&mut self, filename: &str) {
//                 self.temp[0].read(&filename, Some("temp"));
//                 self.ux[0].read(&filename, Some("ux"));
//                 self.uy[0].read(&filename, Some("uy"));
//                 // Read scalars
//                 self.time = read_scalar_from_hdf5::<f64>(&filename, "time", None).unwrap();
//                 println!(" <== {:?}", filename);
//             }
//
//             /// Write Field data to hdf5 file
//             pub fn write(&mut self, filename: &str) {
//                 let result = self.write_return_result(filename);
//                 match result {
//                     Ok(_) => println!(" ==> {:?}", filename),
//                     Err(_) => println!("Error while writing file {:?}.", filename),
//                 }
//             }
//
//             fn write_return_result(&mut self, filename: &str) -> Result<()> {
//                 self.temp[0].backward();
//                 self.ux[0].backward();
//                 self.uy[0].backward();
//                 self.pres.backward();
//                 // Add boundary contribution
//                 if let Some(x) = &self.tempbc {
//                     self.temp[0].v = &self.temp[0].v + &x.v;
//                 }
//                 // Field
//                 self.temp[0].write(&filename, Some("temp"));
//                 self.ux[0].write(&filename, Some("ux"));
//                 self.uy[0].write(&filename, Some("uy"));
//                 self.pres.write(&filename, Some("pres"));
//                 // Write scalars
//                 write_scalar_to_hdf5(&filename, "time", None, self.time).ok();
//                 write_scalar_to_hdf5(&filename, "ra", None, self.ra).ok();
//                 write_scalar_to_hdf5(&filename, "pr", None, self.pr).ok();
//                 write_scalar_to_hdf5(&filename, "nu", None, self.nu).ok();
//                 write_scalar_to_hdf5(&filename, "kappa", None, self.ka).ok();
//                 // Undo addition of bc
//                 if self.tempbc.is_some() {
//                     self.temp[0].backward();
//                 }
//                 Ok(())
//             }
//         }
//     };
// }
//
// impl_read_write_navier!(f64);
// impl_read_write_navier!(Complex<f64>);
