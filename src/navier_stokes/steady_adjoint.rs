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
//! J. Fluid Mech., In Print.
#![allow(clippy::too_many_lines)]
use super::boundary_conditions::{bc_hc, bc_hc_periodic, bc_rbc, bc_rbc_periodic};
use super::functions::{apply_cos_sin, apply_sin_cos, random_field};
use super::functions::{get_ka, get_nu, norm_l2_c64, norm_l2_f64};
use super::navier::{Space2R2c, Space2R2r};
use crate::bases::{cheb_dirichlet, cheb_dirichlet_neumann, cheb_neumann, chebyshev, fourier_r2c};
use crate::field::{BaseSpace, Field2, Space2};
use crate::solver::{Hholtz, HholtzAdi, Poisson, Solve};
use crate::types::Scalar;
use crate::Integrate;
use ndarray::Array2;
use num_complex::Complex;
use num_traits::Zero;
use std::collections::HashMap;
use std::ops::{Div, Mul};

/// Tolerance criteria for residual
const RES_TOL: f64 = 1e-7;
/// Laplacian weight in norm
const WEIGHT_LAPLACIAN: f64 = 1e-1;
/// Timestep of forward navier integration
const DT_NAVIER: f64 = 1e-3;

/// Container for Adjoint Navier-Stokes solver
pub struct Navier2DAdjoint<T, S> {
    /// Field for derivatives and transforms
    pub field: Field2<T, S>,
    /// Temperature
    pub temp: Field2<T, S>,
    /// Horizontal Velocity
    pub velx: Field2<T, S>,
    /// Vertical Velocity
    pub vely: Field2<T, S>,
    /// Pressure
    pub pres: Field2<T, S>,
    /// Temperature (adjoint)
    pub temp_adj: Field2<T, S>,
    /// Horizontal Velocity (adjoint)
    pub velx_adj: Field2<T, S>,
    /// Vertical Velocity (adjoint)
    pub vely_adj: Field2<T, S>,
    /// Pressure(adjoint)
    pub pres_adj: Field2<T, S>,
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
    /// Helmholtz solver for implicit diffusion term
    pub(crate) solver_hholtz: [HholtzAdi<f64, 2>; 3],
    /// Poisson solver for pressure
    pub(crate) solver_pres: Poisson<f64, 2>,
    /// Evaluate steady state in different norm for better convergence \[ux, uy, temp\]
    pub(crate) solver_norm: [Hholtz<f64, 2>; 3],
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
        eval_nu(&mut self.temp, &mut self.field, &self.tempbc, &self.scale)
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
            &mut self.temp,
            &mut self.vely,
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
            &mut self.velx,
            &mut self.vely,
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

        // pressure
        let mut pres = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));
        let pseu = Field2::new(&Space2::new(&cheb_neumann(nx), &cheb_neumann(ny)));
        let field = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));

        // scale coordinates
        velx.scale(scale);
        vely.scale(scale);
        temp.scale(scale);
        pres.scale(scale);

        // Adjoint fields
        let velx_adj = velx.clone();
        let vely_adj = vely.clone();
        let temp_adj = temp.clone();
        let pres_adj = pres.clone();

        // Helmholtz solver
        let solver_velx = HholtzAdi::new(
            &velx,
            [
                DT_NAVIER * nu / scale[0].powi(2),
                DT_NAVIER * nu / scale[1].powi(2),
            ],
        );
        let solver_vely = HholtzAdi::new(
            &vely,
            [
                DT_NAVIER * nu / scale[0].powi(2),
                DT_NAVIER * nu / scale[1].powi(2),
            ],
        );
        let solver_temp = HholtzAdi::new(
            &temp,
            [
                DT_NAVIER * ka / scale[0].powi(2),
                DT_NAVIER * ka / scale[1].powi(2),
            ],
        );
        let solver_hholtz = [solver_velx, solver_vely, solver_temp];

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

        // Buffer for rhs
        let rhs = Array2::zeros(field.vhat.raw_dim());

        // Diagnostics
        let diagnostics = HashMap::new();

        // Return
        Navier2DAdjoint::<f64, Space2R2r> {
            field,
            temp,
            velx,
            vely,
            pres,
            temp_adj,
            velx_adj,
            vely_adj,
            pres_adj,
            pseu,
            tempbc,
            presbc,
            rhs,
            params,
            time: 0.,
            dt,
            scale,
            solver_hholtz,
            solver_pres,
            solver_norm,
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

        // Adjoint fields
        let velx_adj = velx.clone();
        let vely_adj = vely.clone();
        let temp_adj = temp.clone();
        let pres_adj = pres.clone();

        // Helmholtz solver
        let solver_velx = HholtzAdi::new(
            &velx,
            [
                DT_NAVIER * nu / scale[0].powi(2),
                dt * nu / scale[1].powi(2),
            ],
        );
        let solver_vely = HholtzAdi::new(
            &vely,
            [
                DT_NAVIER * nu / scale[0].powi(2),
                dt * nu / scale[1].powi(2),
            ],
        );
        let solver_temp = HholtzAdi::new(
            &temp,
            [
                DT_NAVIER * ka / scale[0].powi(2),
                dt * ka / scale[1].powi(2),
            ],
        );
        let solver_hholtz = [solver_velx, solver_vely, solver_temp];

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

        // buffer for rhs
        let rhs = Array2::zeros(field.vhat.raw_dim());

        // Diagnostics
        let diagnostics = HashMap::new();

        // Return
        Navier2DAdjoint::<Complex<f64>, Space2R2c> {
            field,
            temp,
            velx,
            vely,
            pres,
            temp_adj,
            velx_adj,
            vely_adj,
            pres_adj,
            pseu,
            tempbc,
            presbc,
            rhs,
            params,
            time: 0.,
            dt,
            scale,
            solver_hholtz,
            solver_pres,
            solver_norm,
            diagnostics,
            write_intervall: None,
        }
    }
}

macro_rules! impl_integrate_for_navier_adjoint {
    ($s: ty, $norm: ident) => {
        impl<S> Integrate for Navier2DAdjoint<$s, S>
        where
            S: BaseSpace<f64, 2, Physical = f64, Spectral = $s>,
        {
            /// Update 1 timestep
            fn update(&mut self) {

                // *** Forward step to calculate residual ***
                {
                    let dt_navier = DT_NAVIER;

                    // Calculate fields in physical space for convolution in convection terms
                    let velx = self.velx.space.backward_par(&self.velx.vhat);
                    let vely = self.vely.space.backward_par(&self.vely.vhat);

                    // Save for residual computation: res = (unew - uold) / dt
                    let velx_old = self.velx.to_ortho();
                    let vely_old = self.vely.to_ortho();
                    let temp_old = self.temp.to_ortho();

                    // Solve Velocity
                    self.solve_velx(&velx, &vely, dt_navier);
                    self.solve_vely(&velx, &vely, dt_navier);

                    // Projection
                    let div = self.div();
                    self.solve_pres(&div);
                    self.correct_velocity(1.0);
                    self.update_pres(&div, dt_navier);

                    // Solve Temperature
                    self.solve_temp(&velx, &vely, dt_navier);

                    // residual
                    let res_velx = (&self.velx.to_ortho() - &velx_old) / dt_navier;
                    let res_vely = (&self.vely.to_ortho() - &vely_old) / dt_navier;
                    let res_temp = (&self.temp.to_ortho() - &temp_old) / dt_navier;

                    // apply norm to residual
                    self.solver_norm[0].solve(&res_velx, &mut self.velx_adj.vhat, 0);
                    self.solver_norm[1].solve(&res_vely, &mut self.vely_adj.vhat, 0);
                    self.solver_norm[2].solve(&res_temp, &mut self.temp_adj.vhat, 0);
                    let rescale: $s = (-1.0).into();
                    self.velx_adj.vhat *= rescale;
                    self.vely_adj.vhat *= rescale;
                    self.temp_adj.vhat *= rescale;
                }

                // *** Adjoint step ***
                {
                    // Calculate fields in physical space for convolution in convection terms
                    let velx = self.velx.space.backward_par(&self.velx.vhat);
                    let vely = self.vely.space.backward_par(&self.vely.vhat);
                    let velx_adj = self.velx_adj.space.backward_par(&self.velx_adj.vhat);
                    let vely_adj = self.vely_adj.space.backward_par(&self.vely_adj.vhat);
                    let temp_adj = self.temp_adj.space.backward_par(&self.temp_adj.vhat);

                    // Solve Velocity
                    self.solve_velx_adj(&velx, &vely, &velx_adj, &vely_adj, &temp_adj);
                    self.solve_vely_adj(&velx, &vely, &velx_adj, &vely_adj, &temp_adj);

                    // Projection step
                    let div = self.div();
                    self.solve_pres(&div);
                    self.correct_velocity(1.0);
                    self.update_pres_adj(&div);

                    // Solve Temperature
                    self.solve_temp_adj(&velx, &vely, &velx_adj, &vely_adj, &temp_adj);
                }

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
                use super::steady_adjoint_eq::DivNorm;
                // Break if divergence is nan
                let div = self.div();
                if $norm(&div).is_nan() {
                    return true;
                }
                // Break if residual is below treshold
                let [res_u, res_v, res_t] = self.norm_residual();
                if (res_u + res_v + res_t) / 3. < RES_TOL {
                    println!("Steady state converged!");
                    return true;
                }
                false
            }
        }
    };
}

impl_integrate_for_navier_adjoint!(f64, norm_l2_f64);
impl_integrate_for_navier_adjoint!(Complex<f64>, norm_l2_c64);
