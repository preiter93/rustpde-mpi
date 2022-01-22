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
//!     let mut navier = Navier2D::new_confined(nx, ny, ra, pr, dt, aspect, "rbc");
//!     // // Want to restart?
//!     // navier.read_unwrap("data/flow100.000.h5"", None);
//!     // Write first field
//!     navier.callback();
//!     integrate(&mut navier, 100., Some(1.0));
//! }
//! ```
use super::boundary_conditions::{bc_hc, bc_hc_periodic, bc_rbc, bc_rbc_periodic};
use super::boundary_conditions::{pres_bc_rbc, pres_bc_rbc_periodic};
use super::functions::{apply_cos_sin, apply_sin_cos, random_field};
use super::functions::{get_ka, get_nu, norm_l2_c64, norm_l2_f64};
use super::statistics::Statistics;
use crate::bases::{cheb_dirichlet, cheb_dirichlet_neumann, cheb_neumann, chebyshev, fourier_r2c};
use crate::bases::{BaseR2c, BaseR2r};
use crate::field::{BaseSpace, Field2, Space2};
use crate::solver::{HholtzAdi, Poisson};
use crate::types::Scalar;
use crate::Integrate;
use ndarray::Array2;
use num_complex::Complex;
use num_traits::Zero;
use std::collections::HashMap;
use std::ops::{Div, Mul};

pub(crate) type Space2R2r = Space2<BaseR2r<f64>, BaseR2r<f64>>;
pub(crate) type Space2R2c = Space2<BaseR2c<f64>, BaseR2r<f64>>;

/// Container for Navier Stokes simulations
pub struct Navier2D<T, S> {
    /// Field for derivatives and transforms
    pub field: Field2<T, S>,
    /// Temperature
    pub temp: Field2<T, S>,
    /// Horizontal Velocity
    pub velx: Field2<T, S>,
    /// Vertical Velocity
    pub vely: Field2<T, S>,
    /// Pressure field
    pub pres: Field2<T, S>,
    /// Pseudo pressure
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
    /// Scale of phsical dimension \[scale_x, scale_y\]
    pub scale: [f64; 2],
    /// Helmholtz solver for implicit diffusion term
    pub(crate) solver_hholtz: [HholtzAdi<f64, 2>; 3],
    /// Poisson solver for pressure
    pub(crate) solver_pres: Poisson<f64, 2>,
    /// diagnostics like Nu, ...
    pub diagnostics: HashMap<String, Vec<f64>>,
    /// Time intervall for write fields
    /// If none, same intervall as diagnostics
    pub write_intervall: Option<f64>,
    /// Add a solid obstacle
    pub solid: Option<[Array2<f64>; 2]>,
    /// If set, collect statistics
    pub statistics: Option<Statistics<T, S>>,
}

impl<T, S> Navier2D<T, S>
where
    T: Zero,
{
    pub(crate) fn zero_rhs(&mut self) {
        for r in self.rhs.iter_mut() {
            *r = T::zero();
        }
    }
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
impl<T, S> Navier2D<T, S>
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
    pub fn random_disturbance(&mut self, amp: f64) {
        random_field(&mut self.temp, amp);
        random_field(&mut self.velx, amp);
        random_field(&mut self.vely, amp);
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
    ) -> Navier2D<f64, Space2R2r> {
        // geometry scales
        let scale = [aspect, 1.];
        // diffusivities
        let nu = get_nu(ra, pr, scale[1] * 2.0);
        let ka = get_ka(ra, pr, scale[1] * 2.0);
        let mut params = HashMap::new();
        params.insert("ra", ra);
        params.insert("pr", pr);
        params.insert("nu", nu);
        params.insert("ka", ka);
        // velocities
        let mut velx = Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny)));
        let mut vely = Field2::new(&Space2::new(&cheb_dirichlet(nx), &cheb_dirichlet(ny)));
        // temperature
        let (mut temp, tempbc, presbc) = match bc {
            "rbc" => {
                let temp = Field2::new(&Space2::new(&cheb_neumann(nx), &cheb_dirichlet(ny)));
                let tempbc = bc_rbc(nx, ny);
                let presbc = pres_bc_rbc(nx, ny);
                (temp, Some(tempbc), Some(presbc))
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
        // Scale fields
        velx.scale(scale);
        vely.scale(scale);
        temp.scale(scale);
        pres.scale(scale);
        // define solver
        let solver_velx = HholtzAdi::new(
            &velx,
            [dt * nu / scale[0].powi(2), dt * nu / scale[1].powi(2)],
        );
        let solver_vely = HholtzAdi::new(
            &vely,
            [dt * nu / scale[0].powi(2), dt * nu / scale[1].powi(2)],
        );
        let solver_temp = HholtzAdi::new(
            &temp,
            [dt * ka / scale[0].powi(2), dt * ka / scale[1].powi(2)],
        );
        let solver_pres = Poisson::new(&pseu, [1. / scale[0].powi(2), 1. / scale[1].powi(2)]);
        let solver_hholtz = [solver_velx, solver_vely, solver_temp];
        let rhs = Array2::zeros(temp.v.raw_dim());

        // Diagnostics
        let diagnostics = HashMap::new();

        // Initialize
        let mut navier = Navier2D::<f64, Space2R2r> {
            field,
            temp,
            velx,
            vely,
            pres,
            pseu,
            rhs,
            tempbc,
            presbc,
            params,
            time: 0.0,
            dt,
            scale,
            solver_hholtz,
            solver_pres,
            diagnostics,
            write_intervall: None,
            solid: None,
            statistics: None,
        };
        // Initial condition
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
    ) -> Navier2D<Complex<f64>, Space2R2c> {
        // geometry scales
        let scale = [aspect, 1.];
        // diffusivities
        let nu = get_nu(ra, pr, scale[1] * 2.0);
        let ka = get_ka(ra, pr, scale[1] * 2.0);
        let mut params = HashMap::new();
        params.insert("ra", ra);
        params.insert("pr", pr);
        params.insert("nu", nu);
        params.insert("ka", ka);
        // velocities
        let mut velx = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny)));
        let mut vely = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny)));
        // temperature
        let (mut temp, tempbc, presbc) = match bc {
            "rbc" => {
                let temp = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny)));
                let tempbc = bc_rbc_periodic(nx, ny);
                let presbc = pres_bc_rbc_periodic(nx, ny);
                (temp, Some(tempbc), Some(presbc))
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
        let field = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
        // Scale fields
        velx.scale(scale);
        vely.scale(scale);
        temp.scale(scale);
        pres.scale(scale);
        // define solver
        let solver_velx = HholtzAdi::new(
            &velx,
            [dt * nu / scale[0].powi(2), dt * nu / scale[1].powi(2)],
        );
        let solver_vely = HholtzAdi::new(
            &vely,
            [dt * nu / scale[0].powi(2), dt * nu / scale[1].powi(2)],
        );
        let solver_temp = HholtzAdi::new(
            &temp,
            [dt * ka / scale[0].powi(2), dt * ka / scale[1].powi(2)],
        );
        let solver_pres = Poisson::new(&pseu, [1. / scale[0].powi(2), 1. / scale[1].powi(2)]);
        let solver_hholtz = [solver_velx, solver_vely, solver_temp];
        let rhs = Array2::zeros(field.vhat.raw_dim());

        // Diagnostics
        let diagnostics = HashMap::new();

        // Initialize
        let mut navier = Navier2D::<Complex<f64>, Space2R2c> {
            field,
            temp,
            velx,
            vely,
            pres,
            pseu,
            rhs,
            tempbc,
            presbc,
            params,
            time: 0.0,
            dt,
            scale,
            solver_hholtz,
            solver_pres,
            diagnostics,
            write_intervall: None,
            solid: None,
            statistics: None,
        };
        // Initial condition
        navier.random_disturbance(0.1);
        // Return
        navier
    }
}

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
                self.velx.backward();
                self.vely.backward();
                let velx = self.velx.v.to_owned();
                let vely = self.vely.v.to_owned();

                // Solve Velocity
                self.solve_velx(&velx, &vely);
                self.solve_vely(&velx, &vely, &that);

                // Projection
                let div = self.div();
                self.solve_pres(&div);
                self.correct_velocity(1.0);
                self.update_pres(&div);

                // Solve Temperature
                self.solve_temp(&velx, &vely);

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
                let flowname = format!("data/flow{:0>8.2}.h5", self.time);
                let io_name = "data/info.txt";
                self.callback_from_filename(&flowname, io_name, false);
            }

            fn exit(&mut self) -> bool {
                // Break if divergence is nan
                let div = self.div();
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
