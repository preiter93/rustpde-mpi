//! # Solver for Boussinesq equations (mpi supported)
//! Default: Rayleigh Benard Convection
use super::boundary_conditions::{bc_hc, bc_hc_periodic, bc_rbc, bc_rbc_periodic};
use super::boundary_conditions::{pres_bc_rbc, pres_bc_rbc_periodic};
use super::functions::{apply_cos_sin, apply_random_disturbance, apply_sin_cos};
use super::functions::{get_ka, get_nu};
use super::functions::{norm_l2_c64, norm_l2_f64};
use super::statistics::Statistics;
use crate::bases::fourier_r2c;
use crate::bases::{cheb_dirichlet, cheb_dirichlet_neumann, cheb_neumann, chebyshev};
use crate::bases::{BaseR2c, BaseR2r};
use crate::field::BaseSpace;
use crate::field_mpi::Field2Mpi;
use crate::mpi::{BaseSpaceMpi, Space2Mpi};
use crate::mpi::{Communicator, Integrate, Universe};
use crate::solver_mpi::hholtz_adi::HholtzAdiMpi;
use crate::solver_mpi::poisson::PoissonMpi;
use crate::types::Scalar;
use ndarray::Array2;
use num_complex::Complex;
use num_traits::Zero;
use std::collections::HashMap;
use std::ops::{Div, Mul};

/// Solve 2-dimensional Navier-Stokes equations
/// coupled with temperature equations
pub struct Navier2DMpi<'a, T, S> {
    /// Mpi universe
    pub(crate) universe: &'a Universe,
    /// Field for derivatives and transforms
    pub field: Field2Mpi<T, S>,
    /// Temperature
    pub temp: Field2Mpi<T, S>,
    /// Horizontal Velocity
    pub ux: Field2Mpi<T, S>,
    /// Vertical Velocity
    pub uy: Field2Mpi<T, S>,
    /// Pressure field
    pub pres: Field2Mpi<T, S>,
    /// Pseudo pressure
    pub pseu: Field2Mpi<T, S>,
    /// Helmholtz solvers for implicit diffusion
    pub(crate) solver_hholtz: [HholtzAdiMpi<f64, S, 2>; 3],
    /// Poisson solver for pressure
    pub(crate) solver_pres: PoissonMpi<f64, S, 2>,
    /// Buffer
    pub(crate) rhs: Array2<T>,
    /// Field for temperature boundary condition
    pub tempbc: Option<Field2Mpi<T, S>>,
    /// Field for pressure boundary condition
    pub presbc: Option<Field2Mpi<T, S>>,
    /// Parameter (e.g. diffusivities, ra, pr, ...)
    pub params: HashMap<&'static str, f64>,
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
    /// If set, collect statistics
    pub statistics: Option<Statistics<T, S>>,
}

pub(crate) type Space2R2r<'a> = Space2Mpi<'a, BaseR2r<f64>, BaseR2r<f64>>;
pub(crate) type Space2R2c<'a> = Space2Mpi<'a, BaseR2c<f64>, BaseR2r<f64>>;

impl<T, S> Navier2DMpi<'_, T, S> {
    /// Return current rank
    #[allow(clippy::cast_sign_loss)]
    pub fn nrank(&self) -> usize {
        let world = self.universe.world();
        world.rank() as usize
    }

    /// Return total number of processors
    #[allow(clippy::cast_sign_loss)]
    pub fn nprocs(&self) -> usize {
        let world = self.universe.world();
        world.size() as usize
    }

    /// Reset time
    pub fn reset_time(&mut self) {
        self.time = 0.;
    }
}

impl<T, S> Navier2DMpi<'_, T, S>
where
    T: Zero,
{
    pub(crate) fn zero_rhs(&mut self) {
        for r in self.rhs.iter_mut() {
            *r = T::zero();
        }
    }
}

impl<T, S> Navier2DMpi<'_, T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T> + BaseSpaceMpi<f64, 2>,
    T: Scalar + Mul<f64, Output = T> + Div<f64, Output = T>,
{
    /// Returns Nusselt number (heat flux at the plates)
    /// $$
    /// Nu = \langle - dTdz \rangle\\_x (0/H))
    /// $$
    pub fn eval_nu_mpi(&mut self) -> f64 {
        use super::functions::eval_nu_mpi;
        eval_nu_mpi(&mut self.temp, &mut self.field, &self.tempbc, &self.scale)
    }

    /// Returns volumetric Nusselt number
    /// $$
    /// Nuvol = \langle uy*T/kappa - dTdz \rangle\\_V
    /// $$
    ///
    /// # Panics
    /// If *ka* is not in params
    pub fn eval_nuvol_mpi(&mut self) -> f64 {
        use super::functions::eval_nuvol_mpi;
        let ka = self.params.get("ka").unwrap();
        eval_nuvol_mpi(
            &mut self.temp,
            &mut self.uy,
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
    pub fn eval_re_mpi(&mut self) -> f64 {
        use super::functions::eval_re_mpi;
        let nu = self.params.get("nu").unwrap();
        eval_re_mpi(
            &mut self.ux,
            &mut self.uy,
            &mut self.field,
            *nu,
            &self.scale,
        )
    }
}

impl<T, S> Navier2DMpi<'_, T, S>
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
    ///
    /// # Panics
    /// If *ka* is not in params
    pub fn eval_nuvol(&mut self) -> f64 {
        use super::functions::eval_nuvol;
        let ka = self.params.get("ka").unwrap();
        eval_nuvol(
            &mut self.temp,
            &mut self.uy,
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
            &mut self.ux,
            &mut self.uy,
            &mut self.field,
            *nu,
            &self.scale,
        )
    }
}

impl<T, S> Navier2DMpi<'_, T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = T>,
    T: Zero,
{
    /// Gather all arrays from mpi distribution on root
    pub fn gather(&mut self) {
        for field in &mut [&mut self.ux, &mut self.uy, &mut self.temp, &mut self.pres] {
            field.backward_mpi();
            field.gather_physical();
            field.gather_spectral();
        }
    }

    /// Gather all arrays from mpi distribution to all participating processors
    pub fn all_gather(&mut self) {
        for field in &mut [&mut self.ux, &mut self.uy, &mut self.temp, &mut self.pres] {
            field.backward_mpi();
            field.all_gather_physical();
            field.all_gather_spectral();
        }
    }

    /// Scatter all arrays from root to all processes
    pub fn scatter(&mut self) {
        for field in &mut [&mut self.ux, &mut self.uy, &mut self.temp, &mut self.pres] {
            field.scatter_physical();
            field.scatter_spectral();
        }
    }

    /// Initialize velocity with fourier modes
    ///
    /// ux = amp \* sin(mx)cos(nx)
    /// uy = -amp \* cos(mx)sin(nx)
    pub fn set_velocity(&mut self, amp: f64, m: f64, n: f64) {
        apply_sin_cos(&mut self.ux, amp, m, n);
        apply_cos_sin(&mut self.uy, -amp, m, n);
        self.scatter();
    }
    /// Initialize temperature with fourier modes
    ///
    /// temp = -amp \* cos(mx)sin(ny)
    pub fn set_temperature(&mut self, amp: f64, m: f64, n: f64) {
        apply_cos_sin(&mut self.temp, -amp, m, n);
        self.scatter();
    }

    /// Initialize all fields with random disturbances
    pub fn random_disturbance(&mut self, amp: f64) {
        if self.nrank() == 0 {
            apply_random_disturbance(&mut self.temp, amp);
            apply_random_disturbance(&mut self.ux, amp);
            apply_random_disturbance(&mut self.uy, amp);
            // Remove bc base from temp
            if let Some(x) = &self.tempbc {
                self.temp.v = &self.temp.v - &x.v;
                self.temp.forward();
            }
        }
        self.scatter();
        self.all_gather()
    }
}

impl<'a> Navier2DMpi<'_, f64, Space2R2r<'a>>
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
    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    pub fn new_confined(
        universe: &'a Universe,
        nx: usize,
        ny: usize,
        ra: f64,
        pr: f64,
        dt: f64,
        aspect: f64,
        bc: &'static str,
    ) -> Navier2DMpi<'a, f64, Space2R2r<'a>> {
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
        let mut ux = Field2Mpi::new(&Space2Mpi::new(
            &cheb_dirichlet(nx),
            &cheb_dirichlet(ny),
            universe,
        ));
        let mut uy = Field2Mpi::new(&Space2Mpi::new(
            &cheb_dirichlet(nx),
            &cheb_dirichlet(ny),
            universe,
        ));
        // temperature
        let (mut temp, tempbc, presbc) = match bc {
            "rbc" => {
                let temp = Field2Mpi::new(&Space2Mpi::new(
                    &cheb_neumann(nx),
                    &cheb_dirichlet(ny),
                    universe,
                ));
                let tempbc = bc_rbc(nx, ny, universe);
                let presbc = pres_bc_rbc(nx, ny, universe);
                (temp, Some(tempbc), Some(presbc))
            }
            "hc" => {
                let temp = Field2Mpi::new(&Space2Mpi::new(
                    &cheb_neumann(nx),
                    &cheb_dirichlet_neumann(ny),
                    universe,
                ));
                let tempbc = bc_hc(nx, ny, universe);
                (temp, Some(tempbc), None)
            }
            _ => panic!("Boundary condition type {:?} not recognized!", bc),
        };

        // pressure
        let mut pres = Field2Mpi::new(&Space2Mpi::new(&chebyshev(nx), &chebyshev(ny), universe));
        let pseu = Field2Mpi::new(&Space2Mpi::new(
            &cheb_neumann(nx),
            &cheb_neumann(ny),
            universe,
        ));

        // fields for derivatives
        let field = Field2Mpi::new(&Space2Mpi::new(&chebyshev(nx), &chebyshev(ny), universe));

        // Scale fields
        ux.scale(scale);
        uy.scale(scale);
        temp.scale(scale);
        pres.scale(scale);

        // define solver
        let solver_ux = HholtzAdiMpi::new(
            &ux,
            [dt * nu / scale[0].powi(2), dt * nu / scale[1].powi(2)],
        );
        let solver_uy = HholtzAdiMpi::new(
            &uy,
            [dt * nu / scale[0].powi(2), dt * nu / scale[1].powi(2)],
        );
        let solver_temp = HholtzAdiMpi::new(
            &temp,
            [dt * ka / scale[0].powi(2), dt * ka / scale[1].powi(2)],
        );
        let solver_pres = PoissonMpi::new(&pseu, [1. / scale[0].powi(2), 1. / scale[1].powi(2)]);

        let solver_hholtz = [solver_ux, solver_uy, solver_temp];
        let rhs = Array2::zeros(field.vhat_x_pen.raw_dim());

        // Diagnostics
        let diagnostics = HashMap::new();

        // Initialize
        let mut navier = Navier2DMpi::<f64, Space2R2r> {
            universe,
            field,
            temp,
            ux,
            uy,
            pres,
            pseu,
            solver_hholtz,
            solver_pres,
            rhs,
            tempbc,
            presbc,
            params,
            time: 0.0,
            dt,
            scale,
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

impl<'a> Navier2DMpi<'_, Complex<f64>, Space2R2c<'a>>
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
    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    pub fn new_periodic(
        universe: &'a Universe,
        nx: usize,
        ny: usize,
        ra: f64,
        pr: f64,
        dt: f64,
        aspect: f64,
        bc: &'static str,
    ) -> Navier2DMpi<'a, Complex<f64>, Space2R2c<'a>> {
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
        let mut ux = Field2Mpi::new(&Space2Mpi::new(
            &fourier_r2c(nx),
            &cheb_dirichlet(ny),
            universe,
        ));
        let mut uy = Field2Mpi::new(&Space2Mpi::new(
            &fourier_r2c(nx),
            &cheb_dirichlet(ny),
            universe,
        ));

        // temperature
        let (mut temp, tempbc, presbc) = match bc {
            "rbc" => {
                let temp = Field2Mpi::new(&Space2Mpi::new(
                    &fourier_r2c(nx),
                    &cheb_dirichlet(ny),
                    universe,
                ));
                let tempbc = bc_rbc_periodic(nx, ny, universe);
                let presbc = pres_bc_rbc_periodic(nx, ny, universe);
                (temp, Some(tempbc), Some(presbc))
            }
            "hc" => {
                let temp = Field2Mpi::new(&Space2Mpi::new(
                    &fourier_r2c(nx),
                    &cheb_dirichlet_neumann(ny),
                    universe,
                ));
                let tempbc = bc_hc_periodic(nx, ny, universe);
                (temp, Some(tempbc), None)
            }
            _ => panic!("Boundary condition type {:?} not recognized!", bc),
        };

        // pressure
        let mut pres = Field2Mpi::new(&Space2Mpi::new(&fourier_r2c(nx), &chebyshev(ny), universe));
        let pseu = Field2Mpi::new(&Space2Mpi::new(
            &fourier_r2c(nx),
            &cheb_neumann(ny),
            universe,
        ));

        // fields for derivatives
        let field = Field2Mpi::new(&Space2Mpi::new(&fourier_r2c(nx), &chebyshev(ny), universe));

        // Scale coordinates
        ux.scale(scale);
        uy.scale(scale);
        temp.scale(scale);
        pres.scale(scale);

        // define solver
        let solver_ux = HholtzAdiMpi::new(
            &ux,
            [dt * nu / scale[0].powi(2), dt * nu / scale[1].powi(2)],
        );
        let solver_uy = HholtzAdiMpi::new(
            &uy,
            [dt * nu / scale[0].powi(2), dt * nu / scale[1].powi(2)],
        );
        let solver_temp = HholtzAdiMpi::new(
            &temp,
            [dt * ka / scale[0].powi(2), dt * ka / scale[1].powi(2)],
        );
        let solver_hholtz = [solver_ux, solver_uy, solver_temp];
        let solver_pres = PoissonMpi::new(&pseu, [1. / scale[0].powi(2), 1. / scale[1].powi(2)]);
        let rhs = Array2::zeros(field.vhat_x_pen.raw_dim());

        // Diagnostics
        let diagnostics = HashMap::new();

        // Initialize
        let mut navier = Navier2DMpi::<Complex<f64>, Space2R2c> {
            universe,
            field,
            temp,
            ux,
            uy,
            pres,
            pseu,
            solver_hholtz,
            solver_pres,
            rhs,
            tempbc,
            presbc,
            params,
            time: 0.0,
            dt,
            scale,
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
        impl<S> Integrate for Navier2DMpi<'_, $s, S>
        where
            S: BaseSpace<f64, 2, Physical = f64, Spectral = $s>
                + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = $s>,
        {
            /// Update 1 timestep
            fn update(&mut self) {
                // Buoyancy
                let mut that = self.temp.to_ortho_mpi();
                if let Some(field) = &self.tempbc {
                    that = &that + &field.to_ortho_mpi();
                }
                // Convection Veclocity
                self.ux.backward_mpi();
                self.uy.backward_mpi();
                let ux = self.ux.v_y_pen.to_owned();
                let uy = self.uy.v_y_pen.to_owned();
                // Solve Velocity
                self.solve_ux(&ux, &uy);
                self.solve_uy(&ux, &uy, &that);
                // Projection
                let div = self.div();
                self.solve_pres(&div);
                self.correct_velocity(1.0);
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

            fn nrank(&self) -> usize {
                self.nrank()
            }
        }
    };
}
impl_integrate_for_navier!(f64, norm_l2_f64);
impl_integrate_for_navier!(Complex<f64>, norm_l2_c64);
