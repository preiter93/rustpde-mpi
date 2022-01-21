//! Linearized Navier--Stokes equations
use super::meanfield::MeanFields;
// use crate::bases::{BaseR2c, BaseR2r};
use crate::bases::{cheb_dirichlet, cheb_dirichlet_neumann, cheb_neumann, chebyshev, fourier_r2c};
use crate::bases::{BaseR2c, BaseR2r, Space2};
use crate::field::Field2;
use crate::navier_stokes::functions::{get_ka, get_nu};
use crate::solver::{HholtzAdi, Poisson};
use crate::Integrate;
use funspace::BaseSpace;
use ndarray::Array2;
use num_complex::Complex;
use std::collections::HashMap;
// pub(crate) type Space2R2r = Space2<BaseR2r<f64>, BaseR2r<f64>>;
// pub(crate) type Space2R2c = Space2<BaseR2c<f64>, BaseR2r<f64>>;

/// Output every x timeunits
pub const OUTPUT_INTERVALL: f64 = 1.;

// pub(crate) type Space2R2r = Space2<BaseR2r<f64>, BaseR2r<f64>>;
pub(crate) type Space2R2c = Space2<BaseR2c<f64>, BaseR2r<f64>>;

/// Linearized Navier Stokes solver
pub struct Navier2DLnse<T, S> {
    /// Field for derivatives and transforms
    pub field: Field2<T, S>,
    /// Horizontal Velocity
    pub velx: Field2<T, S>,
    /// Vertical Velocity
    pub vely: Field2<T, S>,
    /// Temperature
    pub temp: Field2<T, S>,
    /// Pressure
    pub pres: Field2<T, S>,
    /// Pseudo Pressure
    pub pseu: Field2<T, S>,
    /// Helmholtz solvers for implicit diffusion
    pub(crate) solver_hholtz: [HholtzAdi<f64, 2>; 3],
    /// Poisson solver for pressure
    pub(crate) solver_pres: Poisson<f64, 2>,
    /// Buffer
    pub(crate) rhs: Array2<T>,
    /// Scale of phsical dimension \[scale_x, scale_y\]
    pub scale: [f64; 2],
    /// Time step size
    pub dt: f64,
    /// Current time
    pub time: f64,
    /// Parameter
    pub params: HashMap<&'static str, f64>,
    /// Quantities
    pub quantities: HashMap<&'static str, f64>,
    /// Mean Field
    pub mean: MeanFields<T, S>,
}

impl<T, S> Navier2DLnse<T, S>
where
    T: num_traits::Zero,
{
    pub(crate) fn zero_rhs(&mut self) {
        for r in self.rhs.iter_mut() {
            *r = T::zero();
        }
    }
}

impl<T, S> Navier2DLnse<T, S>
where
    T: num_traits::Zero,
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
{
    /// Set random initial fields: -amp < x < amp
    pub fn init_random(&mut self, amp: f64) {
        use crate::navier_stokes::functions::random_field;
        random_field(&mut self.temp, amp);
        random_field(&mut self.velx, amp);
        random_field(&mut self.vely, amp);
    }
}

impl Navier2DLnse<Complex<f64>, Space2R2c> {
    /// Linearized Navier Stokes solver with periodic sidewalls
    /// # Panics
    /// 'bc' type not recognized
    pub fn new_periodic(
        nx: usize,
        ny: usize,
        ra: f64,
        pr: f64,
        dt: f64,
        aspect: f64,
        bc: &str,
    ) -> Navier2DLnse<Complex<f64>, Space2R2c> {
        // geometry scales
        let scale = [aspect, 1.];
        // diffusivities
        let nu: f64 = get_nu(ra, pr, scale[1] * 2.0);
        let ka: f64 = get_ka(ra, pr, scale[1] * 2.0);
        // Fill parameters
        let mut params = HashMap::new();
        params.insert("ra", ra);
        params.insert("pr", pr);
        params.insert("nu", nu);
        params.insert("ka", ka);
        let quantities = HashMap::new();
        let time = 0.;

        // Fields
        let field = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
        let mut velx = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny)));
        let mut vely = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny)));
        let mut pres = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
        let pseu = Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_neumann(ny)));
        let mut temp = match bc {
            "rbc" => Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet(ny))),
            "hc" => Field2::new(&Space2::new(&fourier_r2c(nx), &cheb_dirichlet_neumann(ny))),
            _ => panic!("Boundary condition type {:?} not recognized!", bc),
        };

        let rhs: Array2<Complex<f64>> = Array2::zeros(field.vhat.raw_dim());
        // Scale fields
        velx.scale(scale);
        vely.scale(scale);
        temp.scale(scale);
        pres.scale(scale);

        // Mean Field
        let mut mean = MeanFields::read_from_periodic(nx, ny, "mean.h5", Some(bc));
        mean.write_unwrap("mean_field.h5");

        // Solver
        let solver_velx = HholtzAdi::new(
            &velx,
            [dt * nu / scale[0].powf(2.), dt * nu / scale[1].powf(2.)],
        );
        let solver_vely = HholtzAdi::new(
            &vely,
            [dt * nu / scale[0].powf(2.), dt * nu / scale[1].powf(2.)],
        );
        let solver_temp = HholtzAdi::new(
            &temp,
            [dt * ka / scale[0].powf(2.), dt * ka / scale[1].powf(2.)],
        );
        let solver_pres = Poisson::new(&pseu, [1. / scale[0].powf(2.), 1. / scale[1].powf(2.)]);
        let solver_hholtz = [solver_velx, solver_vely, solver_temp];

        // Return
        Navier2DLnse::<Complex<f64>, Space2R2c> {
            field,
            velx,
            vely,
            temp,
            pres,
            pseu,
            solver_hholtz,
            solver_pres,
            rhs,
            scale,
            dt,
            time,
            params,
            quantities,
            mean,
        }
    }
}

macro_rules! impl_integrate_for_navier {
    ($s: ty) => {
        impl<S> Integrate for Navier2DLnse<$s, S>
        where
            S: BaseSpace<f64, 2, Physical = f64, Spectral = $s>,
        {
            /// Update 1 timestep
            fn update(&mut self) {
                // Buoyancy
                let that = self.temp.to_ortho();

                // Convection Veclocity
                self.velx.backward();
                self.vely.backward();
                let ux = self.velx.v.to_owned();
                let uy = self.vely.v.to_owned();

                // Solve Velocity
                self.solve_velx(&ux, &uy);
                self.solve_vely(&ux, &uy, &that);

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
                use crate::navier_stokes::functions::norm_l2_f64;
                // Break if divergence is nan
                let div = self.div();
                if norm_l2_f64(&self.field.space.backward(&div)).is_nan() {
                    return true;
                }
                false
            }
        }
    };
}

impl_integrate_for_navier!(f64);
impl_integrate_for_navier!(Complex<f64>);
