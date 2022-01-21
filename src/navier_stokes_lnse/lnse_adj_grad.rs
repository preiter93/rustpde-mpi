//! Calculate adjoint based sensitivity (gradient of
//! final energy with respect to initial field)
use super::functions::energy;
use super::Navier2DLnse;
use crate::field::BaseSpace;
use crate::field::Field2;
use crate::io::traits::ReadWrite;
use crate::navier_stokes_lnse::lnse_eq::DivNorm;
use crate::solver::{hholtz_adi::HholtzAdi, poisson::Poisson, Solve};
use crate::types::Scalar;
use ndarray::{Ix2, ScalarOperand};
use std::ops::Mul;

impl<T, S> Navier2DLnse<T, S>
where
    T: Scalar + Mul<f64, Output = T> + From<f64> + ScalarOperand,
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    Navier2DLnse<T, S>: DivNorm,
    Poisson<f64, 2>: Solve<T, Ix2>,
    HholtzAdi<f64, 2>: Solve<T, Ix2>,
    Field2<T, S>: ReadWrite<T>,
{
    /// Return time
    pub fn get_time(&self) -> f64 {
        self.time
    }

    /// Return timestep
    pub fn get_dt(&self) -> f64 {
        self.dt
    }

    /// Reset time
    pub fn reset_time(&mut self) {
        self.time = 0.;
    }

    /// Update forward loop
    pub fn update_direct(&mut self) {
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

    /// Update adjoint loop
    pub fn update_adjoint(&mut self) {
        // Buoyancy
        let uyhat = self.vely.to_ortho();

        // Convection Veclocity
        self.velx.backward();
        self.vely.backward();
        self.temp.backward();
        let velx = self.velx.v.to_owned();
        let vely = self.vely.v.to_owned();
        let temp = self.temp.v.to_owned();

        // Solve Velocity
        self.solve_velx_adj(&velx, &vely, &temp);
        self.solve_vely_adj(&velx, &vely, &temp);

        // Projection
        let div = self.div();
        self.solve_pres(&div);
        self.correct_velocity(1.0);
        self.update_pres(&div);

        // Solve Temperature
        self.solve_temp_adj(&velx, &vely, &temp, &uyhat);

        // update time
        self.time += self.dt;
    }

    /// Calculate gradient from forward and backward loop
    ///
    /// # Return
    /// (funval, gradient)
    #[allow(clippy::type_complexity)]
    pub fn grad_adjoint(
        &mut self,
        max_time: f64,
        save_intervall: Option<f64>,
    ) -> (f64, (Field2<T, S>, Field2<T, S>, Field2<T, S>)) {
        let mut timestep: usize = 0;
        let max_timestep = 10_000_000;

        // Weights of norm (vel, temp)
        let (b1, b2) = (0.5, 0.5);
        loop {
            self.update_direct();
            timestep += 1;

            // Save
            if let Some(dt_save) = &save_intervall {
                if (self.get_time() % dt_save) < self.get_dt() / 2.
                    || (self.get_time() % dt_save) > dt_save - self.get_dt() / 2.
                {
                    let fname = format!("data/flow{:0>8.2}.h5", self.time);
                    self.callback_from_filename(&fname, "data/info.txt", true);
                }
            }

            // Break
            if self.exit(self.time, max_time, timestep, max_timestep) {
                break;
            }
        }

        // Energy
        let en = energy(&mut self.velx, &mut self.vely, &mut self.temp, b1, b2);

        // Function value to be returned
        let fun_val = en;

        // Initial conditions of adjoint fields
        let b1_c64: T = b1.into();
        let b2_c64: T = b2.into();
        self.velx.vhat *= b1_c64;
        self.vely.vhat *= b1_c64;
        self.temp.vhat *= b2_c64;

        // Adjoint loop
        self.reset_time();
        loop {
            self.update_adjoint();
            timestep += 1;

            // Save
            if let Some(dt_save) = &save_intervall {
                if (self.time + self.dt / 2.) % dt_save < self.dt {
                    let fname = format!("data/adjoint{:0>8.2}.h5", self.time);
                    self.callback_from_filename(&fname, "data/info_adjoint.txt", false);
                }
            }

            // Break
            if self.exit(self.time, max_time, timestep, max_timestep) {
                break;
            }
        }

        // Gradient calculation
        let mut grad_u = Field2::new(&self.velx.space);
        let mut grad_v = Field2::new(&self.vely.space);
        let mut grad_t = Field2::new(&self.temp.space);
        grad_u.v.assign(&(&self.velx.v));
        grad_v.v.assign(&(&self.vely.v));
        grad_t.v.assign(&(&self.temp.v));
        grad_u.forward();
        grad_v.forward();
        grad_t.forward();
        let filename = "data/grad_adjoint.h5";
        grad_u.write_unwrap(&filename, "ux");
        grad_v.write_unwrap(&filename, "uy");
        grad_t.write_unwrap(&filename, "temp");
        (fun_val, (grad_u, grad_v, grad_t))
    }

    fn exit(&mut self, time: f64, max_time: f64, timestep: usize, max_timestep: usize) -> bool {
        let eps_dt = self.get_dt() * 1e-4;
        // Break
        if time + eps_dt >= max_time {
            // println!("time limit reached: {:?}", time);
            // break;
            return true;
        }
        if timestep >= max_timestep {
            // println!("timestep limit reached: {:?}", timestep);
            // break;
            return true;
        }
        // Break if divergence is nan
        if self.div_norm().is_nan() {
            println!("Divergence is nan");
            return true;
        }
        false
    }
}

use ndarray::Array2;

/// Steepest descent optimization without energy increase of the
/// target flow
///
/// # Panics
/// Input `alpha` > 2.
#[allow(
    clippy::shadow_unrelated,
    clippy::cast_precision_loss,
    clippy::too_many_arguments
)]
pub fn opt_routine(
    velx_0: &Array2<f64>,
    vely_0: &Array2<f64>,
    temp_0: &Array2<f64>,
    grad_velx: &mut Array2<f64>,
    grad_vely: &mut Array2<f64>,
    grad_temp: &mut Array2<f64>,
    velx_new: &mut Array2<f64>,
    vely_new: &mut Array2<f64>,
    temp_new: &mut Array2<f64>,
    beta1: f64,
    beta2: f64,
    alpha: f64,
) {
    use super::functions::l2_norm;
    if alpha > 2. * std::f64::consts::PI {
        panic!("alpha must be less than 2 pi")
    }
    let n = velx_0.len() as f64;
    let e0 = l2_norm(velx_0, velx_0, vely_0, vely_0, temp_0, temp_0, beta1, beta2) / n;
    let eg = l2_norm(
        grad_velx, velx_0, grad_vely, vely_0, grad_temp, temp_0, beta1, beta2,
    ) / n;

    // Project gradient perpendicular to x0
    let ee = eg / e0;
    *grad_velx -= &(ee * velx_0);
    *grad_vely -= &(ee * vely_0);
    *grad_temp -= &(ee * temp_0);

    // Linear combination of old field and gradient
    let eg = l2_norm(
        grad_velx, grad_velx, grad_vely, grad_vely, grad_temp, grad_temp, beta1, beta2,
    ) / n;
    let ee2 = (e0 / eg).sqrt();
    velx_new
        .assign(&(velx_0.mapv(|x| x * alpha.cos()) + grad_velx.mapv(|x| x * ee2 * alpha.sin())));
    vely_new
        .assign(&(vely_0.mapv(|x| x * alpha.cos()) + grad_vely.mapv(|x| x * ee2 * alpha.sin())));
    temp_new
        .assign(&(temp_0.mapv(|x| x * alpha.cos()) + grad_temp.mapv(|x| x * ee2 * alpha.sin())));
}
