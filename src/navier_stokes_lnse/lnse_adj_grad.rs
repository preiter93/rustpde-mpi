//! Calculate adjoint based sensitivity (gradient of
//! final energy with respect to initial field)
use super::functions::l2_norm;
use super::meanfield::MeanFields;
use super::Navier2DLnse;
use crate::field::BaseSpace;
use crate::field::Field2;
use crate::io::traits::ReadWrite;
use crate::navier_stokes_lnse::lnse_eq::DivNorm;
use crate::solver::{hholtz_adi::HholtzAdi, poisson::Poisson, Solve};
use crate::types::Scalar;
use ndarray::{Ix2, ScalarOperand};
use std::ops::Mul;

/// Solve maximization problem instead of minimization
pub const MAXIMIZE: bool = false;

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
        beta1: f64,
        beta2: f64,
        target: Option<&MeanFields<T, S>>,
    ) -> (f64, (Field2<T, S>, Field2<T, S>, Field2<T, S>)) {
        let mut timestep: usize = 0;
        let max_timestep = 10_000_000;

        // Weights of norm (vel, temp)
        loop {
            // +1 timestep
            self.update_direct();
            timestep += 1;
            // Save
            if let Some(dt_save) = &save_intervall {
                if (self.get_time() % dt_save) < self.get_dt() / 2.
                    || (self.get_time() % dt_save) > dt_save - self.get_dt() / 2.
                {
                    let fname = format!("data/flow{:0>8.2}.h5", self.time);
                    self.callback_from_filename(&fname, "data/info.txt", true, None);
                }
            }
            // Break
            if self.exit(self.time, max_time, timestep, max_timestep) {
                break;
            }
        }

        // Energy
        self.velx.backward();
        self.vely.backward();
        self.temp.backward();
        let en = target.as_ref().map_or_else(
            || {
                let (velx, vely, temp) = (&self.velx.v, &self.vely.v, &self.temp.v);
                l2_norm(&velx, &velx, &vely, &vely, &temp, &temp, beta1, beta2)
            },
            |t| {
                let (velx, vely, temp) = (
                    &(&self.velx.v - &t.velx.v),
                    &(&self.vely.v - &t.vely.v),
                    &(&self.temp.v - &t.temp.v),
                );
                l2_norm(&velx, &velx, &vely, &vely, &temp, &temp, beta1, beta2)
            },
        );

        // Function value to be returned
        let fun_val = en;

        // Initial conditions of adjoint fields
        let b1_c64: T = (1. * beta1).into();
        let b2_c64: T = (1. * beta2).into();
        if let Some(t) = &target {
            self.velx.vhat -= &self.velx.space.from_ortho(&t.velx.vhat);
            self.vely.vhat -= &self.vely.space.from_ortho(&t.vely.vhat);
            self.temp.vhat -= &self.temp.space.from_ortho(&t.temp.vhat);
        }
        self.velx.vhat *= b1_c64;
        self.vely.vhat *= b1_c64;
        self.temp.vhat *= b2_c64;

        // Adjoint loop
        self.reset_time();
        loop {
            // +1 timestep
            self.update_adjoint();
            timestep += 1;
            // Save
            if let Some(dt_save) = &save_intervall {
                if (self.time + self.dt / 2.) % dt_save < self.dt {
                    let fname = format!("data/adjoint{:0>8.2}.h5", self.time);
                    self.callback_from_filename(&fname, "data/info_adjoint.txt", false, None);
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
        let fac = if MAXIMIZE { 1. } else { -1. };
        grad_u.v.assign(&(fac * &self.velx.v));
        grad_v.v.assign(&(fac * &self.vely.v));
        grad_t.v.assign(&(fac * &self.temp.v));
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
