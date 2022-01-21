//! Calculate gradient of final energy with respect to initial field
//! using finite differences, i.e. perturb every grid point subsequently.
//! This is much more expensive, than adjoint based gradient computation
//! [`crate::lnse_adj_grad::grad_adjoint()`] and should only be used for testing.
use super::functions::energy;
use super::Navier2DLnse;
use crate::field::BaseSpace;
use crate::field::Field2;
use crate::io::traits::ReadWrite;
use crate::navier_stokes_lnse::lnse_eq::DivNorm;
use crate::solver::{hholtz_adi::HholtzAdi, poisson::Poisson, Solve};
use crate::types::Scalar;
use crate::{integrate, Integrate};
use ndarray::{Ix2, ScalarOperand};
use std::ops::Mul;

impl<T, S> Navier2DLnse<T, S>
where
    T: Scalar + Mul<f64, Output = T> + From<f64> + ScalarOperand,
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    Poisson<f64, 2>: Solve<T, Ix2>,
    HholtzAdi<f64, 2>: Solve<T, Ix2>,
    Field2<T, S>: ReadWrite<T>,
    Navier2DLnse<T, S>: DivNorm + Integrate,
{
    /// Calculate gradient from forward and backward loop
    ///
    /// # Return
    /// gradient
    pub fn grad_fd(
        &mut self,
        max_time: f64,
        save_intervall: Option<f64>,
    ) -> (Field2<T, S>, Field2<T, S>, Field2<T, S>) {
        // Weights of norm (vel, temp)
        let (b1, b2) = (0.5, 0.5);

        // Perturbation strength
        let eps = 1e-5;

        // Save base state variables
        let velx_0 = self.velx.v.to_owned();
        let vely_0 = self.vely.v.to_owned();
        let temp_0 = self.temp.v.to_owned();
        let velx_0_h = self.velx.vhat.to_owned();
        let vely_0_h = self.vely.vhat.to_owned();
        let temp_0_h = self.temp.vhat.to_owned();

        // Get energy of base state
        self.pres.v *= 0.;
        self.pseu.v *= 0.;
        self.pres.vhat *= T::zero();
        self.pseu.vhat *= T::zero();
        integrate(self, max_time, save_intervall);
        let e_base = energy(&mut self.velx, &mut self.vely, &mut self.temp, b1, b2);

        // Gradient with finite differences
        // Perturb at each coordinate and evaluate new energy
        let mut grad_u = Field2::new(&self.velx.space);
        let mut grad_v = Field2::new(&self.vely.space);
        let mut grad_t = Field2::new(&self.temp.space);

        // Perturb velx
        for i in 0..grad_u.v.shape()[0] {
            for j in 0..grad_u.v.shape()[1] {
                println!("{:?} {:?}", i, j);
                // Set base state
                self.time = 0.;
                self.velx.v.assign(&velx_0);
                self.vely.v.assign(&vely_0);
                self.temp.v.assign(&temp_0);
                self.velx.vhat.assign(&velx_0_h);
                self.vely.vhat.assign(&vely_0_h);
                self.temp.vhat.assign(&temp_0_h);
                //self.read("base.h5");
                self.pres.v *= 0.;
                self.pseu.v *= 0.;
                self.pres.vhat *= T::zero();
                self.pseu.vhat *= T::zero();
                // Perturb
                self.velx.v[[i, j]] += eps;
                self.velx.forward();
                // Integrate
                integrate(self, max_time, None);
                // Gradient
                let e_pert = energy(&mut self.velx, &mut self.vely, &mut self.temp, b1, b2);
                grad_u.v[[i, j]] = 1. / eps * (e_pert - e_base);
            }
        }

        // Perturb vely
        for i in 0..grad_v.v.shape()[0] {
            for j in 0..grad_v.v.shape()[1] {
                println!("{:?} {:?}", i, j);
                // Set base state
                self.time = 0.;
                self.velx.v.assign(&velx_0);
                self.vely.v.assign(&vely_0);
                self.temp.v.assign(&temp_0);
                self.velx.vhat.assign(&velx_0_h);
                self.vely.vhat.assign(&vely_0_h);
                self.temp.vhat.assign(&temp_0_h);
                //self.read("base.h5");
                self.pres.v *= 0.;
                self.pseu.v *= 0.;
                self.pres.vhat *= T::zero();
                self.pseu.vhat *= T::zero();
                // Perturb
                self.vely.v[[i, j]] += eps;
                self.vely.forward();
                // Integrate
                integrate(self, max_time, None);
                // Gradient
                let e_pert = energy(&mut self.velx, &mut self.vely, &mut self.temp, b1, b2);
                grad_v.v[[i, j]] = 1. / eps * (e_pert - e_base);
            }
        }

        // Perturb temp
        for i in 0..grad_t.v.shape()[0] {
            for j in 0..grad_t.v.shape()[1] {
                println!("{:?} {:?}", i, j);
                // Set base state
                self.time = 0.;
                self.velx.v.assign(&velx_0);
                self.vely.v.assign(&vely_0);
                self.temp.v.assign(&temp_0);
                self.velx.vhat.assign(&velx_0_h);
                self.vely.vhat.assign(&vely_0_h);
                self.temp.vhat.assign(&temp_0_h);
                //self.read("base.h5");
                self.pres.v *= 0.;
                self.pseu.v *= 0.;
                self.pres.vhat *= T::zero();
                self.pseu.vhat *= T::zero();
                // Perturb
                self.temp.v[[i, j]] += eps;
                self.temp.forward();
                // Integrate
                integrate(self, max_time, None);
                // Gradient
                let e_pert = energy(&mut self.velx, &mut self.vely, &mut self.temp, b1, b2);
                grad_t.v[[i, j]] = 1. / eps * (e_pert - e_base);
            }
        }

        // Write gradient
        grad_u.forward();
        grad_v.forward();
        grad_t.forward();
        let filename = "data/grad_fd.h5";
        grad_u.write_unwrap(&filename, "ux");
        grad_v.write_unwrap(&filename, "uy");
        grad_t.write_unwrap(&filename, "temp");
        (grad_u, grad_v, grad_t)
    }
}
