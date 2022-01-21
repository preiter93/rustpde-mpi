//! Implement adjoint stability equations for `Navier2DLnse`
use super::Navier2DLnse;
use crate::field::BaseSpace;
use crate::solver::{hholtz_adi::HholtzAdi, Solve};
use crate::types::Scalar;
use ndarray::{Array2, Ix2};
use num_traits::Zero;
use std::ops::Mul;

/// Solve momentum and temperature equations
impl<T, S> Navier2DLnse<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Copy + Clone + Zero + Scalar + Mul<f64, Output = T>,
    HholtzAdi<f64, 2>: Solve<T, Ix2>,
{
    /// Solve horizontal momentum equation
    /// $$
    /// (1 - \delta t  \mathcal{D}) u\\_new = -dt*C(u) - \delta t grad(p) + \delta t f + u
    /// $$
    pub(crate) fn solve_velx_adj(
        &mut self,
        velx: &Array2<f64>,
        vely: &Array2<f64>,
        temp: &Array2<f64>,
    ) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.velx.to_ortho();
        // + pres
        self.rhs -= &(self.pres.gradient([1, 0], Some(self.scale)) * self.dt);
        // + convection
        let conv = self.conv_velx_adjoint(velx, vely, temp) * self.dt;
        self.rhs += &conv;
        // solve lhs
        self.solver_hholtz[0].solve_par(&self.rhs, &mut self.velx.vhat, 0);
    }

    /// Solve vertical momentum equation
    pub(crate) fn solve_vely_adj(
        &mut self,
        velx: &Array2<f64>,
        vely: &Array2<f64>,
        temp: &Array2<f64>,
    ) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.vely.to_ortho();
        // + pres
        self.rhs -= &(self.pres.gradient([0, 1], Some(self.scale)) * self.dt);
        // + convection
        let conv = self.conv_vely_adjoint(velx, vely, temp) * self.dt;
        self.rhs += &conv;
        // solve lhs
        self.solver_hholtz[1].solve_par(&self.rhs, &mut self.vely.vhat, 0);
    }

    /// Solve temperature equation:
    /// $$
    /// (1 - dt*D) temp\\_new = -dt*C(temp) + dt*fbc + temp
    /// $$
    pub(crate) fn solve_temp_adj(
        &mut self,
        velx: &Array2<f64>,
        vely: &Array2<f64>,
        temp: &Array2<f64>,
        vely_vhat: &Array2<T>,
    ) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.temp.to_ortho();
        // + convection
        let conv = self.conv_temp_adjoint(velx, vely, temp) * self.dt;
        self.rhs += &conv;
        // + buoyancy (adjoint)
        self.rhs += &(vely_vhat * self.dt);
        // solve lhs
        self.solver_hholtz[2].solve_par(&self.rhs, &mut self.temp.vhat, 0);
    }
}
