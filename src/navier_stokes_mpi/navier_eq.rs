//! Implement equations for navier-stokes simulations
use super::functions::{conv_term, dealias, norm_l2_c64, norm_l2_f64};
use super::navier::Navier2DMpi;
use crate::field::BaseSpace;
use crate::mpi::{all_gather_sum, BaseSpaceMpi};
use crate::solver::Solve;
use crate::solver_mpi::hholtz_adi::HholtzAdiMpi;
use crate::solver_mpi::poisson::PoissonMpi;
use crate::types::Scalar;
use ndarray::{Array2, Ix2, ScalarOperand};
use num_complex::Complex;
use std::ops::Mul;

/// General
impl<T, S> Navier2DMpi<'_, T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar,
{
    /// Divergence: duxdx + duydy
    pub fn div(&mut self) -> Array2<T> {
        self.zero_rhs();
        self.rhs += &self.ux.gradient_mpi([1, 0], Some(self.scale));
        self.rhs += &self.uy.gradient_mpi([0, 1], Some(self.scale));
        self.rhs.to_owned()
    }

    // /// Divergence: duxdx + duydy
    // pub(crate) fn div_global(&mut self) -> Array2<T> {
    //     let mut rhs = self.ux.gradient([1, 0], Some(self.scale));
    //     rhs += &self.uy.gradient([0, 1], Some(self.scale));
    //     rhs.to_owned()
    // }
}

/// Return L2 norm of divergence
pub trait DivNorm {
    /// Return L2 norm of divergence
    fn div_norm(&mut self) -> f64;
}

impl<S> DivNorm for Navier2DMpi<'_, f64, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = f64> + BaseSpaceMpi<f64, 2>,
{
    /// Return L2 norm of divergence
    fn div_norm(&mut self) -> f64 {
        let div = norm_l2_f64(&self.div()).powi(2);
        let mut div_global = 0.;
        all_gather_sum(self.universe, &div, &mut div_global);
        div_global.sqrt()
    }
}

impl<S> DivNorm for Navier2DMpi<'_, Complex<f64>, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = Complex<f64>> + BaseSpaceMpi<f64, 2>,
{
    /// Return L2 norm of divergence
    fn div_norm(&mut self) -> f64 {
        let div = norm_l2_c64(&self.div()).powi(2);
        let mut div_global = 0.;
        all_gather_sum(self.universe, &div, &mut div_global);
        div_global.sqrt()
    }
}

/// Convection
impl<T, S> Navier2DMpi<'_, T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar,
{
    /// Convection term for temperature
    pub(crate) fn conv_temp(&mut self, ux: &Array2<f64>, uy: &Array2<f64>) -> Array2<T> {
        let scale = Some(self.scale);
        // + ux * dTdx + uy * dTdy
        let mut conv = conv_term(ux, &self.temp, &mut self.field.space, [1, 0], scale);
        conv += &conv_term(uy, &self.temp, &mut self.field.space, [0, 1], scale);
        // + bc contribution
        if let Some(field) = &self.tempbc {
            conv += &conv_term(ux, field, &mut self.field.space, [1, 0], scale);
            conv += &conv_term(uy, field, &mut self.field.space, [0, 1], scale);
        }
        // -> spectral space
        self.field.v_y_pen.assign(&conv);
        self.field.forward_mpi();
        dealias(&mut self.field);
        self.field.vhat_x_pen.to_owned()
    }

    /// Convection term for ux
    pub(crate) fn conv_ux(&mut self, ux: &Array2<f64>, uy: &Array2<f64>) -> Array2<T> {
        let scale = Some(self.scale);
        // + ux * dudx + uy * dudy
        let mut conv = conv_term(ux, &self.ux, &mut self.field.space, [1, 0], scale);
        conv += &conv_term(uy, &self.ux, &mut self.field.space, [0, 1], scale);
        // -> spectral space
        self.field.v_y_pen.assign(&conv);
        self.field.forward_mpi();
        dealias(&mut self.field);
        self.field.vhat_x_pen.to_owned()
    }

    /// Convection term for uy
    pub(crate) fn conv_uy(&mut self, ux: &Array2<f64>, uy: &Array2<f64>) -> Array2<T> {
        let scale = Some(self.scale);
        // + ux * dudx + uy * dudy
        let mut conv = conv_term(ux, &self.uy, &mut self.field.space, [1, 0], scale);
        conv += &conv_term(uy, &self.uy, &mut self.field.space, [0, 1], scale);
        // -> spectral space
        self.field.v_y_pen.assign(&conv);
        self.field.forward_mpi();
        dealias(&mut self.field);
        self.field.vhat_x_pen.to_owned()
    }
}

/// Pressure update
impl<T, S> Navier2DMpi<'_, T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar + From<f64> + ScalarOperand,
{
    /// Correct velocity field.
    /// $$
    /// uxnew = ux - c*dpdx
    /// $$
    /// uynew = uy - c*dpdy
    /// $$
    #[allow(clippy::similar_names)]
    pub(crate) fn correct_velocity(&mut self, c: f64) {
        let c_t: T = (-c).into();
        let mut dp_dx = self.pseu.gradient_mpi([1, 0], Some(self.scale));
        let mut dp_dy = self.pseu.gradient_mpi([0, 1], Some(self.scale));
        dp_dx *= c_t;
        dp_dy *= c_t;
        self.ux.vhat_x_pen = &self.ux.vhat_x_pen + &self.ux.space.from_ortho_mpi(&dp_dx);
        self.uy.vhat_x_pen = &self.uy.vhat_x_pen + &self.uy.space.from_ortho_mpi(&dp_dy);
    }
}

impl<T, S> Navier2DMpi<'_, T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar + ScalarOperand + Mul<f64, Output = T>,
{
    /// Update pressure term
    /// $$
    /// presnew = pres - nu * div + 1/dt * pseu
    /// $$
    pub(crate) fn update_pres(&mut self, div: &Array2<T>) {
        let nu = self.params.get("nu").unwrap();
        let a: f64 = -1. * *nu;
        let b: f64 = 1. / self.dt;
        self.pres.vhat_x_pen = &self.pres.vhat_x_pen
            + &div.mapv(|x| x * a)
            + &self.pseu.to_ortho_mpi().mapv(|x| x * b);
    }
}

/// Solve pressure field
impl<T, S> Navier2DMpi<'_, T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar,
    PoissonMpi<f64, S, 2>: Solve<T, Ix2>,
{
    /// Solve pressure poisson equation
    /// $$
    /// D2 pres = f
    /// $$
    /// pseu: pseudo pressure ( in code it is pres\[1\] )
    pub(crate) fn solve_pres(&mut self, f: &Array2<T>) {
        self.solver_pres.solve(f, &mut self.pseu.vhat_x_pen, 0);
        // Remove singularity
        if self.nrank() == 0 {
            self.pseu.vhat_x_pen[[0, 0]] = T::zero();
        }
    }
}

/// Solve momentum and temperature equations
impl<T, S> Navier2DMpi<'_, T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar + Mul<f64, Output = T>,
    HholtzAdiMpi<f64, S, 2>: Solve<T, Ix2>,
{
    /// Solve horizontal momentum equation
    /// $$
    /// (1 - \delta t  \mathcal{D}) u\\_new = -dt*C(u) - \delta t grad(p) + \delta t f + u
    /// $$
    pub(crate) fn solve_ux(&mut self, ux: &Array2<f64>, uy: &Array2<f64>) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.ux.to_ortho_mpi();
        // + pres
        self.rhs -= &(self.pres.gradient_mpi([1, 0], Some(self.scale)) * self.dt);
        // + convection
        let conv = self.conv_ux(ux, uy) * self.dt; //.mapv(|x| x * self.dt);
        self.rhs -= &conv;
        // solve lhs
        self.solver_hholtz[0].solve(&self.rhs, &mut self.ux.vhat_x_pen, 0);
    }

    /// Solve vertical momentum equation
    pub(crate) fn solve_uy(&mut self, ux: &Array2<f64>, uy: &Array2<f64>, buoy: &Array2<T>) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.uy.to_ortho_mpi();
        // + pres
        self.rhs -= &(self.pres.gradient_mpi([0, 1], Some(self.scale)) * self.dt);
        // + buoyancy
        self.rhs += &(buoy * self.dt);
        // + convection
        let conv = self.conv_uy(ux, uy) * self.dt;
        self.rhs -= &conv;
        // solve lhs
        self.solver_hholtz[1].solve(&self.rhs, &mut self.uy.vhat_x_pen, 0);
    }

    /// Solve temperature equation:
    /// $$
    /// (1 - dt*D) temp\\_new = -dt*C(temp) + dt*fbc + temp
    /// $$
    pub(crate) fn solve_temp(&mut self, ux: &Array2<f64>, uy: &Array2<f64>) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.temp.to_ortho_mpi();
        // + diffusion bc contribution
        if let Some(field) = &self.tempbc {
            let ka = self.params.get("ka").unwrap();
            self.rhs += &(field.gradient_mpi([2, 0], Some(self.scale)) * self.dt * *ka);
            self.rhs += &(field.gradient_mpi([0, 2], Some(self.scale)) * self.dt * *ka);
        }
        // + convection
        let conv = self.conv_temp(ux, uy) * self.dt;
        self.rhs -= &conv;
        // solve lhs
        self.solver_hholtz[2].solve(&self.rhs, &mut self.temp.vhat_x_pen, 0);
    }
}
