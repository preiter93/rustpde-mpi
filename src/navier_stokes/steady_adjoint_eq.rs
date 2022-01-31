//! Implement equations for `Navier2DAdjoint`
#![allow(clippy::similar_names)]
use super::functions::{conv_term, dealias, norm_l2_c64, norm_l2_f64};
use super::steady_adjoint::Navier2DAdjoint;
use crate::field::BaseSpace;
use crate::solver::{hholtz_adi::HholtzAdi, poisson::Poisson, Solve};
use crate::types::Scalar;
use ndarray::{Array2, Ix2, ScalarOperand};
use num_complex::Complex;
use std::ops::Mul;

/// General
impl<T, S> Navier2DAdjoint<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar,
{
    /// Divergence: duxdx + duydy
    pub fn div(&mut self) -> Array2<T> {
        self.zero_rhs();
        self.rhs = &self.rhs + &self.navier.velx.gradient([1, 0], Some(self.scale));
        self.rhs = &self.rhs + &self.navier.vely.gradient([0, 1], Some(self.scale));
        self.rhs.to_owned()
    }
}

/// Return L2 norm of divergence
pub trait DivNorm {
    /// Return L2 norm of divergence
    fn div_norm(&mut self) -> f64;
}

impl<S> DivNorm for Navier2DAdjoint<f64, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = f64>,
{
    /// Return L2 norm of divergence
    fn div_norm(&mut self) -> f64 {
        norm_l2_f64(&self.div())
    }
}

impl<S> DivNorm for Navier2DAdjoint<Complex<f64>, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = Complex<f64>>,
{
    /// Return L2 norm of divergence
    fn div_norm(&mut self) -> f64 {
        norm_l2_c64(&self.div())
    }
}

/// Convection
impl<T, S> Navier2DAdjoint<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar + std::fmt::LowerExp,
{
    /// Convection term for temperature
    pub(crate) fn conv_temp(&mut self, velx: &Array2<f64>, vely: &Array2<f64>) -> Array2<T> {
        let scale = Some(self.scale);
        // + velx * dTdx + vely * dTdy
        let mut conv = conv_term(velx, &self.temp, &mut self.field.space, [1, 0], scale);
        conv += &conv_term(vely, &self.temp, &mut self.field.space, [0, 1], scale);
        // + bc contribution
        if let Some(field) = &self.tempbc {
            conv += &conv_term(velx, field, &mut self.field.space, [1, 0], scale);
            conv += &conv_term(vely, field, &mut self.field.space, [0, 1], scale);
        }
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        dealias(&mut self.field);
        self.field.vhat.to_owned()
    }

    /// Convection term for velx
    pub(crate) fn conv_velx(&mut self, velx: &Array2<f64>, vely: &Array2<f64>) -> Array2<T> {
        let scale = Some(self.scale);
        // + velx * dudx + vely * dudy
        let mut conv = conv_term(velx, &self.velx, &mut self.field.space, [1, 0], scale);
        conv += &conv_term(vely, &self.velx, &mut self.field.space, [0, 1], scale);
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        dealias(&mut self.field);
        self.field.vhat.to_owned()
    }

    /// Convection term for vely
    pub(crate) fn conv_vely(&mut self, velx: &Array2<f64>, vely: &Array2<f64>) -> Array2<T> {
        let scale = Some(self.scale);
        // + velx * dudx + vely * dudy
        let mut conv = conv_term(velx, &self.vely, &mut self.field.space, [1, 0], scale);
        conv += &conv_term(vely, &self.vely, &mut self.field.space, [0, 1], scale);
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        dealias(&mut self.field);
        self.field.vhat.to_owned()
    }
}

/// Pressure update
impl<T, S> Navier2DAdjoint<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar + ScalarOperand + From<f64> + Mul<f64, Output = T>,
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
        let mut dp_dx = self.pseu.gradient([1, 0], Some(self.scale));
        let mut dp_dy = self.pseu.gradient([0, 1], Some(self.scale));
        dp_dx *= c_t;
        dp_dy *= c_t;
        self.navier.velx.vhat = &self.navier.velx.vhat + &self.navier.velx.space.from_ortho(&dp_dx);
        self.navier.vely.vhat = &self.navier.vely.vhat + &self.navier.vely.space.from_ortho(&dp_dy);
    }

    /// Update pressure field
    /// $$
    /// presnew = pres - nu * div + 1/dt * pseu
    /// $$
    pub(crate) fn update_pres(&mut self, div: &Array2<T>) {
        // let nu = self.params.get("nu").unwrap();
        // let a: f64 = -1. * nu;
        let b: f64 = 1. / self.dt;
        // self.pres.vhat =
        //     &self.pres.vhat + &div.mapv(|x| x * a) + &self.pseu.to_ortho().mapv(|x| x * b);
        self.pres.vhat = &self.pres.vhat + &self.pseu.to_ortho().mapv(|x| x * b);
    }
}

/// Solve pressure field
impl<T, S> Navier2DAdjoint<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar,
    Poisson<f64, 2>: Solve<T, Ix2>,
{
    /// Solve pressure poisson equation
    /// $$
    /// D2 pres = f
    /// $$
    /// pseu: pseudo pressure ( in code it is pres\[1\] )
    pub(crate) fn solve_pres(&mut self, f: &Array2<T>) {
        self.solver_pres.solve_par(&f, &mut self.pseu.vhat, 0);
        // Remove singularity
        self.pseu.vhat[[0, 0]] = T::zero();
    }
}

impl<T, S> Navier2DAdjoint<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar + std::fmt::Debug + std::fmt::LowerExp,
{
    /// Convection term for ux
    #[allow(dead_code)]
    pub(crate) fn conv_velx_adjoint(
        &mut self,
        velx: &Array2<f64>,
        vely: &Array2<f64>,
        velx_adj: &Array2<f64>,
        vely_adj: &Array2<f64>,
        temp_adj: &Array2<f64>,
    ) -> Array2<T> {
        let scale = Some(self.scale);
        let velx_mean = &self.navier.velx;
        let vely_mean = &self.navier.vely;
        let temp_mean = &self.navier.temp;
        let space = &mut self.field.space;
        // + Ui(di uj*)
        let mut conv = conv_term(velx, &self.velx, space, [1, 0], scale);
        conv += &conv_term(vely, &self.velx, space, [0, 1], scale);

        // - (dj Ui) ui*
        // conv -= &conv_term(velx_adj, velx_mean, space, [1, 0], scale);
        // conv -= &conv_term(vely_adj, vely_mean, space, [1, 0], scale);
        conv += &conv_term(velx, &self.velx, space, [1, 0], scale);
        conv += &conv_term(vely, &self.vely, space, [1, 0], scale);

        conv -= &conv_term(temp_adj, temp_mean, space, [1, 0], scale);
        if let Some(field) = &self.tempbc {
            conv -= &conv_term(temp_adj, &field, space, [1, 0], scale);
        }
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        dealias(&mut self.field);
        self.field.vhat.to_owned()
    }

    /// Convection term for uy
    #[allow(dead_code)]
    pub(crate) fn conv_vely_adjoint(
        &mut self,
        velx: &Array2<f64>,
        vely: &Array2<f64>,
        velx_adj: &Array2<f64>,
        vely_adj: &Array2<f64>,
        temp_adj: &Array2<f64>,
    ) -> Array2<T> {
        let scale = Some(self.scale);
        let velx_mean = &self.navier.velx;
        let vely_mean = &self.navier.vely;
        let temp_mean = &self.navier.temp;
        let space = &mut self.field.space;
        // + Ui(di uj*)
        let mut conv = conv_term(velx, &self.vely, space, [1, 0], scale);
        conv += &conv_term(vely, &self.vely, space, [0, 1], scale);
        // - (dj Ui) ui*
        // conv -= &conv_term(velx_adj, velx_mean, space, [0, 1], scale);
        // conv -= &conv_term(vely_adj, vely_mean, space, [0, 1], scale);
        conv += &conv_term(velx, &self.velx, space, [0, 1], scale);
        conv += &conv_term(vely, &self.vely, space, [0, 1], scale);
        conv -= &conv_term(temp_adj, temp_mean, space, [0, 1], scale);
        if let Some(field) = &self.tempbc {
            conv -= &conv_term(temp_adj, &field, space, [0, 1], scale);
        }
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        dealias(&mut self.field);
        self.field.vhat.to_owned()
    }

    /// Convection term for temp
    #[allow(dead_code)]
    pub(crate) fn conv_temp_adjoint(
        &mut self,
        velx: &Array2<f64>,
        vely: &Array2<f64>,
        _velx_adj: &Array2<f64>,
        _vely_adj: &Array2<f64>,
        _temp_adj: &Array2<f64>,
    ) -> Array2<T> {
        let scale = Some(self.scale);
        let velx_mean = &self.navier.velx;
        let vely_mean = &self.navier.vely;
        let space = &mut self.field.space;
        // + Ui(di uj*)
        let mut conv = conv_term(velx, &self.temp, space, [1, 0], scale);
        conv += &conv_term(vely, &self.temp, space, [0, 1], scale);
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        dealias(&mut self.field);
        self.field.vhat.to_owned()
    }
}

/// Solve momentum and temperature equations
impl<T, S> Navier2DAdjoint<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar + Mul<f64, Output = T> + std::fmt::Debug + std::fmt::LowerExp,
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
        velx_adj: &Array2<f64>,
        vely_adj: &Array2<f64>,
        temp_adj: &Array2<f64>,
    ) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.navier.velx.to_ortho();
        // + pres
        self.rhs -= &(self.pres.gradient([1, 0], Some(self.scale)) * self.dt);
        // + convection
        let conv = self.conv_velx_adjoint(velx, vely, velx_adj, vely_adj, temp_adj) * self.dt;
        self.rhs += &conv;
        // + diffusion
        let nu = self.params.get("nu").unwrap();
        self.rhs += &(self.velx.gradient([2, 0], Some(self.scale)) * self.dt * *nu);
        self.rhs += &(self.velx.gradient([0, 2], Some(self.scale)) * self.dt * *nu);
        // update ux
        self.navier.velx.from_ortho(&self.rhs);
        // solve lhs
        // self.solver_hholtz[0].solve_par(&self.rhs, &mut self.velx.vhat, 0);
    }

    /// Solve vertical momentum equation
    pub(crate) fn solve_vely_adj(
        &mut self,
        velx: &Array2<f64>,
        vely: &Array2<f64>,
        velx_adj: &Array2<f64>,
        vely_adj: &Array2<f64>,
        temp_adj: &Array2<f64>,
    ) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.navier.vely.to_ortho();
        // + pres
        self.rhs -= &(self.pres.gradient([0, 1], Some(self.scale)) * self.dt);
        // + convection
        let conv = self.conv_vely_adjoint(velx, vely, velx_adj, vely_adj, temp_adj) * self.dt;
        self.rhs += &conv;
        // + diffusion
        let nu = self.params.get("nu").unwrap();
        self.rhs += &(self.vely.gradient([2, 0], Some(self.scale)) * self.dt * *nu);
        self.rhs += &(self.vely.gradient([0, 2], Some(self.scale)) * self.dt * *nu);
        // update uy
        self.navier.vely.from_ortho(&self.rhs);
        // // solve lhs
        // self.solver_hholtz[1].solve_par(&self.rhs, &mut self.vely.vhat, 0);
    }

    /// Solve temperature equation:
    /// $$
    /// (1 - dt*D) temp\\_new = -dt*C(temp) + dt*fbc + temp
    /// $$
    pub(crate) fn solve_temp_adj(
        &mut self,
        velx: &Array2<f64>,
        vely: &Array2<f64>,
        velx_adj: &Array2<f64>,
        vely_adj: &Array2<f64>,
        temp_adj: &Array2<f64>,
        // vely_vhat: &Array2<T>,
    ) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.navier.temp.to_ortho();
        // + convection
        let conv = self.conv_temp_adjoint(velx, vely, velx_adj, vely_adj, temp_adj) * self.dt;
        self.rhs += &conv;
        // + buoyancy (adjoint)
        self.rhs += &(&self.vely.to_ortho() * self.dt);
        // + diffusion
        let ka = self.params.get("ka").unwrap();
        self.rhs += &(self.temp.gradient([2, 0], Some(self.scale)) * self.dt * *ka);
        self.rhs += &(self.temp.gradient([0, 2], Some(self.scale)) * self.dt * *ka);
        // update ux
        self.navier.temp.from_ortho(&self.rhs);
        // // solve lhs
        // self.solver_hholtz[2].solve_par(&self.rhs, &mut self.temp.vhat, 0);
    }
}
