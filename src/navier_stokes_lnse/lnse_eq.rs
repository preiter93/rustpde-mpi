//! Implement equations for `Navier2DLnse`
use super::Navier2DLnse;
use crate::field::BaseSpace;
use crate::navier_stokes::functions::{conv_term, dealias, norm_l2_c64, norm_l2_f64};
use crate::solver::{hholtz_adi::HholtzAdi, poisson::Poisson, Solve};
use crate::types::Scalar;
use ndarray::ScalarOperand;
use ndarray::{Array2, Ix2};
use num_complex::Complex;
use std::ops::Mul;

/// General
impl<T, S> Navier2DLnse<T, S>
where
    T: Scalar,
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
{
    /// Divergence: duxdx + duydy
    pub(crate) fn div(&mut self) -> Array2<T> {
        self.zero_rhs();
        self.rhs = &self.rhs + &self.velx.gradient([1, 0], Some(self.scale));
        self.rhs = &self.rhs + &self.vely.gradient([0, 1], Some(self.scale));
        self.rhs.to_owned()
    }
}

/// Return L2 norm of divergence
pub trait DivNorm {
    /// Return L2 norm of divergence
    fn div_norm(&mut self) -> f64;
}

impl<S> DivNorm for Navier2DLnse<f64, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = f64>,
{
    /// Return L2 norm of divergence
    fn div_norm(&mut self) -> f64 {
        norm_l2_f64(&self.div())
    }
}

impl<S> DivNorm for Navier2DLnse<Complex<f64>, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = Complex<f64>>,
{
    /// Return L2 norm of divergence
    fn div_norm(&mut self) -> f64 {
        norm_l2_c64(&self.div())
    }
}

impl<T, S> Navier2DLnse<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar,
{
    /// Convection term for ux
    pub(crate) fn conv_velx(&mut self, velx: &Array2<f64>, vely: &Array2<f64>) -> Array2<T> {
        let scale = Some(self.scale);
        let ux_mean = &self.mean.velx.v;
        let uy_mean = &self.mean.vely.v;
        // + ux * dUdx + uy * dUdy
        let mut conv = conv_term(velx, &self.mean.velx, &mut self.field.space, [1, 0], scale);
        conv += &conv_term(vely, &self.mean.velx, &mut self.field.space, [0, 1], scale);
        // + Ux * dudx + Uy * dudy
        conv += &conv_term(ux_mean, &self.velx, &mut self.field.space, [1, 0], scale);
        conv += &conv_term(uy_mean, &self.velx, &mut self.field.space, [0, 1], scale);
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        dealias(&mut self.field);
        self.field.vhat.to_owned()
    }

    /// Convection term for uy
    pub(crate) fn conv_vely(&mut self, velx: &Array2<f64>, vely: &Array2<f64>) -> Array2<T> {
        let scale = Some(self.scale);
        let ux_mean = &self.mean.velx.v;
        let uy_mean = &self.mean.vely.v;
        // + ux * dVdx + uy * dVdy
        let mut conv = conv_term(velx, &self.mean.vely, &mut self.field.space, [1, 0], scale);
        conv += &conv_term(vely, &self.mean.vely, &mut self.field.space, [0, 1], scale);
        // + Ux * dvdx + Uy * dvdy
        conv += &conv_term(ux_mean, &self.vely, &mut self.field.space, [1, 0], scale);
        conv += &conv_term(uy_mean, &self.vely, &mut self.field.space, [0, 1], scale);
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        dealias(&mut self.field);
        self.field.vhat.to_owned()
    }

    /// Convection term for temp
    pub(crate) fn conv_temp(&mut self, velx: &Array2<f64>, vely: &Array2<f64>) -> Array2<T> {
        let scale = Some(self.scale);
        let ux_mean = &self.mean.velx.v;
        let uy_mean = &self.mean.vely.v;
        // + ux * dTdx + uy * dTdy
        let mut conv = conv_term(velx, &self.mean.temp, &mut self.field.space, [1, 0], scale);
        conv += &conv_term(vely, &self.mean.temp, &mut self.field.space, [0, 1], scale);
        // + Ux * dtdx + Uy * dtdy
        conv += &conv_term(ux_mean, &self.temp, &mut self.field.space, [1, 0], scale);
        conv += &conv_term(uy_mean, &self.temp, &mut self.field.space, [0, 1], scale);
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        dealias(&mut self.field);
        self.field.vhat.to_owned()
    }
}

impl<T, S> Navier2DLnse<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar,
{
    /// Convection term for ux
    #[allow(dead_code)]
    pub(crate) fn conv_velx_adjoint(
        &mut self,
        velx: &Array2<f64>,
        vely: &Array2<f64>,
        temp: &Array2<f64>,
    ) -> Array2<T> {
        let scale = Some(self.scale);
        let velx_mean = &self.mean.velx;
        let vely_mean = &self.mean.vely;
        let temp_mean = &self.mean.temp;
        // + Ui(di uj*)
        let mut conv = conv_term(
            &velx_mean.v,
            &self.velx,
            &mut self.field.space,
            [1, 0],
            scale,
        );
        conv += &conv_term(
            &vely_mean.v,
            &self.velx,
            &mut self.field.space,
            [0, 1],
            scale,
        );
        // - (dj Ui) ui*
        conv -= &conv_term(velx, velx_mean, &mut self.field.space, [1, 0], scale);
        conv -= &conv_term(vely, vely_mean, &mut self.field.space, [1, 0], scale);
        conv -= &conv_term(temp, temp_mean, &mut self.field.space, [1, 0], scale);
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
        temp: &Array2<f64>,
    ) -> Array2<T> {
        let scale = Some(self.scale);
        let velx_mean = &self.mean.velx;
        let vely_mean = &self.mean.vely;
        let temp_mean = &self.mean.temp;
        // + Ui(di uj*)
        let mut conv = conv_term(
            &velx_mean.v,
            &self.vely,
            &mut self.field.space,
            [1, 0],
            scale,
        );
        conv += &conv_term(
            &vely_mean.v,
            &self.vely,
            &mut self.field.space,
            [0, 1],
            scale,
        );
        // - (dj Ui) ui*
        conv -= &conv_term(velx, velx_mean, &mut self.field.space, [0, 1], scale);
        conv -= &conv_term(vely, vely_mean, &mut self.field.space, [0, 1], scale);
        conv -= &conv_term(temp, temp_mean, &mut self.field.space, [0, 1], scale);
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
        _velx: &Array2<f64>,
        _vely: &Array2<f64>,
        _temp: &Array2<f64>,
    ) -> Array2<T> {
        let scale = Some(self.scale);
        let velx_mean = &self.mean.velx;
        let vely_mean = &self.mean.vely;
        // + Ui(di uj*)
        let mut conv = conv_term(
            &velx_mean.v,
            &self.temp,
            &mut self.field.space,
            [1, 0],
            scale,
        );
        conv += &conv_term(
            &vely_mean.v,
            &self.temp,
            &mut self.field.space,
            [0, 1],
            scale,
        );
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        dealias(&mut self.field);
        self.field.vhat.to_owned()
    }
}

/// Pressure update
impl<T, S> Navier2DLnse<T, S>
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
        self.velx.vhat = &self.velx.vhat + &self.velx.space.from_ortho(&dp_dx);
        self.vely.vhat = &self.vely.vhat + &self.vely.space.from_ortho(&dp_dy);
    }

    /// Update pressure field
    /// $$
    /// presnew = pres - nu * div + 1/dt * pseu
    /// $$
    pub(crate) fn update_pres(&mut self, div: &Array2<T>) {
        // self.pres.vhat = &self.pres.vhat - &(div * *self.params.get("nu").unwrap());
        // let inv_dt: T = (1. / self.dt).into();
        // self.pres.vhat = &self.pres.vhat + &(&self.pseu.to_ortho() * inv_dt);

        let nu = self.params.get("nu").unwrap();
        let a: f64 = -1. * nu;
        let b: f64 = 1. / self.dt;
        self.pres.vhat =
            &self.pres.vhat + &div.mapv(|x| x * a) + &self.pseu.to_ortho().mapv(|x| x * b);
    }
}

/// Solve pressure field
impl<T, S> Navier2DLnse<T, S>
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

/// Solve momentum and temperature equations
impl<T, S> Navier2DLnse<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar + Mul<f64, Output = T>,
    HholtzAdi<f64, 2>: Solve<T, Ix2>,
{
    /// Solve horizontal momentum equation
    /// $$
    /// (1 - \delta t  \mathcal{D}) u\\_new = -dt*C(u) - \delta t grad(p) + \delta t f + u
    /// $$
    pub(crate) fn solve_velx(&mut self, ux: &Array2<f64>, uy: &Array2<f64>) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.velx.to_ortho();
        // + pres
        self.rhs -= &(self.pres.gradient([1, 0], Some(self.scale)) * self.dt);
        // + convection
        let conv = self.conv_velx(ux, uy) * self.dt;
        self.rhs -= &conv;
        // solve lhs
        self.solver_hholtz[0].solve_par(&self.rhs, &mut self.velx.vhat, 0);
    }

    /// Solve vertical momentum equation
    pub(crate) fn solve_vely(&mut self, ux: &Array2<f64>, uy: &Array2<f64>, buoy: &Array2<T>) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.vely.to_ortho();
        // + pres
        self.rhs -= &(self.pres.gradient([0, 1], Some(self.scale)) * self.dt);
        // + buoyancy
        self.rhs += &(buoy * self.dt);
        // + convection
        let conv = self.conv_vely(ux, uy) * self.dt;
        self.rhs -= &conv;
        // solve lhs
        self.solver_hholtz[1].solve_par(&self.rhs, &mut self.vely.vhat, 0);
    }

    /// Solve temperature equation:
    /// $$
    /// (1 - dt*D) temp\\_new = -dt*C(temp) + dt*fbc + temp
    /// $$
    pub(crate) fn solve_temp(&mut self, ux: &Array2<f64>, uy: &Array2<f64>) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.temp.to_ortho();
        // + convection
        let conv = self.conv_temp(ux, uy) * self.dt;
        self.rhs -= &conv;
        // solve lhs
        self.solver_hholtz[2].solve_par(&self.rhs, &mut self.temp.vhat, 0);
    }
}
