//! Calculate adjoint based sensitivity (gradient of
//! final energy with respect to initial field)
//! for non-linear Navier Stokes simulations
use super::Navier2DNonLin;
use crate::field::BaseSpace;
use crate::field::Field2;
use crate::navier_stokes::functions::{conv_term, dealias};
use crate::solver::{hholtz_adi::HholtzAdi, Solve};
use crate::types::Scalar;
use ndarray::{Array2, Ix2};
use num_traits::Zero;
use std::ops::Mul;

impl<T, S> Navier2DNonLin<T, S>
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
        velx_nonlin: &Field2<T, S>,
        vely_nonlin: &Field2<T, S>,
        temp_nonlin: &Field2<T, S>,
    ) -> Array2<T> {
        let scale = Some(self.scale);
        let velx_mean = &self.mean.velx;
        let vely_mean = &self.mean.vely;
        let temp_mean = &self.mean.temp;
        let space = &mut self.field.space;
        // // Same as linearized solver
        // + Ui(di uj*)
        let mut conv = conv_term(&velx_mean.v, &self.velx, space, [1, 0], scale);
        conv += &conv_term(&vely_mean.v, &self.velx, space, [0, 1], scale);
        // - (dj Ui) ui*
        conv -= &conv_term(velx, velx_mean, space, [1, 0], scale);
        conv -= &conv_term(vely, vely_mean, space, [1, 0], scale);
        conv -= &conv_term(temp, temp_mean, space, [1, 0], scale);
        // // Contributions from non linear term
        // + Ui(di uj*)
        conv += &conv_term(&velx_nonlin.v, &self.velx, space, [1, 0], scale);
        conv += &conv_term(&vely_nonlin.v, &self.velx, space, [0, 1], scale);
        // - (dj Ui) ui*
        conv -= &conv_term(velx, velx_nonlin, space, [1, 0], scale);
        conv -= &conv_term(vely, vely_nonlin, space, [1, 0], scale);
        conv -= &conv_term(temp, temp_nonlin, space, [1, 0], scale);
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
        velx_nonlin: &Field2<T, S>,
        vely_nonlin: &Field2<T, S>,
        temp_nonlin: &Field2<T, S>,
    ) -> Array2<T> {
        let scale = Some(self.scale);
        let velx_mean = &self.mean.velx;
        let vely_mean = &self.mean.vely;
        let temp_mean = &self.mean.temp;
        let space = &mut self.field.space;
        // // Same as linearized solver
        // + Ui(di uj*)
        let mut conv = conv_term(&velx_mean.v, &self.vely, space, [1, 0], scale);
        conv += &conv_term(&vely_mean.v, &self.vely, space, [0, 1], scale);
        // - (dj Ui) ui*
        conv -= &conv_term(velx, velx_mean, space, [0, 1], scale);
        conv -= &conv_term(vely, vely_mean, space, [0, 1], scale);
        conv -= &conv_term(temp, temp_mean, space, [0, 1], scale);
        // // Contributions from non linear term
        // + Ui(di uj*)
        conv += &conv_term(&velx_nonlin.v, &self.vely, space, [1, 0], scale);
        conv += &conv_term(&vely_nonlin.v, &self.vely, space, [0, 1], scale);
        // - (dj Ui) ui*
        conv -= &conv_term(velx, velx_nonlin, space, [0, 1], scale);
        conv -= &conv_term(vely, vely_nonlin, space, [0, 1], scale);
        conv -= &conv_term(temp, temp_nonlin, space, [0, 1], scale);
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
        velx_nonlin: &Field2<T, S>,
        vely_nonlin: &Field2<T, S>,
        _temp_nonlin: &Field2<T, S>,
    ) -> Array2<T> {
        let scale = Some(self.scale);
        let velx_mean = &self.mean.velx;
        let vely_mean = &self.mean.vely;
        let space = &mut self.field.space;
        // // Same as linearized solver
        // + Ui(di uj*)
        let mut conv = conv_term(&velx_mean.v, &self.temp, space, [1, 0], scale);
        conv += &conv_term(&vely_mean.v, &self.temp, space, [0, 1], scale);
        // // Contributions from non linear term
        // + Ui(di uj*)
        conv += &conv_term(&velx_nonlin.v, &self.temp, space, [1, 0], scale);
        conv += &conv_term(&vely_nonlin.v, &self.temp, space, [0, 1], scale);
        // -> spectral space
        self.field.v.assign(&conv);
        self.field.forward();
        dealias(&mut self.field);
        self.field.vhat.to_owned()
    }
}

/// Solve momentum and temperature equations
impl<T, S> Navier2DNonLin<T, S>
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
        velx_nonlin: &Field2<T, S>,
        vely_nonlin: &Field2<T, S>,
        temp_nonlin: &Field2<T, S>,
    ) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.velx.to_ortho();
        // + pres
        self.rhs -= &(self.pres.gradient([1, 0], Some(self.scale)) * self.dt);
        // + convection
        let conv = self.conv_velx_adjoint(velx, vely, temp, velx_nonlin, vely_nonlin, temp_nonlin)
            * self.dt;
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
        velx_nonlin: &Field2<T, S>,
        vely_nonlin: &Field2<T, S>,
        temp_nonlin: &Field2<T, S>,
    ) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.vely.to_ortho();
        // + pres
        self.rhs -= &(self.pres.gradient([0, 1], Some(self.scale)) * self.dt);
        // + convection
        let conv = self.conv_vely_adjoint(velx, vely, temp, velx_nonlin, vely_nonlin, temp_nonlin)
            * self.dt;
        self.rhs += &conv;
        // solve lhs
        self.solver_hholtz[1].solve_par(&self.rhs, &mut self.vely.vhat, 0);
    }

    /// Solve temperature equation:
    /// $$
    /// (1 - dt*D) temp\\_new = -dt*C(temp) + dt*fbc + temp
    /// $$
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn solve_temp_adj(
        &mut self,
        velx: &Array2<f64>,
        vely: &Array2<f64>,
        temp: &Array2<f64>,
        vely_vhat: &Array2<T>,
        velx_nonlin: &Field2<T, S>,
        vely_nonlin: &Field2<T, S>,
        temp_nonlin: &Field2<T, S>,
    ) {
        self.zero_rhs();
        // + old field
        self.rhs += &self.temp.to_ortho();
        // + convection
        let conv = self.conv_temp_adjoint(velx, vely, temp, velx_nonlin, vely_nonlin, temp_nonlin)
            * self.dt;
        self.rhs += &conv;
        // + buoyancy (adjoint)
        self.rhs += &(vely_vhat * self.dt);
        // solve lhs
        self.solver_hholtz[2].solve_par(&self.rhs, &mut self.temp.vhat, 0);
    }
}
