//! Some functions
use crate::field::BaseSpace;
use crate::mpi::{BaseSpaceMpi, FieldBaseMpi};
use crate::types::FloatNum;
use crate::types::Scalar;
use ndarray::Array2;
use ndarray::ScalarOperand;
use num_complex::Complex;
use std::ops::{Div, Mul};

/// Calculate u*dvdx
///
/// # Input
///
///    *field*: Field<Space2D, 2>
///        Contains field variable vhat in spectral space
///
///   *u*:  ndarray (2D)
///        Velocity field in physical space
///
///   *deriv*: [usize; 2]
///        \[1,0\] for partial x, \[0,1\] for partial y
///
/// # Return
/// Array of u*dvdx term in physical space.
///
/// Collect all convective terms, thatn transform to spectral space.
pub fn conv_term<T2, S>(
    field: &FieldBaseMpi<f64, f64, T2, S, 2>,
    deriv_field: &mut FieldBaseMpi<f64, f64, T2, S, 2>,
    u: &Array2<f64>,
    deriv: [usize; 2],
    scale: Option<[f64; 2]>,
) -> Array2<f64>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T2>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = T2>,
    T2: Scalar,
{
    //dvdx
    for v in deriv_field.vhat_x_pen.iter_mut() {
        *v = T2::zero();
    }
    deriv_field
        .vhat_x_pen
        .assign(&field.gradient_mpi(deriv, scale));
    deriv_field.backward_mpi();
    //u*dvdx
    u * &deriv_field.v_y_pen
}

/// Returns Nusselt number (heat flux at the plates)
/// $$
/// Nu = \langle - dTdz \rangle\\_x (0/H))
/// $$
pub fn eval_nu<A, T2, S>(
    temp: &mut FieldBaseMpi<A, A, T2, S, 2>,
    field: &mut FieldBaseMpi<A, A, T2, S, 2>,
    tempbc: &Option<FieldBaseMpi<A, A, T2, S, 2>>,
    scale: &[A; 2],
) -> A
where
    A: FloatNum,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 2, Physical = A, Spectral = T2>,
    T2: Scalar + Mul<A, Output = T2>,
{
    let two = A::one() + A::one();
    //self.temp.backward();
    field.vhat.assign(&temp.to_ortho());
    if let Some(x) = &tempbc {
        field.vhat = &field.vhat + &x.to_ortho();
    }
    let mut dtdz = field.gradient([0, 1], None) * -A::one();
    dtdz = dtdz * (A::one() / (scale[1] / two));
    field.vhat.assign(&dtdz);
    field.backward();
    let x_avg = field.average_axis(0);
    (x_avg[x_avg.len() - 1] + x_avg[0]) / two
}

/// Returns volumetric Nusselt number
/// $$
/// Nuvol = \langle uy*T/kappa - dTdz \rangle\\_V
/// $$
pub fn eval_nuvol<A, T2, S>(
    temp: &mut FieldBaseMpi<A, A, T2, S, 2>,
    uy: &mut FieldBaseMpi<A, A, T2, S, 2>,
    field: &mut FieldBaseMpi<A, A, T2, S, 2>,
    tempbc: &Option<FieldBaseMpi<A, A, T2, S, 2>>,
    kappa: A,
    scale: &[A; 2],
) -> A
where
    A: FloatNum,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 2, Physical = A, Spectral = T2>,
    T2: Scalar + Div<A, Output = T2>,
{
    let two = A::one() + A::one();
    // temp
    field.vhat.assign(&temp.to_ortho());
    if let Some(x) = &tempbc {
        field.vhat = &field.vhat + &x.to_ortho();
    }
    field.backward();
    // uy
    uy.backward();
    let uy_temp = &field.v * &uy.v;
    // dtdz
    let dtdz = field.gradient([0, 1], None) / (scale[1] * -A::one());
    field.vhat.assign(&dtdz);
    field.backward();
    let dtdz = &field.v;
    // Nuvol
    field.v = (dtdz + uy_temp / kappa) * two * scale[1];
    //average
    field.average()
}

/// Returns Reynolds number base on kinetic energy
/// $$
/// Re = U*L / nu
/// U = \sqrt{(ux^2 + uy^2)}
/// $$
pub fn eval_re<A, T2, S>(
    ux: &mut FieldBaseMpi<A, A, T2, S, 2>,
    uy: &mut FieldBaseMpi<A, A, T2, S, 2>,
    field: &mut FieldBaseMpi<A, A, T2, S, 2>,
    nu: A,
    scale: &[A; 2],
) -> A
where
    A: FloatNum,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 2, Physical = A, Spectral = T2>,
{
    ux.backward();
    uy.backward();
    let ekin = &ux.v.mapv(|x| x.powi(2)) + &uy.v.mapv(|x| x.powi(2));
    field.v.assign(&ekin.mapv(A::sqrt));
    let two = A::one() + A::one();
    field.v *= two * scale[1] / nu;
    field.average()
}
