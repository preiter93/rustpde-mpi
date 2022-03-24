//! Functions for `Navier2DLnse`
use crate::field::FieldBase;
use crate::types::{FloatNum, Scalar};
use funspace::BaseSpace;
use ndarray::Array2;
use ndarray::ScalarOperand;
use num_complex::Complex;
use std::ops::{Div, Mul};

/// b1 * ux**2 + b1 * uy**2 + b2 * temp**2
pub fn energy<A, T2, S>(
    velx: &mut FieldBase<A, A, T2, S, 2>,
    vely: &mut FieldBase<A, A, T2, S, 2>,
    temp: &mut FieldBase<A, A, T2, S, 2>,
    b1: A,
    b2: A,
) -> A
where
    A: std::ops::AddAssign + crate::types::FloatNum,
    Complex<A>: ndarray::ScalarOperand,
    S: BaseSpace<A, 2, Physical = A, Spectral = T2>,
{
    velx.backward();
    vely.backward();
    temp.backward();
    l2_norm(&velx.v, &velx.v, &vely.v, &vely.v, &temp.v, &temp.v, b1, b2)
}

/// Norm with three scalar products
// a1 * a2 + ...
#[allow(clippy::too_many_arguments)]
pub fn l2_norm<A>(
    a1: &Array2<A>,
    a2: &Array2<A>,
    b1: &Array2<A>,
    b2: &Array2<A>,
    c1: &Array2<A>,
    c2: &Array2<A>,
    beta1: A,
    beta2: A,
) -> A
where
    A: std::ops::AddAssign + crate::types::FloatNum,
{
    let mut s = A::zero();
    ndarray::Zip::from(a1)
        .and(a2)
        .and(b1)
        .and(b2)
        .and(c1)
        .and(c2)
        .for_each(|&x1, &x2, &y1, &y2, &z1, &z2| {
            s += beta1 * x1 * x2 + beta1 * y1 * y2 + beta2 * z1 * z2;
        });
    let fac = A::one() / (A::one() + A::one());
    fac * s
}

/// Returns Nusselt number (heat flux at the plates)
/// $$
/// Nu = \langle - dTdz \rangle\\_x (0/H))
/// $$
pub fn eval_nu<A, T2, S>(
    temp: &Array2<T2>,
    field: &mut FieldBase<A, A, T2, S, 2>,
    scale: &[A; 2],
) -> A
where
    A: FloatNum + ScalarOperand,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 2, Physical = A, Spectral = T2>,
    T2: Scalar + Mul<A, Output = T2>,
{
    let two = A::one() + A::one();
    field.vhat.assign(temp);
    let dtdz = field.gradient([0, 1], None) * (-two / scale[1]);
    field.vhat.assign(&dtdz);
    field.backward();
    let x_avg = field.average_axis(0);
    (x_avg[x_avg.len() - 1] + x_avg[0]) / two
}

/// Returns volumetric Nusselt number
/// $$
/// Nuvol = \langle vely*T/kappa - dTdz \rangle\\_V
/// $$
pub fn eval_nuvol<A, T2, S>(
    temp: &Array2<T2>,
    vely: &Array2<T2>,
    field: &mut FieldBase<A, A, T2, S, 2>,
    kappa: A,
    scale: &[A; 2],
) -> A
where
    A: FloatNum + ScalarOperand,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 2, Physical = A, Spectral = T2>,
    T2: Scalar + Div<A, Output = T2>,
{
    let two = A::one() + A::one();
    // temp
    field.vhat.assign(temp);
    field.backward();
    let temp = field.v.to_owned();
    // dtdz
    let dtdz = field.gradient([0, 1], None) / (scale[1] * -A::one());
    field.vhat.assign(&dtdz);
    field.backward();
    let dtdz = field.v.to_owned();
    // vely
    field.vhat.assign(vely);
    field.backward();
    let vely_temp = &field.v * &temp;
    // Nuvol
    field.v = (&dtdz + &vely_temp / kappa) * two * scale[1];
    //average
    field.average()
}

/// Returns Reynolds number base on kinetic energy
/// $$
/// Re = U*L / nu
/// U = \sqrt{(velx^2 + vely^2)}
/// $$
pub fn eval_re<A, T2, S>(
    velx: &Array2<T2>,
    vely: &Array2<T2>,
    field: &mut FieldBase<A, A, T2, S, 2>,
    nu: A,
    scale: &[A; 2],
) -> A
where
    A: FloatNum + ScalarOperand,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 2, Physical = A, Spectral = T2>,
{
    let two = A::one() + A::one();
    let velx_v = field.space.backward(velx);
    let vely_v = field.space.backward(vely);
    let ekin = &velx_v.mapv(|x| x.powi(2)) + &vely_v.mapv(|x| x.powi(2));
    field.v.assign(&ekin.mapv(A::sqrt));
    field.v *= two * scale[1] / nu;
    field.average()
}
