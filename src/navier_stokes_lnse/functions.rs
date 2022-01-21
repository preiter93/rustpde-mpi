//! Functions for `Navier2DLnse`
use crate::field::FieldBase;
use funspace::BaseSpace;
use ndarray::Array2;
use num_complex::Complex;

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
