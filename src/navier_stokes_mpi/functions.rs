//! Some functions
use crate::field::BaseSpace;
use crate::field_mpi::{Field2Mpi, FieldBaseMpi};
use crate::mpi::BaseSpaceMpi;
use crate::types::FloatNum;
use crate::types::Scalar;
use ndarray::ScalarOperand;
use ndarray::{s, Array2};
use num_complex::Complex;
use num_traits::Zero;
use std::ops::{Div, Mul};

/// Return viscosity from Ra, Pr, and height of the cell
pub fn get_nu(ra: f64, pr: f64, height: f64) -> f64 {
    let f = pr / (ra / height.powf(3.0));
    f.sqrt()
}

/// Return diffusivity from Ra, Pr, and height of the cell
pub fn get_ka(ra: f64, pr: f64, height: f64) -> f64 {
    let f = 1. / ((ra / height.powf(3.0)) * pr);
    f.sqrt()
}

/// Return l2 norm of real array
pub fn norm_l2_f64(array: &Array2<f64>) -> f64 {
    array.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
}

/// Return l2 norm of complex array
pub fn norm_l2_c64(array: &Array2<Complex<f64>>) -> f64 {
    array
        .iter()
        .map(|x| x.re.powi(2) + x.im.powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Calculate u*dvdx
///
/// # Input
///   *u*: ndarray (2D)
///        Convection Velocity field in physical space
///
///   *field*: Field<Space2D, 2>
///       Contains field variable vhat in spectral space
///
///   *space*: Field<Space2D, 2>
///       Space for transformation
///
///   *deriv*: [usize; 2]
///        \[1,0\] for partial x, \[0,1\] for partial y
///
/// # Return
/// Array of u*dvdx term in physical space.
///
/// Collect all convective terms, thatn transform to spectral space.
pub fn conv_term<T2, S>(
    u: &Array2<f64>,
    field: &FieldBaseMpi<f64, f64, T2, S, 2>,
    space: &mut S,
    deriv: [usize; 2],
    scale: Option<[f64; 2]>,
) -> Array2<f64>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T2>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = T2>,
    T2: Scalar,
{
    // u *dvdx
    u * space.backward_mpi(&field.gradient_mpi(deriv, scale))
}

/// Dealias field (2/3 rule)
pub fn dealias<S, T2>(field: &mut Field2Mpi<T2, S>)
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T2>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = T2>,
    T2: Zero + Clone + Copy,
{
    let zero = T2::zero();
    let n_x: usize = field.vhat.shape()[0] * 2 / 3;
    let n_y: usize = field.vhat.shape()[1] * 2 / 3;
    // x dim
    field.vhat_x_pen.slice_mut(s![n_x.., ..]).fill(zero);
    // y dim is broken
    let dcp = field.space.get_decomp_from_global_shape(field.vhat.shape());
    if n_y < dcp.x_pencil.en[1] {
        let yst = if n_y > dcp.x_pencil.st[1] {
            n_y - dcp.x_pencil.st[1]
        } else {
            0
        };
        field.vhat_x_pen.slice_mut(s![.., yst..]).fill(zero);
    }
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
    let dtdz = field.gradient([0, 1], None) * (-two / scale[1]);
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
    // temp * uy
    field.vhat.assign(&temp.to_ortho());
    if let Some(x) = &tempbc {
        field.vhat = &field.vhat + &x.to_ortho();
    }
    field.backward();
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

/// Returns Nusselt number (heat flux at the plates)
/// $$
/// Nu = \langle - dTdz \rangle\\_x (0/H))
/// $$
pub fn eval_nu_mpi<A, T2, S>(
    temp: &mut FieldBaseMpi<A, A, T2, S, 2>,
    field: &mut FieldBaseMpi<A, A, T2, S, 2>,
    tempbc: &Option<FieldBaseMpi<A, A, T2, S, 2>>,
    scale: &[A; 2],
) -> A
where
    A: FloatNum + crate::mpi::Equivalence + std::iter::Sum,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 2, Physical = A, Spectral = T2>
        + BaseSpaceMpi<A, 2, Physical = A, Spectral = T2>,
    T2: Scalar + Mul<A, Output = T2>,
{
    let two = A::one() + A::one();
    field.vhat_x_pen.assign(&temp.to_ortho_mpi());
    if let Some(bc) = &tempbc {
        field.vhat_x_pen = &field.vhat_x_pen + &bc.to_ortho_mpi();
    }
    let dtdz = field.gradient_mpi([0, 1], None) * (-two / scale[1]);
    field.vhat_x_pen.assign(&dtdz);
    field.backward_mpi();
    let x_avg = field.average_axis_mpi(0);
    // check wether rank holds bottom or top, both or none
    let nrank = temp.space.get_nrank();
    let nprocs = temp.space.get_nprocs();
    let nu = if nrank == 0 && nrank == nprocs - 1 {
        x_avg[x_avg.len() - 1] + x_avg[0]
    } else if nrank == 0 {
        x_avg[0]
    } else if nrank == nprocs - 1 {
        x_avg[x_avg.len() - 1]
    } else {
        A::zero()
    };
    let mut nu_global = A::zero();
    crate::mpi::all_gather_sum(&temp.space.get_universe(), &nu, &mut nu_global);
    nu_global / two
}

/// Returns volumetric Nusselt number
/// $$
/// Nuvol = \langle uy*T/kappa - dTdz \rangle\\_V
/// $$
pub fn eval_nuvol_mpi<A, T2, S>(
    temp: &mut FieldBaseMpi<A, A, T2, S, 2>,
    uy: &mut FieldBaseMpi<A, A, T2, S, 2>,
    field: &mut FieldBaseMpi<A, A, T2, S, 2>,
    tempbc: &Option<FieldBaseMpi<A, A, T2, S, 2>>,
    kappa: A,
    scale: &[A; 2],
) -> A
where
    A: FloatNum + crate::mpi::Equivalence + std::iter::Sum,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 2, Physical = A, Spectral = T2>
        + BaseSpaceMpi<A, 2, Physical = A, Spectral = T2>,
    T2: Scalar + Div<A, Output = T2>,
{
    let two = A::one() + A::one();
    // temp * uy
    field.vhat_x_pen.assign(&temp.to_ortho_mpi());
    if let Some(bc) = &tempbc {
        field.vhat_x_pen = &field.vhat_x_pen + &bc.to_ortho_mpi();
    }
    field.backward_mpi();
    uy.backward_mpi();
    let uy_temp = &field.v_y_pen * &uy.v_y_pen;
    // dtdz
    let dtdz = field.gradient_mpi([0, 1], None) / (scale[1] * -A::one());
    field.vhat_x_pen.assign(&dtdz);
    field.backward_mpi();
    let dtdz = &field.v_y_pen;
    // Nuvol
    field.v_y_pen = (dtdz + uy_temp / kappa) * two * scale[1];
    //average
    field.average_mpi()
}

/// Returns Reynolds number base on kinetic energy
/// $$
/// Re = U*L / nu
/// U = \sqrt{(ux^2 + uy^2)}
/// $$
pub fn eval_re_mpi<A, T2, S>(
    ux: &mut FieldBaseMpi<A, A, T2, S, 2>,
    uy: &mut FieldBaseMpi<A, A, T2, S, 2>,
    field: &mut FieldBaseMpi<A, A, T2, S, 2>,
    nu: A,
    scale: &[A; 2],
) -> A
where
    A: FloatNum + crate::mpi::Equivalence + std::iter::Sum,
    Complex<A>: ScalarOperand,
    S: BaseSpace<A, 2, Physical = A, Spectral = T2>
        + BaseSpaceMpi<A, 2, Physical = A, Spectral = T2>,
    T2: Scalar + Div<A, Output = T2>,
{
    ux.backward_mpi();
    uy.backward_mpi();
    let ekin = &ux.v_y_pen.mapv(|x| x.powi(2)) + &uy.v_y_pen.mapv(|x| x.powi(2));
    field.v_y_pen.assign(&ekin.mapv(A::sqrt));
    let two = A::one() + A::one();
    field.v_y_pen *= two * scale[1] / nu;
    field.average_mpi()
}

/// Construct field f(x,y) = amp \* sin(pi\*m)cos(pi\*n)
pub fn apply_sin_cos<S, T2>(field: &mut Field2Mpi<T2, S>, amp: f64, m: f64, n: f64)
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T2>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = T2>,
{
    use std::f64::consts::PI;
    let nx = field.v.shape()[0];
    let ny = field.v.shape()[1];
    let x = &field.x[0];
    let y = &field.x[1];
    let x = &((x - x[0]) / (x[x.len() - 1] - x[0]));
    let y = &((y - y[0]) / (y[y.len() - 1] - y[0]));
    let arg_x = PI * m;
    let arg_y = PI * n;
    for i in 0..nx {
        for j in 0..ny {
            field.v[[i, j]] = amp * (arg_x * x[i]).sin() * (arg_y * y[j]).cos();
        }
    }
    field.forward();
}

/// Construct field f(x,y) = amp \* cos(pi\*m)sin(pi\*n)
pub fn apply_cos_sin<S, T2>(field: &mut Field2Mpi<T2, S>, amp: f64, m: f64, n: f64)
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T2>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = T2>,
{
    use std::f64::consts::PI;
    let nx = field.v.shape()[0];
    let ny = field.v.shape()[1];
    let x = &field.x[0];
    let y = &field.x[1];
    let x = &((x - x[0]) / (x[x.len() - 1] - x[0]));
    let y = &((y - y[0]) / (y[y.len() - 1] - y[0]));
    let arg_x = PI * m;
    let arg_y = PI * n;
    for i in 0..nx {
        for j in 0..ny {
            field.v[[i, j]] = amp * (arg_x * x[i]).cos() * (arg_y * y[j]).sin();
        }
    }
    field.forward();
}

/// Apply random disturbance [-c, c]
pub fn apply_random_disturbance<S, T2>(field: &mut Field2Mpi<T2, S>, c: f64)
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T2>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = T2>,
{
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    let nx = field.v.shape()[0];
    let ny = field.v.shape()[1];
    let rand: Array2<f64> = Array2::random((nx, ny), Uniform::new(-c, c));
    field.v.assign(&rand);
    field.forward();
}
