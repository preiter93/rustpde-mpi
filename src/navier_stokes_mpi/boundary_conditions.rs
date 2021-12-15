//! Collection of Boundary conditions
use crate::bases::{cheb_dirichlet_bc, chebyshev, fourier_r2c};
use crate::bases::{BaseR2c, BaseR2r};
use crate::field_mpi::{Field2Mpi, Space2Mpi};
use crate::mpi::Universe;
use ndarray::{s, Array1, Array2};
use num_complex::Complex;

type Space2R2r<'a> = Space2Mpi<'a, BaseR2r<f64>, BaseR2r<f64>>;
type Space2R2c<'a> = Space2Mpi<'a, BaseR2c<f64>, BaseR2r<f64>>;

/// Return field for rayleigh benard
/// type temperature boundary conditions:
///
/// T = 0.5 at the bottom and T = -0.5
/// at the top
pub fn bc_rbc<'a>(nx: usize, ny: usize, universe: &'a Universe) -> Field2Mpi<f64, Space2R2r<'a>> {
    use crate::bases::Transform;
    // Create base and field
    let mut x_base = chebyshev(nx);
    let y_base = cheb_dirichlet_bc(ny);
    let mut field_bc = Field2Mpi::new(&Space2Mpi::new(&x_base, &y_base, universe));
    let mut field_ortho = Field2Mpi::new(&Space2Mpi::new(&chebyshev(nx), &chebyshev(ny), universe));

    // Set boundary condition along axis
    let mut bc = field_bc.vhat.to_owned();
    bc.slice_mut(s![.., 0]).fill(0.5);
    bc.slice_mut(s![.., 1]).fill(-0.5);
    x_base.forward_inplace(&bc, &mut field_bc.vhat, 0);
    field_bc.backward();
    field_bc.forward();

    // BC base to orthogonal base
    field_ortho.vhat.assign(&field_bc.to_ortho());
    field_ortho.backward();
    field_ortho.scatter_physical();
    field_ortho.scatter_spectral();
    field_ortho
}

/// Return field for rayleigh benard
/// type pressure boundary conditions:
pub fn pres_bc_rbc<'a>(
    nx: usize,
    ny: usize,
    universe: &'a Universe,
) -> Field2Mpi<f64, Space2R2r<'a>> {
    use ndarray::Axis;
    use num_traits::Pow;

    /// Return a, b of a*x**2 + b*x
    /// from derivatives at the boundaries
    fn parabola_coeff(df_l: f64, df_r: f64, x: &Array1<f64>) -> (f64, f64) {
        let x_l = x[0];
        let x_r = x[x.len() - 1];
        let a = 0.5 * (df_r - df_l) / (x_r - x_l);
        let b = df_l - 2. * a * x_l;
        (a, b)
    }

    // Create base and field
    let x_base = chebyshev(nx);
    let y_base = chebyshev(ny);
    let space = Space2Mpi::new(&x_base, &y_base, universe);
    let mut fieldbc = Field2Mpi::new(&space);

    let y = &fieldbc.x[1];
    let (a, b) = parabola_coeff(0.5, -0.5, y);
    let parabola = a * y.mapv(|y| y.pow(2)) + b * y;
    for mut axis in fieldbc.v.axis_iter_mut(Axis(0)) {
        axis.assign(&parabola);
    }

    // Transform
    fieldbc.forward();
    fieldbc.backward();
    fieldbc.scatter_physical();
    fieldbc.scatter_spectral();

    fieldbc
}

/// Return field for zero sidewall boundary
/// condition with smooth transfer function
/// to T = 0.5 at the bottom and T = -0.5
/// at the top
///
/// # Arguments
///
/// * `k` - Transition parameter (larger means smoother)
pub fn bc_zero<'a>(
    nx: usize,
    ny: usize,
    k: f64,
    universe: &'a Universe,
) -> Field2Mpi<f64, Space2R2r<'a>> {
    use crate::bases::Transform;
    // Create base and field
    let x_base = cheb_dirichlet_bc(ny);
    let mut y_base = chebyshev(nx);
    let mut field_bc = Field2Mpi::new(&Space2Mpi::new(&x_base, &y_base, universe));
    let mut field_ortho = Field2Mpi::new(&Space2Mpi::new(&chebyshev(nx), &chebyshev(ny), universe));
    // Set boundary condition along axis
    let transfer = transfer_function(&field_bc.x[1], 0.5, 0., -0.5, k);
    let mut bc = field_bc.vhat.to_owned();
    bc.slice_mut(s![0, ..]).assign(&transfer);
    bc.slice_mut(s![1, ..]).assign(&transfer);
    y_base.forward_inplace(&bc, &mut field_bc.vhat, 1);
    field_bc.backward();
    field_bc.forward();

    // BC base to orthogonal base
    field_ortho.vhat.assign(&field_bc.to_ortho());
    field_ortho.backward();
    field_ortho.scatter_physical();
    field_ortho.scatter_spectral();
    field_ortho
}

/// Return field for rayleigh benard
/// type temperature boundary conditions:
///
/// T = 0.5 at the bottom and T = -0.5
/// at the top
pub fn bc_rbc_periodic<'a>(
    nx: usize,
    ny: usize,
    universe: &'a Universe,
) -> Field2Mpi<Complex<f64>, Space2R2c<'a>> {
    use crate::bases::Transform;
    // Create base and field
    let mut x_base = fourier_r2c(nx);
    let y_base = cheb_dirichlet_bc(ny);

    let mut field_bc = Field2Mpi::new(&Space2Mpi::new(&x_base, &y_base, universe));
    let mut field_ortho =
        Field2Mpi::new(&Space2Mpi::new(&fourier_r2c(nx), &chebyshev(ny), universe));

    // Set boundary condition along axis
    let mut bc = Array2::<f64>::zeros((nx, 2));
    bc.slice_mut(s![.., 0]).fill(0.5);
    bc.slice_mut(s![.., 1]).fill(-0.5);
    x_base.forward_inplace(&bc, &mut field_bc.vhat, 0);
    field_bc.backward();
    field_bc.forward();

    // BC base to orthogonal base
    field_ortho.vhat.assign(&field_bc.to_ortho());
    field_ortho.backward();
    field_ortho.scatter_physical();
    field_ortho.scatter_spectral();
    field_ortho
}

/// Return field for rayleigh benard
/// type pressure boundary conditions:
pub fn pres_bc_rbc_periodic<'a>(
    nx: usize,
    ny: usize,
    universe: &'a Universe,
) -> Field2Mpi<Complex<f64>, Space2R2c<'a>> {
    use ndarray::Axis;
    use num_traits::Pow;

    /// Return a, b of a*x**2 + b*x
    /// from derivatives at the boundaries
    fn parabola_coeff(df_l: f64, df_r: f64, x: &Array1<f64>) -> (f64, f64) {
        let x_l = x[0];
        let x_r = x[x.len() - 1];
        let a = 0.5 * (df_r - df_l) / (x_r - x_l);
        let b = df_l - 2. * a * x_l;
        (a, b)
    }

    // Create base and field
    let x_base = fourier_r2c(nx);
    let y_base = chebyshev(ny);
    let space = Space2Mpi::new(&x_base, &y_base, universe);
    let mut fieldbc = Field2Mpi::new(&space);

    let y = &fieldbc.x[1];
    let (a, b) = parabola_coeff(0.5, -0.5, y);
    let parabola = a * y.mapv(|y| y.pow(2)) + b * y;
    for mut axis in fieldbc.v.axis_iter_mut(Axis(0)) {
        axis.assign(&parabola);
    }

    // Transform
    fieldbc.forward();
    fieldbc.backward();
    fieldbc.scatter_physical();
    fieldbc.scatter_spectral();

    fieldbc
}

/// Transfer function for zero sidewall boundary condition
pub fn transfer_function(x: &Array1<f64>, v_l: f64, v_m: f64, v_r: f64, k: f64) -> Array1<f64> {
    let mut result = Array1::zeros(x.raw_dim());
    let length = x[x.len() - 1] - x[0];
    for (i, xi) in x.iter().enumerate() {
        let xs = xi * 2. / length;
        if xs < 0. {
            result[i] = -1.0 * k * xs / (k + xs + 1.) * (v_l - v_m) + v_m;
        } else {
            result[i] = 1.0 * k * xs / (k - xs + 1.) * (v_r - v_m) + v_m;
        }
    }
    result
}
