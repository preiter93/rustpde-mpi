//! Collection of Boundary conditions
use crate::bases::{chebyshev, fourier_r2c};
use crate::bases::{BaseR2c, BaseR2r};
use crate::field::{Field2, Space2};
use ndarray::{Array1, Axis};
use num_complex::Complex;
use num_traits::Pow;
use std::f64::consts::PI;

type Space2R2r = Space2<BaseR2r<f64>, BaseR2r<f64>>;
type Space2R2c = Space2<BaseR2c<f64>, BaseR2r<f64>>;

/// Return field for rayleigh benard
/// type temperature boundary conditions:
///
/// T = 0.5 at the bottom and T = -0.5
/// at the top
pub fn bc_rbc(nx: usize, ny: usize) -> Field2<f64, Space2R2r> {
    let x_base = chebyshev(nx);
    let y_base = chebyshev(ny);
    let space = Space2::new(&x_base, &y_base);
    let mut fieldbc = Field2::new(&space);

    let x = &fieldbc.x[1];
    let (x1, x2) = (x[0], x[x.len() - 1]);
    let (y1, y2) = (0.5, -0.5);
    let m = (y2 - y1) / (x2 - x1);
    let n = (y1 * x2 - y2 * x1) / (x2 - x1);
    let profile = m * x + n;
    for mut axis in fieldbc.v.axis_iter_mut(Axis(0)) {
        axis.assign(&profile);
    }
    fieldbc.forward();
    fieldbc.backward();
    fieldbc
}

/// Return field for rayleigh benard
/// type pressure boundary conditions:
pub fn pres_bc_rbc(nx: usize, ny: usize) -> Field2<f64, Space2R2r> {
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
    let space = Space2::new(&x_base, &y_base);
    let mut fieldbc = Field2::new(&space);

    let y = &fieldbc.x[1];
    let (a, b) = parabola_coeff(0.5, -0.5, y);
    let parabola = a * y.mapv(|y| y.pow(2)) + b * y;
    for mut axis in fieldbc.v.axis_iter_mut(Axis(0)) {
        axis.assign(&parabola);
    }

    // Transform
    fieldbc.forward();
    fieldbc.backward();

    //println!("{:?}", fieldbc.to_ortho());
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
pub fn bc_zero(nx: usize, ny: usize, k: f64) -> Field2<f64, Space2R2r> {
    let x_base = chebyshev(nx);
    let y_base = chebyshev(ny);
    let space = Space2::new(&x_base, &y_base);
    let mut fieldbc = Field2::new(&space);

    // Sidewall temp function
    let profile = transfer_function(&fieldbc.x[1], 0.5, 0., -0.5, k);
    for mut axis in fieldbc.v.axis_iter_mut(Axis(0)) {
        axis.assign(&profile);
    }
    fieldbc.forward();
    fieldbc.backward();
    fieldbc
}

/// Return field for Horizontal convection
/// type temperature boundary conditions:
///
/// T = sin(2*pi/L * x) at the bottom
/// and T = T' = 0 at the top
pub fn bc_hc(nx: usize, ny: usize) -> Field2<f64, Space2R2r> {
    /// Return y = a(x-xs)**2 + ys, where xs and
    /// ys (=0) are the coordinates of the parabola.
    ///
    /// The vertex with ys=0 and dydx=0 is at the
    /// right boundary and *a* is calculated
    /// from the value at the left boundary,
    fn _parabola(x: &Array1<f64>, f_xl: f64) -> Array1<f64> {
        let x_l = x[0];
        let x_r = x[x.len() - 1];
        let a = f_xl / (x_l - x_r).powi(2);
        x.mapv(|x| a * (x - x_r).powi(2))
    }

    // Create base and field
    let x_base = chebyshev(nx);
    let y_base = chebyshev(ny);
    let space = Space2::new(&x_base, &y_base);
    let mut fieldbc = Field2::new(&space);

    let x = &fieldbc.x[0];
    let y = &fieldbc.x[1];
    let x0 = x[0];
    let length = x[x.len() - 1] - x[0];
    for (mut axis, xi) in fieldbc.v.axis_iter_mut(Axis(0)).zip(x.iter()) {
        let f_x = -0.5 * (2. * PI * (xi - x0) / length).cos();
        let parabola = _parabola(y, f_x);
        axis.assign(&parabola);
    }

    // Transform
    fieldbc.forward();
    fieldbc.backward();

    fieldbc
}

/// Return field for rayleigh benard
/// type temperature boundary conditions:
///
/// T = 0.5 at the bottom and T = -0.5
/// at the top
pub fn bc_rbc_periodic(nx: usize, ny: usize) -> Field2<Complex<f64>, Space2R2c> {
    let x_base = fourier_r2c(nx);
    let y_base = chebyshev(ny);
    let space = Space2::new(&x_base, &y_base);
    let mut fieldbc = Field2::new(&space);

    let x = &fieldbc.x[1];
    let (x1, x2) = (x[0], x[x.len() - 1]);
    let (y1, y2) = (0.5, -0.5);
    let m = (y2 - y1) / (x2 - x1);
    let n = (y1 * x2 - y2 * x1) / (x2 - x1);
    let profile = m * x + n;
    for mut axis in fieldbc.v.axis_iter_mut(Axis(0)) {
        axis.assign(&profile);
    }
    fieldbc.forward();
    fieldbc.backward();
    fieldbc
}

/// Return field for Horizontal convection
/// type temperature boundary conditions:
///
/// T = sin(2*pi/L * x) at the bottom
/// and T = T' = 0 at the top
pub fn bc_hc_periodic(nx: usize, ny: usize) -> Field2<Complex<f64>, Space2R2c> {
    /// Return y = a(x-xs)**2 + ys, where xs and
    /// ys (=0) are the coordinates of the parabola.
    ///
    /// The vertex with ys=0 and dydx=0 is at the
    /// right boundary and *a* is calculated
    /// from the value at the left boundary,
    fn _parabola(x: &Array1<f64>, f_xl: f64) -> Array1<f64> {
        let x_l = x[0];
        let x_r = x[x.len() - 1];
        let a = f_xl / (x_l - x_r).powi(2);
        x.mapv(|x| a * (x - x_r).powi(2))
    }

    // Create base and field
    let x_base = fourier_r2c(nx);
    let y_base = chebyshev(ny);
    let space = Space2::new(&x_base, &y_base);
    let mut fieldbc = Field2::new(&space);

    let x = &fieldbc.x[0];
    let y = &fieldbc.x[1];
    let x0 = x[0];
    let length = x[x.len() - 1] - x[0];
    for (mut axis, xi) in fieldbc.v.axis_iter_mut(Axis(0)).zip(x.iter()) {
        let f_x = -0.5 * (2. * PI * (xi - x0) / length).cos();
        let parabola = _parabola(y, f_x);
        axis.assign(&parabola);
    }

    // Transform
    fieldbc.forward();
    fieldbc.backward();

    fieldbc
}

/// Return field for rayleigh benard
/// type pressure boundary conditions:
pub fn pres_bc_rbc_periodic(nx: usize, ny: usize) -> Field2<Complex<f64>, Space2R2c> {
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
    let space = Space2::new(&x_base, &y_base);
    let mut fieldbc = Field2::new(&space);

    let y = &fieldbc.x[1];
    let (a, b) = parabola_coeff(0.5, -0.5, y);
    let parabola = a * y.mapv(|y| y.pow(2)) + b * y;
    for mut axis in fieldbc.v.axis_iter_mut(Axis(0)) {
        axis.assign(&parabola);
    }

    // Transform
    fieldbc.forward();
    fieldbc.backward();

    //println!("{:?}", fieldbc.to_ortho());
    fieldbc
}

/// Return bc field with only zeros
pub fn pres_bc_empty(nx: usize, ny: usize) -> Field2<f64, Space2R2r> {
    let x_base = chebyshev(nx);
    let y_base = chebyshev(ny);
    let space = Space2::new(&x_base, &y_base);
    let mut fieldbc = Field2::new(&space);
    fieldbc.v.fill(0.);
    fieldbc.forward();
    fieldbc
}

/// Return bc field with only zeros
pub fn pres_bc_empty_periodic(nx: usize, ny: usize) -> Field2<Complex<f64>, Space2R2c> {
    let x_base = fourier_r2c(nx);
    let y_base = chebyshev(ny);
    let space = Space2::new(&x_base, &y_base);
    let mut fieldbc = Field2::new(&space);
    fieldbc.v.fill(0.);
    fieldbc.forward();
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
