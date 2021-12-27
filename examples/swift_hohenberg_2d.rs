//! # 2D Swift hohenberg equation
//! ```
//! du/dt = [r - (lap + 1)^2] u - u^3
//! ```
//!
//! Run example:
//!
//! ```
//! cargo run --example swift_hohenberg_2d --release
//! ```
use ndarray::{s, Array1, Array2, ArrayViewMut1};
use num_complex::Complex;
use num_traits::identities::Zero;
use rustpde::bases::{fourier_c2c, fourier_r2c};
use rustpde::bases::{BaseC2c, BaseR2c};
use rustpde::field::{Field2, Space2};
use rustpde::hdf5::{write_scalar_to_hdf5, Result};
use rustpde::Integrate;

type Space2R2c = Space2<BaseC2c<f64>, BaseR2c<f64>>;

pub struct SwiftHohenberg2D {
    /// Field for derivatives and transforms
    pub field: Field2<Complex<f64>, Space2R2c>,
    /// Field variable
    pub theta: Field2<Complex<f64>, Space2R2c>,
    // Control parameter
    r: f64,
    /// Buffer
    rhs: Array2<Complex<f64>>,
    /// Time step size
    dt: f64,
    /// Time
    time: f64,
    /// Scale of phsical dimension \[scale_x, scale_y\]
    scale: [f64; 2],
    /// Store matrix for implicit update
    matl: Array2<f64>,
}

impl SwiftHohenberg2D {
    /// Initialize Swift Hohenberg Pde
    /// # Example
    /// ``` ignore
    /// use rustpde::integrate;
    /// let (nx, ny) = (512, 512);
    /// let length = 20.;
    /// let r = 0.35;
    /// let dt = 0.02;
    /// let mut pde = SwiftHohenberg2D::new(nx, ny, r, dt, length);
    /// integrate(&mut pde, 100., Some(10.));
    /// ```
    #[must_use]
    pub fn new(nx: usize, ny: usize, r: f64, dt: f64, length: f64) -> Self {
        use ndarray::Zip;
        //fields for derivatives
        let field = Field2::new(&Space2::new(&fourier_c2c(nx), &fourier_r2c(ny)));
        //fields variable
        let mut theta = Field2::new(&Space2::new(&fourier_c2c(nx), &fourier_r2c(ny)));
        // Self::init_cos(&mut theta, 1e-1, 10., 10.);
        Self::init_random(&mut theta, 1e-1);
        // Buffer array
        let rhs = Array2::zeros(field.vhat.raw_dim());
        // geometry scales
        let scale = [length, length];
        // build matrix for implicit update
        let mut matl = Array2::<f64>::zeros(theta.vhat.raw_dim());
        Zip::from(&mut matl)
            .and(&Self::wavenumber_matrix_x(&theta, &scale))
            .and(&Self::wavenumber_matrix_y(&theta, &scale))
            .for_each(|m, &x, &y| {
                *m = 1. - r * dt
                    + dt * (x.powi(2).re + y.powi(2).re + 1.) * (x.powi(2).re + y.powi(2).re + 1.);
            });
        Self {
            field,
            theta,
            r,
            rhs,
            dt,
            time: 0.,
            scale,
            matl,
        }
    }

    fn zero_rhs(&mut self) {
        for r in self.rhs.iter_mut() {
            *r = Complex::<f64>::zero();
        }
    }

    /// Two dimensional matrix with wavenumber elements.
    /// Constant in y.
    fn wavenumber_matrix_x(
        field: &Field2<Complex<f64>, Space2R2c>,
        scale: &[f64; 2],
    ) -> Array2<Complex<f64>> {
        let mut k_x = Array2::<Complex<f64>>::zeros(field.vhat.raw_dim());
        let k_x_lane = Self::_wavenumber_c2c(field.v.shape()[0]).mapv(|x| x / scale[0]);
        for mut lane in k_x.axis_iter_mut(ndarray::Axis(1)) {
            lane.assign(&k_x_lane);
        }
        k_x
    }

    /// Two dimensional matrix with wavenumber elements.
    /// Constant in x.
    fn wavenumber_matrix_y(
        field: &Field2<Complex<f64>, Space2R2c>,
        scale: &[f64; 2],
    ) -> Array2<Complex<f64>> {
        let mut k_y = Array2::<Complex<f64>>::zeros(field.vhat.raw_dim());
        let k_y_lane = Self::_wavenumber_r2c(field.v.shape()[1]).mapv(|x| x / scale[1]);
        for mut lane in k_y.axis_iter_mut(ndarray::Axis(0)) {
            lane.assign(&k_y_lane);
        }
        k_y
    }

    /// Return complex wavenumber vector for r2c transform (0, 1, 2, 3)
    #[allow(clippy::cast_precision_loss)]
    fn _wavenumber_r2c(n: usize) -> Array1<Complex<f64>> {
        let n2 = n / 2 + 1;
        let mut k: Array1<Complex<f64>> = Array1::zeros(n2);
        for (i, ki) in k.iter_mut().enumerate() {
            ki.im = i as f64;
        }
        k
    }

    /// Return complex wavenumber vector(0, 1, 2, -3, -2, -1)
    #[allow(clippy::cast_precision_loss)]
    fn _wavenumber_c2c(n: usize) -> Array1<Complex<f64>> {
        let n2 = (n - 1) / 2 + 1;
        let mut k: Array1<Complex<f64>> = Array1::zeros(n);
        for (i, ki) in k.iter_mut().take(n2).enumerate() {
            ki.im = i as f64;
        }
        for (i, ki) in k.iter_mut().rev().take(n / 2).enumerate() {
            ki.im = -1. * (i + 1) as f64;
        }
        k
    }

    /// Ensure that the elements of a vector are complex conjugate.
    /// First row (and last row for even input) must be complex conjugate,
    /// with respect to equal absolute wavenumbers, since data in physical space
    /// is purely real.
    fn enforce_hermitian_symmetry(input: &mut ArrayViewMut1<Complex<f64>>) {
        let n = input.len();
        let n2 = (n - 1) / 2 + 1;
        for i in 1..n2 {
            input[n - i].re = input[i].re;
            input[n - i].im = -1. * input[i].im;
        }
    }

    /// Dealias field (2/3 rule)
    /// Use:
    /// ```ignore
    /// Self::dealias(
    ///     &mut self.field,
    ///     &Self::_wavenumber_c2c(self.theta.v.shape()[0]).mapv(|x| x.im.abs() as usize),
    ///     &Self::_wavenumber_r2c(self.theta.v.shape()[1]).mapv(|x| x.im.abs() as usize),
    /// );
    /// ```
    #[allow(dead_code)]
    fn dealias(
        field: &mut Field2<Complex<f64>, Space2R2c>,
        k_x: &Array1<usize>,
        k_y: &Array1<usize>,
    ) {
        let zero = Complex::<f64>::zero();
        // x
        let cutoff_x = (field.v.shape()[0] / 2 + 1) * 2 / 3;
        for (i, k) in k_x.iter().enumerate() {
            if k > &cutoff_x {
                field.vhat.slice_mut(s![i, ..]).fill(zero);
            }
        }
        // y
        let cutoff_y = (field.v.shape()[1] / 2 + 1) * 2 / 3;
        for (i, k) in k_y.iter().enumerate() {
            if k > &cutoff_y {
                field.vhat.slice_mut(s![.., i]).fill(zero);
            }
        }
    }

    /// Initialize with random disturbance [-c, c]
    #[allow(dead_code)]
    fn init_random(field: &mut Field2<Complex<f64>, Space2R2c>, c: f64) {
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;
        let nx = field.v.shape()[0];
        let ny = field.v.shape()[1];
        let rand: Array2<f64> = Array2::random((nx, ny), Uniform::new(-c, c));
        field.v.assign(&rand);
        field.forward();
    }

    /// Initialize with cosines in x and y
    #[allow(dead_code)]
    fn init_cos(field: &mut Field2<Complex<f64>, Space2R2c>, c: f64, kx: f64, ky: f64) {
        let length = field.x[0][field.x[0].len() - 1] - field.x[0][0];
        let height = field.x[1][field.x[1].len() - 1] - field.x[1][0];
        for (i, xi) in field.x[0].iter().enumerate() {
            for (j, yi) in field.x[1].iter().enumerate() {
                field.v[[i, j]] = c
                    * (xi / length * kx * std::f64::consts::PI).cos()
                    * (yi / height * ky * std::f64::consts::PI).cos();
            }
        }
        field.forward();
    }

    // Return norm of complex array
    #[allow(clippy::cast_precision_loss)]
    fn norm_l2_c64(array: &Array2<Complex<f64>>) -> f64 {
        array
            .iter()
            .map(|x| x.re.powi(2) + x.im.powi(2))
            .sum::<f64>()
            .sqrt()
            / array.len() as f64
    }

    /// Write to file
    fn write(&mut self, filename: &str) {
        let result = self._write(filename);
        match result {
            Ok(_) => println!(" ==> {:?}", filename),
            Err(_) => println!("Error while writing file {:?}.", filename),
        }
    }

    fn _write(&mut self, filename: &str) -> Result<()> {
        use rustpde::field::WriteField;
        // Write field
        self.theta.backward();
        self.theta.write(&filename, Some("temp"));
        // Write scalars
        write_scalar_to_hdf5(&filename, "time", None, self.time)?;
        write_scalar_to_hdf5(&filename, "dt", None, self.dt)?;
        write_scalar_to_hdf5(&filename, "r", None, self.r)?;
        Ok(())
    }

    /// Update pde - diffusion explicit
    #[allow(dead_code)]
    fn update_explicit(&mut self) {
        self.zero_rhs();

        // Non-linear term: - theta^3
        self.field.vhat.assign(&self.theta.vhat);
        self.field.backward();
        self.field
            .v
            .assign(&(&self.field.v * &self.field.v * &self.field.v));
        self.field.forward();
        self.rhs -= &self.field.vhat;

        // Linear term: (r - (nab + 1)^2) * theta
        self.rhs += &(&self.theta.vhat * self.r);
        self.rhs -= &(self.theta.vhat);
        self.rhs -= &(self.theta.gradient([2, 0], Some(self.scale)) * 2.);
        self.rhs -= &(self.theta.gradient([0, 2], Some(self.scale)) * 2.);
        self.rhs -= &(self.theta.gradient([2, 2], Some(self.scale)) * 2.);
        self.rhs -= &(self.theta.gradient([4, 0], Some(self.scale)) * 1.);
        self.rhs -= &(self.theta.gradient([0, 4], Some(self.scale)) * 1.);

        // Update field
        self.theta.vhat += &(&self.rhs * self.dt);
        self.theta.vhat[[0, 0]] = Complex::<f64>::zero();
    }

    /// Update pde - diffusion implicit
    fn update_implicit(&mut self) {
        self.zero_rhs();

        // + old theta
        self.rhs += &self.theta.vhat;

        // - dt * theta^3
        self.field.vhat.assign(&self.theta.vhat);
        self.field.backward();
        self.field
            .v
            .assign(&(&self.field.v * &self.field.v * &self.field.v));
        self.field.forward();
        self.rhs -= &(&self.field.vhat * self.dt);

        // update field
        self.theta.vhat = &self.rhs / &self.matl;
        self.theta.vhat[[0, 0]] = Complex::<f64>::zero();

        // enfore hermitian symmetry on first row
        // (algorithm is otherwise unstable)
        Self::enforce_hermitian_symmetry(&mut self.theta.vhat.slice_mut(s!(.., 0)));
    }
}

impl Integrate for SwiftHohenberg2D {
    /// Update pde
    fn update(&mut self) {
        // Update solution
        self.update_implicit();
        // update time
        self.time += self.dt;
    }

    /// Receive current time
    fn get_time(&self) -> f64 {
        self.time
    }

    /// Get timestep
    fn get_dt(&self) -> f64 {
        self.dt
    }

    /// Callback function (can be used for i/o)
    fn callback(&mut self) {
        println!("Time = {:6.2e}", self.time);
        // Folder for files
        std::fs::create_dir_all("data").unwrap();
        // Write flow field
        let fname = format!("data/flow{:0>8.2}.h5", self.time);
        self.write(&fname);
        // Diagnostic
        self.theta.backward();
        println!("|F| = {:6.2e}", Self::norm_l2_c64(&self.theta.vhat));
    }

    /// Additional break criteria
    fn exit(&mut self) -> bool {
        // Check for nan
        (&self.theta.vhat).into_iter().any(|v| v.is_nan())
    }
}

fn main() {
    use rustpde::integrate;
    let (nx, ny) = (512, 512);
    let length = 20.;
    let r = 0.35;
    let dt = 0.02;
    let mut pde = SwiftHohenberg2D::new(nx, ny, r, dt, length);
    pde.callback();
    integrate(&mut pde, 1000., Some(10.));
}
