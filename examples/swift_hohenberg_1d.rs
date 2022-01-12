//! # 1D Swift hohenberg equation
//! ```
//! du/dt = [r - (lap + 1)^2] u - u^3
//! ```
//!
//! Run example:
//!
//! ```
//! cargo run --example swift_hohenberg_1d --release
//! ```
use ndarray::Array1;
use num_complex::Complex;
use num_traits::identities::Zero;
use rustpde::bases::fourier_r2c;
use rustpde::bases::BaseR2c;
use rustpde::field::{Field1, Space1};
use rustpde::hdf5::{write_scalar_to_hdf5, Result};
use rustpde::Integrate;

type Space1R2c = Space1<BaseR2c<f64>>;

pub struct SwiftHohenberg1D {
    /// Field for derivatives and transforms
    pub field: Field1<Complex<f64>, Space1R2c>,
    /// Field variable
    pub theta: Field1<Complex<f64>, Space1R2c>,
    // Control parameter
    r: f64,
    /// Buffer
    rhs: Array1<Complex<f64>>,
    /// Time step size
    dt: f64,
    /// Time
    time: f64,
    /// Scale of phsical dimensions
    scale: [f64; 1],
}

impl SwiftHohenberg1D {
    pub fn new(nx: usize, r: f64, dt: f64, length: f64) -> Self {
        //fields for derivatives
        let field = Field1::new(&Space1::new(&fourier_r2c(nx)));
        //fields variable
        let mut theta = Field1::new(&Space1::new(&fourier_r2c(nx)));
        Self::init_cos(&mut theta, 1e-5);
        // Buffer array
        let rhs = Array1::zeros(field.vhat.raw_dim());
        // geometry scales
        let scale = [length];
        // Return
        Self {
            field,
            theta,
            r,
            rhs,
            dt,
            time: 0.,
            scale,
        }
    }

    fn wavenumber(&self) -> Array1<Complex<f64>> {
        let n = self.theta.v.len();
        Self::_wavenumber(n).mapv(|x| x / self.scale[0])
    }
    /// Return complex wavenumber vector for r2c transform (0, 1, 2, 3)
    fn _wavenumber(n: usize) -> Array1<Complex<f64>> {
        let n2 = n / 2 + 1;
        let mut k: Array1<f64> = Array1::zeros(n2);
        for (i, ki) in Array1::range(0., n2 as f64, 1.)
            .iter()
            .zip(k.slice_mut(ndarray::s![..n2]))
        {
            *ki = *i as f64;
        }
        k.mapv(|x| Complex::new(0., x))
    }

    fn zero_rhs(&mut self) {
        for r in self.rhs.iter_mut() {
            *r = Complex::<f64>::zero();
        }
    }

    /// Dealias field (2/3 rule)
    fn dealias(field: &mut Field1<Complex<f64>, Space1R2c>) {
        use ndarray::s;
        let zero = Complex::<f64>::zero();
        let n_x: usize = field.vhat.shape()[0] * 2 / 3;
        field.vhat.slice_mut(s![n_x..]).fill(zero);
    }

    /// Apply random disturbance [-c, c]
    fn _init_random(field: &mut Field1<Complex<f64>, Space1R2c>, c: f64) {
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;
        let nx = field.v.shape()[0];
        let rand: Array1<f64> = Array1::random(nx, Uniform::new(-c, c));
        field.v.assign(&rand);
        field.forward();
    }

    fn init_cos(field: &mut Field1<Complex<f64>, Space1R2c>, c: f64) {
        let length = field.x[0][field.x[0].len() - 1] - field.x[0][0];
        for (i, xi) in field.x[0].iter().enumerate() {
            field.v[i] = c * (xi / length * 2. * std::f64::consts::PI).cos();
        }
        field.forward();
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
    fn _update_explicit(&mut self) {
        self.zero_rhs();
        self.field.vhat.assign(&self.theta.vhat);

        // Non-linear term: - theta^3
        self.field.backward();
        self.field
            .v
            .assign(&(&self.field.v * &self.field.v * &self.field.v));
        self.field.forward();
        Self::dealias(&mut self.field);
        self.rhs -= &self.field.vhat;

        // Linear term: (r - (nab + 1)^2) * theta
        self.rhs += &(&self.theta.vhat * (self.r - 1.));
        self.rhs -= &(self.theta.gradient([2], Some(self.scale)) * 2.);
        self.rhs -= &(self.theta.gradient([4], Some(self.scale)) * 1.);

        // Update field
        self.theta.vhat += &(&self.rhs * self.dt);
        self.theta.vhat[0] = Complex::<f64>::zero();

        // update time
        self.time += self.dt;
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
        Self::dealias(&mut self.field);
        self.rhs -= &(&self.field.vhat * self.dt);

        // build implicit matrices
        let matl = self
            .wavenumber()
            .mapv(|x| (2. * x.powi(2).re + x.powi(4).re - (self.r - 1.)) * self.dt + 1.);

        // update field
        self.theta.vhat = &self.rhs / &matl;

        // update time
        self.time += self.dt;
    }
}

impl Integrate for SwiftHohenberg1D {
    /// Update pde
    fn update(&mut self) {
        self.update_implicit();
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
        println!("Time {:?}", self.time);
        // Folder for files
        std::fs::create_dir_all("data").unwrap();
        // Write flow field
        let fname = format!("data/flow{:0>8.2}.h5", self.time);
        self.write(&fname);
    }

    /// Additional break criteria
    fn exit(&mut self) -> bool {
        // Check for nan
        (&self.theta.vhat).into_iter().any(|v| v.is_nan())
    }
}

fn main() {
    use rustpde::integrate;
    let nx = 128;
    let length = 10.;
    let r = 0.2;
    let dt = 0.01;
    let mut pde = SwiftHohenberg1D::new(nx, r, dt, length);
    integrate(&mut pde, 100., Some(5.));
}
