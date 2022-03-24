//! Storage for mean field
use crate::bases::{chebyshev, fourier_r2c};
use crate::bases::{BaseR2c, BaseR2r};
use crate::field::{Field2, Space2};
use crate::io::read_write_hdf5::read_from_hdf5;
use crate::io::traits::ReadWrite;
use crate::io::Result;
use funspace::BaseSpace;
use ndarray::{Axis, Ix2};
use num_complex::Complex;

type Space2R2c = Space2<BaseR2c<f64>, BaseR2r<f64>>;
type Space2R2r = Space2<BaseR2r<f64>, BaseR2r<f64>>;

/// `MeanFields` used for the linearized Navier-Stokes solver
#[derive(Clone)]
pub struct MeanFields<T, S> {
    /// Horizontal velocity field
    pub velx: Field2<T, S>,
    /// Vertical velocity field
    pub vely: Field2<T, S>,
    /// Temperature field
    pub temp: Field2<T, S>,
}

impl MeanFields<f64, Space2R2r> {
    /// Return `MeanFields` for confined Rayleigh-Benard convection
    pub fn new_rbc_confined(nx: usize, ny: usize) -> MeanFields<f64, Space2R2r> {
        // Allocate
        let velx = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));
        let vely = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));
        let mut temp = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));

        // Set linear temperature profile
        let y = &temp.x[1];
        let height = y[y.len() - 1] - y[0];
        let linear_profile = -(y - y[0]) / height + 0.5;
        for mut axis in temp.v.axis_iter_mut(Axis(0)) {
            axis.assign(&linear_profile);
        }
        temp.forward();

        // Return
        Self { velx, vely, temp }
    }

    /// Return field for Horizontal convection
    /// type temperature boundary conditions:
    ///
    /// T = sin(2*pi/L * x) at the bottom
    /// and T = T' = 0 at the top
    pub fn new_hc_confined(nx: usize, ny: usize) -> MeanFields<f64, Space2R2r> {
        /// Return y = a(x-xs)**2 + ys, where xs and
        /// ys (=0) are the coordinates of the parabola.
        ///
        /// The vertex with ys=0 and dydx=0 is at the
        /// right boundary and *a* is calculated
        /// from the value at the left boundary,
        fn _parabola(x: &ndarray::Array1<f64>, f_xl: f64) -> ndarray::Array1<f64> {
            let x_l = x[0];
            let x_r = x[x.len() - 1];
            let a = f_xl / (x_l - x_r).powi(2);
            x.mapv(|x| a * (x - x_r).powi(2))
        }

        let velx = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));
        let vely = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));
        let mut temp = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));

        // Set field
        let x = &temp.x[0];
        let y = &temp.x[1];
        let x0 = x[0];
        let length = x[x.len() - 1] - x[0];
        for (mut axis, xi) in temp.v.axis_iter_mut(Axis(0)).zip(x.iter()) {
            let f_x = -0.5 * (2. * std::f64::consts::PI * (xi - x0) / length).cos();
            let parabola = _parabola(y, f_x);
            axis.assign(&parabola);
        }

        // Transform
        temp.forward();
        temp.backward();

        // Return
        Self { velx, vely, temp }
    }

    /// Read meanfield from file
    /// # Panics
    /// 'bc' type not recognized
    pub fn read_from_confined(
        nx: usize,
        ny: usize,
        filename: &str,
        bc: Option<&str>,
    ) -> MeanFields<f64, Space2R2r> {
        use std::path::Path;
        let is_file = Path::new(filename).is_file();
        if is_file {
            // Allocate
            let velx = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));
            let vely = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));
            let temp = Field2::new(&Space2::new(&chebyshev(nx), &chebyshev(ny)));
            let mut meanfield = Self { velx, vely, temp };
            meanfield.read_unwrap(filename);
            meanfield
        } else {
            println!(
                "File {:?} does not exist. Use {:?} meanfield.",
                filename, bc
            );
            if let Some(bc) = bc {
                match bc {
                    "rbc" => Self::new_rbc_confined(nx, ny),
                    "hc" => Self::new_hc_confined(nx, ny),
                    _ => panic!("Boundary condition type {:?} not recognized!", bc),
                }
            } else {
                Self::new_rbc_confined(nx, ny)
            }
        }
    }
}

impl MeanFields<Complex<f64>, Space2R2c> {
    /// Return `MeanFields` for periodic Rayleigh-Benard convection
    /// # Panics
    /// 'bc' type not recognized
    pub fn new_rbc_periodic(nx: usize, ny: usize) -> MeanFields<Complex<f64>, Space2R2c> {
        // Allocate
        let velx = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
        let vely = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
        let mut temp = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));

        // Set linear temperature profile
        let y = &temp.x[1];
        let height = y[y.len() - 1] - y[0];
        let linear_profile = -(y - y[0]) / height + 0.5;
        for mut axis in temp.v.axis_iter_mut(Axis(0)) {
            axis.assign(&linear_profile);
        }
        temp.forward();

        // Return
        Self { velx, vely, temp }
    }

    /// Return field for Horizontal convection
    /// type temperature boundary conditions:
    ///
    /// T = sin(2*pi/L * x) at the bottom
    /// and T = T' = 0 at the top
    pub fn new_hc_periodic(nx: usize, ny: usize) -> MeanFields<Complex<f64>, Space2R2c> {
        /// Return y = a(x-xs)**2 + ys, where xs and
        /// ys (=0) are the coordinates of the parabola.
        ///
        /// The vertex with ys=0 and dydx=0 is at the
        /// right boundary and *a* is calculated
        /// from the value at the left boundary,
        fn _parabola(x: &ndarray::Array1<f64>, f_xl: f64) -> ndarray::Array1<f64> {
            let x_l = x[0];
            let x_r = x[x.len() - 1];
            let a = f_xl / (x_l - x_r).powi(2);
            x.mapv(|x| a * (x - x_r).powi(2))
        }

        let velx = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
        let vely = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
        let mut temp = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));

        // Set field
        let x = &temp.x[0];
        let y = &temp.x[1];
        let x0 = x[0];
        let length = x[x.len() - 1] - x[0];
        for (mut axis, xi) in temp.v.axis_iter_mut(Axis(0)).zip(x.iter()) {
            let f_x = -0.5 * (2. * std::f64::consts::PI * (xi - x0) / length).cos();
            let parabola = _parabola(y, f_x);
            axis.assign(&parabola);
        }

        // Transform
        temp.forward();
        temp.backward();

        // Return
        Self { velx, vely, temp }
    }

    /// Read meanfield from file
    /// # Panics
    /// If file does not exists and 'bc' type not recognized
    pub fn read_from_periodic(
        nx: usize,
        ny: usize,
        filename: &str,
        bc: Option<&str>,
    ) -> MeanFields<Complex<f64>, Space2R2c> {
        use std::path::Path;
        let is_file = Path::new(filename).is_file();
        if is_file {
            // Allocate
            println!("Read MeanField from: {:?}", filename);
            let velx = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
            let vely = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
            let temp = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
            let mut meanfield = Self { velx, vely, temp };
            meanfield.read_unwrap(filename);
            meanfield
        } else {
            println!(
                "File {:?} does not exist. Use {:?} meanfield.",
                filename, bc
            );
            if let Some(bc) = bc {
                match bc {
                    "rbc" => Self::new_rbc_periodic(nx, ny),
                    "hc" => Self::new_hc_periodic(nx, ny),
                    _ => panic!("Boundary condition type {:?} not recognized!", bc),
                }
            } else {
                Self::new_rbc_periodic(nx, ny)
            }
        }
    }
}

impl<T, S> MeanFields<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    Field2<T, S>: ReadWrite<T>,
{
    /// Read snapshot from file
    /// # Errors
    /// Failed to read
    pub fn read(&mut self, filename: &str) -> Result<()> {
        // Read arrays in physical space.
        // Spectral space shape can deviate, since meanfields
        // are defined as orthogonal spaces
        self.velx
            .v
            .assign(&read_from_hdf5::<f64, Ix2>(filename, "ux/v")?);
        self.vely
            .v
            .assign(&read_from_hdf5::<f64, Ix2>(filename, "uy/v")?);
        self.temp
            .v
            .assign(&read_from_hdf5::<f64, Ix2>(filename, "temp/v")?);
        // Add bc if present
        if let Ok(x) = read_from_hdf5::<f64, Ix2>(filename, "tempbc/v") {
            self.temp.v += &x;
        }
        self.velx.forward();
        self.vely.forward();
        self.temp.forward();
        Ok(())
    }

    /// Read snapshot from file, and handle error
    pub fn read_unwrap(&mut self, filename: &str) {
        match self.read(filename) {
            Ok(_) => {
                println!("Reading file {:?} was successfull.", filename);
            }
            Err(e) => eprintln!("Error while reading file {:?}. Error: {}", filename, e),
        }
    }

    /// Write snapshot to file
    /// # Errors
    /// Failed to write
    pub fn write(&mut self, filename: &str) -> Result<()> {
        self.velx.backward();
        self.vely.backward();
        self.temp.backward();
        self.velx.write(filename, "ux")?;
        self.vely.write(filename, "uy")?;
        self.temp.write(filename, "temp")?;
        Ok(())
    }

    /// Write snapshot to file, and handle error
    pub fn write_unwrap(&mut self, filename: &str) {
        match self.write(filename) {
            Ok(_) => (),
            Err(e) => eprintln!("Error while writing file {:?}. Error: {}", filename, e),
        }
    }
}

//
// impl MeanFields<f64, Space2R2r> {
//     /// Restart from file
//     pub fn read(&mut self, filename: &str) {
//         use ndarray::Array2;
//         // use rustpde::field::ReadField;
//         // Field
//         // self.temp.read(&filename, Some("temp"));
//         // self.velx.read(&filename, Some("ux"));
//         // self.vely.read(&filename, Some("uy"));
//         let velx: Array2<f64> = read_from_hdf5(filename, "ux/v").unwrap();
//         let vely: Array2<f64> = read_from_hdf5(filename, "uy/v").unwrap();
//         let temp: Array2<f64> = read_from_hdf5(filename, "temp/v").unwrap();
//         self.velx.v.assign(&velx);
//         self.vely.v.assign(&vely);
//         self.temp.v.assign(&temp);
//         self.velx.forward();
//         self.vely.forward();
//         self.temp.forward();
//     }
//
//     pub fn write(&mut self, filename: &str) -> Result<()> {
//         use rustpde::field::WriteField;
//
//         self.temp.backward();
//         self.velx.backward();
//         self.vely.backward();
//
//         self.velx.write(&filename, Some("ux"));
//         self.vely.write(&filename, Some("uy"));
//         self.temp.write(&filename, Some("temp"));
//
//         Ok(())
//     }
// }
//
// impl MeanFields<Complex<f64>, Space2R2c> {
//     /// Restart from file
//     pub fn read(&mut self, filename: &str) {
//         use ndarray::Array2;
//         // Field
//         let velx: Array2<f64> = read_from_hdf5(filename, "v", Some("ux")).unwrap();
//         let vely: Array2<f64> = read_from_hdf5(filename, "v", Some("uy")).unwrap();
//         let temp: Array2<f64> = read_from_hdf5(filename, "v", Some("temp")).unwrap();
//         self.velx.v.assign(&velx);
//         self.vely.v.assign(&vely);
//         self.temp.v.assign(&temp);
//         self.velx.forward();
//         self.vely.forward();
//         self.temp.forward();
//     }
//
//     pub fn write(&mut self, filename: &str) -> Result<()> {
//         use rustpde::field::WriteField;
//
//         self.temp.backward();
//         self.velx.backward();
//         self.vely.backward();
//
//         self.velx.write(&filename, Some("ux"));
//         self.vely.write(&filename, Some("uy"));
//         self.temp.write(&filename, Some("temp"));
//
//         Ok(())
//     }
// }
