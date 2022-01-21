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

    /// Read meanfield from file
    pub fn read_from_confined(nx: usize, ny: usize, filename: &str) -> MeanFields<f64, Space2R2r> {
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
            println!("File {:?} does not exist. Use RBC meanfield.", filename);
            Self::new_rbc_confined(nx, ny)
        }
    }
}

impl MeanFields<Complex<f64>, Space2R2c> {
    /// Return `MeanFields` for periodic Rayleigh-Benard convection
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

    /// Read meanfield from file
    pub fn read_from_periodic(
        nx: usize,
        ny: usize,
        filename: &str,
    ) -> MeanFields<Complex<f64>, Space2R2c> {
        use std::path::Path;
        let is_file = Path::new(filename).is_file();
        if is_file {
            // Allocate
            let velx = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
            let vely = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
            let temp = Field2::new(&Space2::new(&fourier_r2c(nx), &chebyshev(ny)));
            let mut meanfield = Self { velx, vely, temp };
            meanfield.read_unwrap(filename);
            meanfield
        } else {
            println!("File {:?} does not exist. Use RBC meanfield.", filename);
            Self::new_rbc_periodic(nx, ny)
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
            .assign(&read_from_hdf5::<f64, Ix2>(&filename, "ux/v")?);
        self.vely
            .v
            .assign(&read_from_hdf5::<f64, Ix2>(&filename, "ux/v")?);
        self.temp
            .v
            .assign(&read_from_hdf5::<f64, Ix2>(&filename, "temp/v")?);
        // Add bc if present
        if let Ok(x) = read_from_hdf5::<f64, Ix2>(&filename, "tempbc/v") {
            self.temp.v += &x
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
        self.velx.write(&filename, "ux")?;
        self.vely.write(&filename, "uy")?;
        self.temp.write(&filename, "temp")?;
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
