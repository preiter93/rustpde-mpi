//! `ReadWrite` trait
use super::read_write_hdf5::{read_from_hdf5, write_to_hdf5};
use super::read_write_hdf5::{read_from_hdf5_complex, write_to_hdf5_complex};
use super::Result;
use hdf5::Error;
use ndarray::{ArrayBase, Data, DataMut, Dimension};
use num_complex::Complex;

/// Read and write field (hdf5)
pub trait ReadWrite<A> {
    /// Read field data from hdf5 file
    ///
    /// # Errors
    /// Can't read file
    fn read(&mut self, filename: &str, varname: &str) -> Result<()>;
    /// Read field data from hdf5 file and handle result
    fn read_unwrap(&mut self, filename: &str, varname: &str);
    /// Write field data from hdf5 file
    ///
    /// # Errors
    /// Can't write file
    fn write(&self, filename: &str, varname: &str) -> Result<()>;
    /// Write field data from hdf5 file and handle result
    fn write_unwrap(&self, filename: &str, varname: &str);
}

/// Implement on real type arrays
macro_rules! impl_read_write_real {
    ($a: ty) => {
        impl<S, D> ReadWrite<$a> for ArrayBase<S, D>
        where
            S: Data<Elem = $a> + DataMut,
            D: Dimension,
        {
            fn read(&mut self, filename: &str, varname: &str) -> Result<()> {
                let data = read_from_hdf5::<$a, D>(filename, varname)?;
                if data.shape() == self.shape() {
                    self.assign(&data);
                    Ok(())
                } else {
                    Err(Error::Internal("Shape mismatch while reading.".to_owned()))
                }
            }

            fn read_unwrap(&mut self, filename: &str, varname: &str) {
                match self.read(filename, varname) {
                    Ok(_) => {
                        println!("Reading file {:?} was successfull.", filename);
                    }
                    Err(e) => eprintln!("Error while reading file {:?}. Error: {}", filename, e),
                }
            }

            fn write(&self, filename: &str, varname: &str) -> Result<()> {
                write_to_hdf5(filename, varname, &self)?;
                Ok(())
            }

            fn write_unwrap(&self, filename: &str, varname: &str) {
                match self.write(filename, varname) {
                    Ok(_) => (),
                    Err(e) => eprintln!("Error while writing file {:?}. Error: {}", filename, e),
                }
            }
        }
    };
}

impl_read_write_real!(f64);

/// Implement on complex type arrays
macro_rules! impl_read_write_complex {
    ($a: ty) => {
        impl<S, D> ReadWrite<Complex<$a>> for ArrayBase<S, D>
        where
            S: Data<Elem = Complex<$a>> + DataMut,
            D: Dimension,
        {
            fn read(&mut self, filename: &str, varname: &str) -> Result<()> {
                let data = read_from_hdf5_complex::<$a, D>(filename, varname)?;
                if data.shape() == self.shape() {
                    self.assign(&data);
                    Ok(())
                } else {
                    Err(Error::Internal("Shape mismatch while reading.".to_owned()))
                }
            }

            fn read_unwrap(&mut self, filename: &str, varname: &str) {
                match self.read(filename, varname) {
                    Ok(_) => {
                        println!("Reading file {:?} was successfull.", filename);
                    }
                    Err(e) => eprintln!("Error while reading file {:?}. Error: {}", filename, e),
                }
            }

            fn write(&self, filename: &str, varname: &str) -> Result<()> {
                write_to_hdf5_complex(filename, varname, &self)?;
                Ok(())
            }

            fn write_unwrap(&self, filename: &str, varname: &str) {
                match self.write(filename, varname) {
                    Ok(_) => (),
                    Err(e) => eprintln!("Error while writing file {:?}. Error: {}", filename, e),
                }
            }
        }
    };
}

impl_read_write_complex!(f64);
