//! Implement io routines for `FieldBaseMpi`
//!
//! This part is currently under test.
//! Writing with MPI can be slow at times. Reason unknown...
#![cfg(feature = "mpio")]
use super::{BaseSpace, BaseSpaceMpi, Field2Mpi};
use crate::io::read_write_hdf5::write_to_hdf5;
use crate::io::read_write_mpi_hdf5::{read_mpi, write_mpi};
use crate::io::read_write_mpi_hdf5::{read_mpi_complex, write_mpi_complex};
use crate::io::traits::ReadWrite;
use crate::io::Result;
use num_complex::Complex;

macro_rules! impl_read_write_mpi_fieldmpi2 {
    ($s: ty, $read_s: ident , $write_s: ident) => {
        impl<S> ReadWrite<$s> for Field2Mpi<$s, S>
        where
            S: BaseSpace<f64, 2, Physical = f64, Spectral = $s> + BaseSpaceMpi<f64, 2>,
        {
            /// Read file succesively from all processors.
            /// Possibly slower than the `read` method, but does not allocate a global.
            ///
            /// # Errors
            /// Can't read file
            fn read(&mut self, filename: &str, dsetname: &str) -> Result<()> {
                // Get mpi decomp and local slice
                let shape = &self.space.shape_spectral();
                let dcp = &self.space.get_decomp_from_global_shape(shape).x_pencil;
                let slice = ndarray::s![dcp.st[0]..=dcp.en[0], dcp.st[1]..=dcp.en[1]];
                // Read in parallel
                let data: ndarray::Array2<$s> = $read_s(
                    &self.universe(),
                    filename,
                    &format!("{}/vhat", dsetname),
                    slice,
                )?;
                self.vhat_x_pen.assign(&data);
                self.backward_mpi();
                Ok(())
            }

            /// Unwrap `read_mpi`
            fn read_unwrap(&mut self, filename: &str, dsetname: &str) {
                match self.read(filename, dsetname) {
                    Ok(_) => {
                        println!("Reading file {:?} was successfull.", filename);
                    }
                    Err(e) => eprintln!("Error while reading file {:?}. Error: {}", filename, e),
                }
            }

            /// Write file succesively from all processors.
            /// Possibly slower than the `write` method, but does not allocate a global.
            ///
            /// # Errors
            /// Can't write file
            fn write(&self, filename: &str, varname: &str) -> Result<()> {
                // Geometry is all globally on root
                if self.nrank() == 0 {
                    write_to_hdf5(filename, &format!("{}/x", varname), &self.x[0])?;
                    write_to_hdf5(filename, &format!("{}/dx", varname), &self.x[0])?;
                    write_to_hdf5(filename, &format!("{}/y", varname), &self.x[1])?;
                    write_to_hdf5(filename, &format!("{}/dy", varname), &self.x[1])?;
                }

                // Physical decomp
                let shape_p = &self.space.shape_physical();
                let dcp_p = &self.space.get_decomp_from_global_shape(shape_p).y_pencil;
                let slice_p = ndarray::s![dcp_p.st[0]..=dcp_p.en[0], dcp_p.st[1]..=dcp_p.en[1]];

                // Spectral decomp
                let shape_s = &self.space.shape_spectral();
                let dcp_s = &self.space.get_decomp_from_global_shape(shape_s).x_pencil;
                let slice_s = ndarray::s![dcp_s.st[0]..=dcp_s.en[0], dcp_s.st[1]..=dcp_s.en[1]];

                // Write in parallel
                write_mpi(
                    &self.universe(),
                    filename,
                    &format!("{}/v", varname),
                    &self.v_y_pen,
                    slice_p,
                    shape_p,
                )?;
                $write_s(
                    &self.universe(),
                    filename,
                    &format!("{}/vhat", varname),
                    &self.vhat_x_pen,
                    slice_s,
                    shape_s,
                )?;
                Ok(())
            }

            /// Unwrap `write_mpi`
            fn write_unwrap(&self, filename: &str, varname: &str) {
                match self.write(filename, varname) {
                    Ok(_) => (),
                    Err(e) => eprintln!("Error while writing file {:?}. Error: {}", filename, e),
                }
            }
        }
    };
}

impl_read_write_mpi_fieldmpi2!(f64, read_mpi, write_mpi);
impl_read_write_mpi_fieldmpi2!(Complex<f64>, read_mpi_complex, write_mpi_complex);
