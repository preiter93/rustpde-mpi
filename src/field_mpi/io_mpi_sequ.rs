//! Implement io routines for `FieldBaseMpi`
//!
//! Here writing is done one processor after
//! the other, since hdf5 mpi config does not work. Though
//! this might be slower, no global array as to be allocated,
//! for serial read write.
//!
//! This module is not active at the moment, it got replaced
//! by true parallel writing in the`io_mpi` module
use super::{BaseSpace, BaseSpaceMpi, Field2Mpi};
use crate::io::read_write_hdf5::write_to_hdf5;
use crate::io::read_write_slice_hdf5::{read_slice, write_slice};
use crate::io::read_write_slice_hdf5::{read_slice_complex, write_slice_complex};
use crate::io::Result;
use funspace::mpi::CommunicatorCollectives;
use num_complex::Complex;

macro_rules! impl_read_write_mpi_fieldmpi2 {
    ($s: ty, $read_s: ident , $write_s: ident) => {
        impl<S> Field2Mpi<$s, S>
        where
            S: BaseSpace<f64, 2, Physical = f64, Spectral = $s> + BaseSpaceMpi<f64, 2>,
        {
            /// Read file succesively from all processors.
            /// Possibly slower than the `read` method, but does not allocate a global.
            ///
            /// # Errors
            /// Can't read file
            pub fn read_mpi(&mut self, filename: &str, dsetname: &str) -> Result<()> {
                // Get mpi decomp and local slice
                let rank = self.nrank();
                let procs = self.nprocs();
                let shape = &self.space.shape_spectral();
                let dcp = &self.space.get_decomp_from_global_shape(shape).x_pencil;
                let slice = ndarray::s![dcp.st[0]..=dcp.en[0], dcp.st[1]..=dcp.en[1]];
                // Write slice on processor after another
                // TODO: Figure hdf5 parallel writing out...
                for i in 0..procs {
                    if rank == i {
                        self.vhat_x_pen.assign(&$read_s(
                            filename,
                            &format!("{}/vhat", dsetname),
                            slice,
                        )?);
                    }
                    self.universe().world().barrier();
                }
                self.backward_mpi();
                Ok(())
            }

            /// Unwrap `read_mpi`
            pub fn read_mpi_unwrap(&mut self, filename: &str, dsetname: &str) {
                match self.read_mpi(filename, dsetname) {
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
            pub fn write_mpi(&self, filename: &str, varname: &str) -> Result<()> {
                // Geometry is all globally on root
                if self.nrank() == 0 {
                    write_to_hdf5(filename, &format!("{}/x", varname), &self.x[0])?;
                    write_to_hdf5(filename, &format!("{}/dx", varname), &self.x[0])?;
                    write_to_hdf5(filename, &format!("{}/y", varname), &self.x[1])?;
                    write_to_hdf5(filename, &format!("{}/dy", varname), &self.x[1])?;
                }

                // Write field
                let rank = self.nrank();
                let procs = self.nprocs();

                // Physical decomp
                let shape_p = &self.space.shape_physical();
                let dcp_p = &self.space.get_decomp_from_global_shape(shape_p).y_pencil;
                let slice_p = ndarray::s![dcp_p.st[0]..=dcp_p.en[0], dcp_p.st[1]..=dcp_p.en[1]];

                // Spectral decomp
                let shape_s = &self.space.shape_spectral();
                let dcp_s = &self.space.get_decomp_from_global_shape(shape_s).x_pencil;
                let slice_s = ndarray::s![dcp_s.st[0]..=dcp_s.en[0], dcp_s.st[1]..=dcp_s.en[1]];

                // Write slice on processor after another
                // TODO: Figure hdf5 parallel writing out...
                for i in 0..procs {
                    if rank == i {
                        write_slice(
                            filename,
                            &format!("{}/v", varname),
                            &self.v_y_pen,
                            slice_p,
                            shape_p,
                        )?;
                        $write_s(
                            filename,
                            &format!("{}/vhat", varname),
                            &self.vhat_x_pen,
                            slice_s,
                            shape_s,
                        )?;
                    }
                    self.universe().world().barrier();
                }
                Ok(())
            }

            /// Unwrap `write_mpi`
            pub fn write_mpi_unwrap(&self, filename: &str, varname: &str) {
                match self.write_mpi(filename, varname) {
                    Ok(_) => (),
                    Err(e) => eprintln!("Error while writing file {:?}. Error: {}", filename, e),
                }
            }
        }
    };
}

impl_read_write_mpi_fieldmpi2!(f64, read_slice, write_slice);
impl_read_write_mpi_fieldmpi2!(Complex<f64>, read_slice_complex, write_slice_complex);
