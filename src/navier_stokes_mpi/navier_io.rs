//! Implement io routines for `Navier2DMpi`
use super::navier_eq::DivNorm;
use super::Navier2DMpi;
use crate::field_mpi::Field2Mpi;
use crate::io::read_write_hdf5::{read_scalar_from_hdf5, write_scalar_to_hdf5};
use crate::io::traits::ReadWrite;
use crate::io::Result;
use crate::mpi::{broadcast_scalar, BaseSpaceMpi};
use crate::types::Scalar;
use funspace::BaseSpace;
use std::ops::{Div, Mul};

impl<T, S> Navier2DMpi<'_, T, S>
where
    T: num_traits::Zero,
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>
        + BaseSpaceMpi<f64, 2, Physical = f64, Spectral = T>,
    Field2Mpi<T, S>: ReadWrite<T>,
{
    /// Read snapshot from file
    /// # Errors
    /// Failed to read
    pub fn read(&mut self, filename: &str) -> Result<()> {
        if self.nrank() == 0 {
            self.ux.read(&filename, "ux")?;
            self.uy.read(&filename, "uy")?;
            self.temp.read(&filename, "temp")?;
            self.pres.read(&filename, "pres")?;
            self.time = read_scalar_from_hdf5::<f64>(&filename, "time")?;
            println!(" <== {:?}", filename);
        }
        self.scatter();
        broadcast_scalar(self.universe, &mut self.time);
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
        if self.nrank() == 0 {
            self.ux.backward();
            self.uy.backward();
            self.temp.backward();
            self.pres.backward();
            self.ux.write(&filename, "ux")?;
            self.uy.write(&filename, "uy")?;
            self.temp.write(&filename, "temp")?;
            self.pres.write(&filename, "pres")?;
            if let Some(field) = &self.tempbc {
                field.write(&filename, "tempbc")?;
            }
            // Write scalars
            write_scalar_to_hdf5(&filename, "time", self.time)?;
            for (key, value) in &self.params {
                write_scalar_to_hdf5(&filename, key, *value)?;
            }
        }
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

impl<'a, T, S> Navier2DMpi<'a, T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T> + BaseSpaceMpi<f64, 2>,
    Field2Mpi<T, S>: ReadWrite<T>,
    T: Scalar + Mul<f64, Output = T> + Div<f64, Output = T> + From<f64>,
    Navier2DMpi<'a, T, S>: DivNorm,
{
    /// Define all I/O related routines for `Navier2DLnse`
    ///
    /// # Panics
    /// If folder `data` or file `info_name` cannot be created
    pub fn callback_from_filename(&mut self, flow_name: &str, info_name: &str, suppress_io: bool) {
        use std::io::Write;

        // Write hdf5 file
        if self.nrank() == 0 {
            std::fs::create_dir_all("data").unwrap();
        }

        // Write flow field
        if let Some(dt_save) = &self.write_intervall {
            if (self.time + self.dt / 2.) % dt_save < self.dt {
                self.gather();
                self.write_unwrap(&flow_name);
            }
        } else {
            // TODO: Parallel writing, get rid of gather...
            self.gather();
            self.write_unwrap(&flow_name);
        }

        // I/O
        if !suppress_io {
            let (div, nu, nuv, re) = (
                self.div_norm(),
                self.eval_nu_mpi(),
                self.eval_nuvol_mpi(),
                self.eval_re_mpi(),
            );
            if self.nrank() == 0 {
                println!(
                "time = {:4.2}      |div| = {:4.2e}     Nu = {:5.3e}     Nuv = {:5.3e}    Re = {:5.3e}",
                self.time,
                div,
                nu,
                nuv,
                re,
            );
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .append(true)
                    .create(true)
                    .open(info_name)
                    .unwrap();
                if let Err(e) = writeln!(file, "{} {} {} {}", self.time, nu, nuv, re) {
                    eprintln!("Couldn't write to file: {}", e);
                }
            }
        }
    }
}
