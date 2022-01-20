//! Implement io routines for `Navier2DMpi`
use super::Navier2DMpi;
use crate::field_mpi::Field2Mpi;
use crate::io::read_write_hdf5::{read_scalar_from_hdf5, write_scalar_to_hdf5};
use crate::io::traits::ReadWrite;
use crate::io::Result;
use crate::mpi::{broadcast_scalar, BaseSpaceMpi};
use funspace::BaseSpace;

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
        write_scalar_to_hdf5(&filename, "ra", self.ra)?;
        write_scalar_to_hdf5(&filename, "pr", self.pr)?;
        write_scalar_to_hdf5(&filename, "nu", self.nu)?;
        write_scalar_to_hdf5(&filename, "ka", self.ka)?;
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
