//! Implement io routines for `Navier2DAdjoint`
//! Implement io routines for `Navier2D`
#![allow(clippy::similar_names)]
use super::steady_adjoint::Navier2DAdjoint;
use super::steady_adjoint_eq::DivNorm;
use crate::field::Field2;
use crate::io::read_write_hdf5::{read_scalar_from_hdf5, write_scalar_to_hdf5};
use crate::io::traits::ReadWrite;
use crate::io::Result;
use crate::types::Scalar;
use funspace::BaseSpace;
use std::ops::{Div, Mul};

impl<T, S> Navier2DAdjoint<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    Field2<T, S>: ReadWrite<T>,
{
    /// Read snapshot from file
    /// # Errors
    /// Failed to read
    pub fn read(&mut self, filename: &str) -> Result<()> {
        // self.velx.read(&filename, "ux")?;
        // self.vely.read(&filename, "uy")?;
        // self.temp.read(&filename, "temp")?;
        // self.pres.read(&filename, "pres")?;
        self.navier.read(&filename)?;

        // TEST
        self.navier.pres.v.fill(0.);
        self.navier.pres.forward();
        // Read time
        self.time = read_scalar_from_hdf5::<f64>(&filename, "time")?;
        println!(" <== {:?}", filename);
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
        // self.velx.backward();
        // self.vely.backward();
        // self.temp.backward();
        // self.pres.backward()
        // self.velx.write(&filename, "ux")?;
        // self.vely.write(&filename, "uy")?;
        // self.temp.write(&filename, "temp")?;
        // self.pres.write(&filename, "pres")?;
        // if let Some(field) = &self.tempbc {
        //     field.write(&filename, "tempbc")?;
        // }
        self.navier.write(&filename)?;

        // Write scalars
        write_scalar_to_hdf5(&filename, "time", self.time)?;
        for (key, value) in &self.params {
            write_scalar_to_hdf5(&filename, key, *value)?;
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

impl<T, S> Navier2DAdjoint<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    Field2<T, S>: ReadWrite<T>,
    T: Scalar + Mul<f64, Output = T> + Div<f64, Output = T> + From<f64>,
    Navier2DAdjoint<T, S>: DivNorm,
{
    /// Define all I/O related routines for `Navier2DLnse`
    ///
    /// # Panics
    /// If folder `data` or file `info_name` cannot be created
    pub fn callback_from_filename(
        &mut self,
        flow_name: &str,
        info_name: &str,
        suppress_io: bool,
        write_flow_intervall: Option<f64>,
    ) {
        use std::io::Write;

        // Write hdf5 file
        std::fs::create_dir_all("data").unwrap();

        // Write flow field
        if let Some(dt_save) = write_flow_intervall {
            if (self.time + self.dt / 2.) % dt_save < self.dt {
                self.write_unwrap(&flow_name);
            }
        } else {
            self.write_unwrap(&flow_name);
        }

        // I/O
        if !suppress_io {
            let (div, nu, nuv, re) = (
                self.div_norm(),
                self.eval_nu(),
                self.eval_nuvol(),
                self.eval_re(),
            );
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
