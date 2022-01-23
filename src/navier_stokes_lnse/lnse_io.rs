//! Implement io routines for `Navier2DLnse`
use super::lnse::OUTPUT_INTERVALL;
use super::Navier2DLnse;
use crate::field::Field2;
use crate::io::read_write_hdf5::{read_scalar_from_hdf5, write_scalar_to_hdf5};
use crate::io::traits::ReadWrite;
use crate::io::Result;
use crate::types::Scalar;
use funspace::BaseSpace;

impl<T, S> Navier2DLnse<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    Field2<T, S>: ReadWrite<T>,
{
    /// Read snapshot from file
    /// # Errors
    /// Failed to read
    pub fn read(&mut self, filename: &str) -> Result<()> {
        self.velx.read(&filename, "ux")?;
        self.vely.read(&filename, "uy")?;
        self.temp.read(&filename, "temp")?;
        self.pres.read(&filename, "pres")?;
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
        self.velx.backward();
        self.vely.backward();
        self.temp.backward();
        self.pres.backward();
        self.velx.write(&filename, "ux")?;
        self.vely.write(&filename, "uy")?;
        self.temp.write(&filename, "temp")?;
        self.pres.write(&filename, "pres")?;
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

impl<T, S> Navier2DLnse<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar,
    Field2<T, S>: ReadWrite<T>,
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
        use crate::navier_stokes::functions::norm_l2_f64;
        use std::io::Write;

        // Write hdf5 file
        std::fs::create_dir_all("data").unwrap();

        // Write flow field
        let out_intervall = write_flow_intervall.map_or(OUTPUT_INTERVALL, |x| x);
        if (self.time + self.dt / 2.) % out_intervall < self.dt {
            self.write_unwrap(&flow_name);
        }

        // I/O
        if !suppress_io {
            let div = self.div();
            self.field.vhat.assign(&div);
            self.field.v.assign(&self.velx.v.mapv(|x| x.powi(2)));
            let u2 = self.field.average();
            self.field.v.assign(&self.vely.v.mapv(|x| x.powi(2)));
            let v2 = self.field.average();
            self.field.v.assign(&self.temp.v.mapv(|x| x.powi(2)));
            let t2 = self.field.average();
            println!(
                "time = {:5.3}      |div| = {:4.2e}     u2 = {:5.3e}     v2 = {:5.3e}    t2 = {:5.3e}",
                self.time,
                norm_l2_f64(&self.field.space.backward(&div)),
                u2,
                v2,
                t2
            );
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .append(true)
                .create(true)
                .open(info_name)
                .unwrap();
            //write!(file, "{} {}", time, nu);
            if let Err(e) = writeln!(file, "{} {} {} {}", self.time, u2, v2, t2) {
                eprintln!("Couldn't write to file: {}", e);
            }
        }
    }
}
