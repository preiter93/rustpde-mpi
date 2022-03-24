//! Implement io routines for `Navier2DLnse`
use super::lnse::OUTPUT_INTERVALL;
use super::nonlin_eq::DivNorm;
use super::Navier2DNonLin;
use crate::field::Field2;
use crate::io::read_write_hdf5::{read_scalar_from_hdf5, write_scalar_to_hdf5};
use crate::io::traits::ReadWrite;
use crate::io::Result;
use crate::types::Scalar;
use funspace::BaseSpace;
use std::ops::{Div, Mul};

impl<T, S> Navier2DNonLin<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    Field2<T, S>: ReadWrite<T>,
{
    /// Read snapshot from file
    /// # Errors
    /// Failed to read
    pub fn read(&mut self, filename: &str) -> Result<()> {
        self.velx.read(filename, "ux")?;
        self.vely.read(filename, "uy")?;
        self.temp.read(filename, "temp")?;
        self.pres.read(filename, "pres")?;
        self.time = read_scalar_from_hdf5::<f64>(filename, "time")?;
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
        self.velx.write(filename, "ux")?;
        self.vely.write(filename, "uy")?;
        self.temp.write(filename, "temp")?;
        self.pres.write(filename, "pres")?;
        self.mean.velx.write(filename, "ux_base")?;
        self.mean.vely.write(filename, "uy_base")?;
        self.mean.temp.write(filename, "temp_base")?;
        // Write scalars
        write_scalar_to_hdf5(filename, "time", self.time)?;
        for (key, value) in &self.params {
            write_scalar_to_hdf5(filename, key, *value)?;
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

impl<T, S> Navier2DNonLin<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar + Mul<f64, Output = T> + Div<f64, Output = T>,
    Field2<T, S>: ReadWrite<T>,
    Navier2DNonLin<T, S>: DivNorm,
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
        let out_intervall = write_flow_intervall.map_or(OUTPUT_INTERVALL, |x| x);
        if (self.time + self.dt / 2.) % out_intervall < self.dt {
            self.write_unwrap(flow_name);
        }

        // I/O
        if !suppress_io {
            let (div, nu, nuv, re) = (
                self.div_norm(),
                self.eval_nu(),
                self.eval_nuvol(),
                self.eval_re(),
            );

            self.field.v.assign(&self.velx.v.mapv(|x| x.powi(2)));
            let u2 = self.field.average();
            self.field.v.assign(&self.vely.v.mapv(|x| x.powi(2)));
            let v2 = self.field.average();
            self.field.v.assign(&self.temp.v.mapv(|x| x.powi(2)));
            let t2 = self.field.average();
            println!(
                "time = {:5.3} |div| = {:4.2e} Nu = {:5.3e} Nuv = {:5.3e} Re = {:5.3e} u2 = {:5.3e} v2 = {:5.3e} t2 = {:5.3e}",
                self.time,
                div,
                nu,
                nuv,
                re,
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
            if let Err(e) = writeln!(
                file,
                "{} {} {} {} {} {} {}",
                self.time, nu, nuv, re, u2, v2, t2
            ) {
                eprintln!("Couldn't write to file: {}", e);
            }
        }
    }
}

impl<T, S> Navier2DNonLin<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: Scalar + Mul<f64, Output = T> + Div<f64, Output = T>,
{
    /// Returns Nusselt number (heat flux at the plates)
    /// $$
    /// Nu = \langle - dTdz \rangle\\_x (0/H))
    /// $$
    pub fn eval_nu(&mut self) -> f64 {
        use super::functions::eval_nu;
        eval_nu::<f64, T, S>(
            &(&self.temp.to_ortho() + &self.mean.temp.vhat),
            &mut self.field,
            &self.scale,
        )
    }

    /// Returns volumetric Nusselt number
    /// $$
    /// Nuvol = \langle vely*T/kappa - dTdz \rangle\\_V
    /// $$
    ///
    /// # Panics
    /// If *ka* is not in params
    pub fn eval_nuvol(&mut self) -> f64 {
        use super::functions::eval_nuvol;
        let ka = self.params.get("ka").unwrap();
        eval_nuvol::<f64, T, S>(
            &(&self.temp.to_ortho() + &self.mean.temp.vhat),
            &(&self.vely.to_ortho() + &self.mean.vely.vhat),
            &mut self.field,
            *ka,
            &self.scale,
        )
    }

    /// Returns Reynolds number based on kinetic energy
    ///
    /// # Panics
    /// If *nu* is not in params
    pub fn eval_re(&mut self) -> f64 {
        use super::functions::eval_re;
        let nu = self.params.get("nu").unwrap();
        eval_re(
            &(&self.velx.to_ortho() + &self.mean.velx.vhat),
            &(&self.vely.to_ortho() + &self.mean.vely.vhat),
            &mut self.field,
            *nu,
            &self.scale,
        )
    }
}
