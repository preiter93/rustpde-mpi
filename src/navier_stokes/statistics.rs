//! Collect Statistics
use super::navier::Navier2D;
use crate::bases::BaseSpace;
use crate::field::Field2;
use crate::io::read_write_hdf5::{read_scalar_from_hdf5, write_scalar_to_hdf5};
use crate::io::traits::ReadWrite;
use crate::io::Result;
use ndarray::Array2;
use std::collections::HashMap;

/// Collection of fields for statistics
pub struct Statistics<T, S> {
    /// Parameter (e.g. diffusivities, ra, pr, ...)
    pub params: HashMap<&'static str, f64>,
    /// Scale of phsical dimension \[scale_x, scale_y\]
    pub scale: [f64; 2],
    /// Additional field for calculations
    pub field: Field2<T, S>,
    /// Temperature
    pub t_avg: Field2<T, S>,
    /// Horizontal velocity
    pub ux_avg: Field2<T, S>,
    /// Vertical velocity
    pub uy_avg: Field2<T, S>,
    /// Nusselt number
    pub nusselt: Field2<T, S>,
    /// Save stats every n-timeunits
    pub save_stat: f64,
    /// Write status every n-timeunits
    pub write_stat: f64,
    /// Time of average to this point
    pub avg_time: f64,
    /// Total time to this point
    pub tot_time: f64,
    /// Number of flow fields seen by statistics
    pub num_save: usize,
}

impl<T, S> Statistics<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: std::ops::Add<Output = T>
        + From<f64>
        + ndarray::LinalgScalar
        + std::ops::Mul<f64, Output = T>
        + std::ops::Div<f64, Output = T>,
{
    /// Allocate statistics
    #[allow(clippy::similar_names)]
    pub fn new(navier: &Navier2D<T, S>, save_stat: f64, write_stat: f64) -> Self {
        let params = navier.params.clone();
        let scale = navier.scale;
        let space: S = navier.field.space.clone();
        let field = Field2::new(&space);
        let t_avg = Field2::new(&space);
        let ux_avg = Field2::new(&space);
        let uy_avg = Field2::new(&space);
        let nusselt = Field2::new(&space);
        let avg_time = 0.;
        let tot_time = navier.time;
        let num_save = 0;
        Self {
            params,
            scale,
            field,
            t_avg,
            ux_avg,
            uy_avg,
            nusselt,
            save_stat,
            write_stat,
            avg_time,
            tot_time,
            num_save,
        }
    }

    /// Update Statistics
    ///
    /// # Panics
    /// If *ka* is not in params
    #[allow(clippy::similar_names)]
    #[allow(clippy::cast_precision_loss)]
    pub fn update(&mut self, that: &Array2<T>, uxhat: &Array2<T>, uyhat: &Array2<T>, time: f64) {
        // Statistics total time should be less then
        // Naviers total time, otherwise something is
        // wrong
        if time < self.tot_time {
            println!(
                "Statistics time mismatch (navier < stat): {:?} < {:?}",
                time, self.tot_time
            );
            return;
        }
        let weight = self.num_save as f64;
        self.t_avg
            .vhat
            .assign(&((&self.t_avg.vhat * weight + that) / (weight + 1.)));
        self.ux_avg.vhat.assign(uxhat);
        self.uy_avg.vhat.assign(uyhat);
        let ka = self.params.get("ka").unwrap();
        nusselt(&mut self.field, that, uyhat, *ka, &self.scale);
        self.nusselt.vhat.assign(&self.field.vhat);
        // Update time info
        self.num_save += 1;
        self.avg_time += time - self.tot_time;
        self.tot_time = time;
    }
}

impl<T, S> Statistics<T, S>
where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    Field2<T, S>: ReadWrite<T>,
{
    /// Read fields
    /// # Errors
    /// Failed to read
    pub fn read(&mut self, filename: &str) -> Result<()> {
        for field in &mut self.member_list() {
            field.0.read(filename, field.1)?;
        }
        // Read scalars
        self.tot_time = read_scalar_from_hdf5::<f64>(filename, "tot_time")?;
        self.avg_time = read_scalar_from_hdf5::<f64>(filename, "avg_time")?;
        self.num_save = read_scalar_from_hdf5::<usize>(filename, "num_save")?;
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
        // Write fields
        for field in &mut self.member_list() {
            field.0.backward();
            field.0.write(filename, field.1)?;
        }
        // Write scalars
        write_scalar_to_hdf5(filename, "tot_time", self.tot_time)?;
        write_scalar_to_hdf5(filename, "avg_time", self.avg_time)?;
        write_scalar_to_hdf5(filename, "num_save", self.num_save)?;
        // Write scalars
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

    /// Return member list [(reference, &str)]
    fn member_list(&mut self) -> [(&mut Field2<T, S>, &str); 4] {
        [
            (&mut self.t_avg, "temp"),
            (&mut self.ux_avg, "ux"),
            (&mut self.uy_avg, "uy"),
            (&mut self.nusselt, "nusselt"),
        ]
    }
}
//
// macro_rules! impl_read_write {
//     ($s: ty) => {
//         impl<S> Statistics<$s, S>
//         where
//             S: BaseSpace<f64, 2, Physical = f64, Spectral = $s>,
//         {
//             /// Write statistics
//             pub fn write(&mut self, filename: &str) {
//                 let result = self.write_return_result(filename);
//                 match result {
//                     Ok(_) => println!(" ==> {:?}", filename),
//                     Err(_) => println!("Error while writing file {:?}.", filename),
//                 }
//             }
//
//             fn write_return_result(&mut self, filename: &str) -> Result<()> {
//                 // use crate::hdf5::write_to_hdf5;
//                 use crate::field::write::WriteField;
//                 use crate::io::write_scalar_to_hdf5;
//                 // Transform to physical space
//                 self.t_avg.backward();
//                 self.ux_avg.backward();
//                 self.uy_avg.backward();
//                 self.nusselt.backward();
//                 // Write to file
//                 self.t_avg.write(&filename, Some("temp"));
//                 self.ux_avg.write(&filename, Some("ux"));
//                 self.uy_avg.write(&filename, Some("uy"));
//                 self.nusselt.write(&filename, Some("nusselt"));
//                 // Write scalars
//                 write_scalar_to_hdf5(&filename, "tot_time", None, self.tot_time)?;
//                 write_scalar_to_hdf5(&filename, "avg_time", None, self.avg_time)?;
//                 write_scalar_to_hdf5(&filename, "num_save", None, self.num_save)?;
//                 write_scalar_to_hdf5(&filename, "ra", None, self.ra)?;
//                 write_scalar_to_hdf5(&filename, "pr", None, self.pr)?;
//                 write_scalar_to_hdf5(&filename, "nu", None, self.nu)?;
//                 write_scalar_to_hdf5(&filename, "ka", None, self.ka)?;
//                 Ok(())
//             }
//
//             /// Read statistics file
//             pub fn read(&mut self, filename: &str) {
//                 use crate::field::read::ReadField;
//                 use crate::io::read_scalar_from_hdf5;
//                 // Field
//                 self.t_avg.read(&filename, Some("temp"));
//                 self.ux_avg.read(&filename, Some("ux"));
//                 self.uy_avg.read(&filename, Some("uy"));
//                 self.nusselt.read(&filename, Some("nusselt"));
//                 // Read scalars
//                 self.tot_time = read_scalar_from_hdf5::<f64>(&filename, "tot_time", None).unwrap();
//                 self.avg_time = read_scalar_from_hdf5::<f64>(&filename, "avg_time", None).unwrap();
//                 self.num_save =
//                     read_scalar_from_hdf5::<usize>(&filename, "num_save", None).unwrap();
//                 println!(" <== {:?}", filename);
//             }
//         }
//     };
// }
//
// impl_read_write!(f64);
// impl_read_write!(Complex<f64>);

/// Returns Nusselt field in field
fn nusselt<T, S>(
    field: &mut Field2<T, S>,
    that: &Array2<T>,
    uyhat: &Array2<T>,
    kappa: f64,
    scale: &[f64; 2],
) where
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
    T: ndarray::LinalgScalar + std::ops::Mul<f64, Output = T> + std::ops::Div<f64, Output = T>,
{
    // uy
    field.vhat.assign(uyhat);
    field.backward();
    let uy_v = field.v.clone();
    // temp
    field.vhat.assign(that);
    field.backward();
    // uy * T
    let uy_temp = &field.v * &uy_v;
    // dtdz
    let dtdz = field.gradient([0, 1], None) / (scale[1] * -1.);
    field.vhat.assign(&dtdz);
    field.backward();
    let dtdz = &field.v;
    // Nuvol
    field.v = (dtdz + uy_temp / kappa) * 2. * scale[1];
    field.forward();
}
