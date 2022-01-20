//! Traits for Field
use super::FieldBase;
use crate::io::read_from_hdf5_complex;
use crate::io::H5Type;
use crate::io::Result;
use ndarray::{Array, ArrayBase, ArrayD, Data, DataMut, Dimension, Ix1};
use num_complex::Complex;

/// Read and write field (hdf5)
pub trait ReadWriteHdf5 {
    /// Read field data from hdf5 file
    fn read(&mut self, filename: &str, varname: &str) -> Result<()>;
    /// Read field data from hdf5 file and handle result
    fn read_unwrap(&mut self, filename: &str, varname: &str);
    /// Write field data from hdf5 file
    fn write(&mut self, filename: &str, varname: &str) -> Result<()>;
}

// impl<A, S> ReadWriteHdf5 for FieldBase<A, A, A, S, 1>
// where
// // A: FloatNum + H5Type,
// // Complex<A>: ScalarOperand,
// // S: BaseSpace<A, 1, Physical = A, Spectral = A>,
// {
//     fn read(&mut self, filename: &str, group: Option<&str>) -> Result<()> {
//         todo!()
//     }
//
//     fn write(&mut self, filename: &str, group: Option<&str>) -> Result<()> {
//         todo!()
//     }
// }



impl<A, S, D> ReadWriteHdf5 for ArrayBase<S, D>
where
    S: Data<Elem = A> + DataMut,
    A: H5Type + Clone,
    D: Dimension,
{
    fn read(&mut self, filename: &str, varname: &str) -> Result<()> {
        let data = read_from_hdf5::<A, Ix1>(filename, varname)?;
        self.assign(&data);
        Ok(())
    }

    fn read_unwrap(&mut self, filename: &str, varname: &str) {
        match self.read(filename, varname) {
            Ok(_) => {
                println!("Reading file {:?} was successfull.", filename);
            }
            Err(_) => println!("Error while reading file {:?}.", filename),
        }
    }

    fn write(&mut self, filename: &str, varname: &str) -> Result<()> {
        todo!()
    }
}
