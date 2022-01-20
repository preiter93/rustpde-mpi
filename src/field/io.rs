//! Implement io routines for `FieldBase`
#![allow(clippy::single_component_path_imports)]
use super::{BaseSpace, Field1, Field2};
use crate::io::read_write_hdf5::{read_from_hdf5, write_to_hdf5};
use crate::io::read_write_hdf5::{read_from_hdf5_complex, write_to_hdf5_complex};
use crate::io::traits::ReadWrite;
use crate::io::Result;
use ndarray::{s, Array1, Array2, Ix1, Ix2};
use num_complex::Complex;
use num_traits::{One, Zero};
use std::ops::{Div, MulAssign};
// pub(crate) use impl_read_write_field1;

macro_rules! impl_read_write_field1 {
    ($f: ty, $a: ty, $s: ty, $read: ident , $write_p: ident , $write_s: ident) => {
        impl<S> ReadWrite<$a> for $f
        where
            S: BaseSpace<$a, 1, Physical = $a, Spectral = $s>,
        {
            fn read(&mut self, filename: &str, varname: &str) -> Result<()> {
                let name_vhat = format!("{}/vhat", varname);
                let data = $read::<$a, Ix1>(filename, &name_vhat)?;
                if data.shape() == self.vhat.shape() {
                    self.vhat.assign(&data);
                } else {
                    interpolate_1d(&data, &mut self.vhat, &self.space);
                }
                self.backward();
                Ok(())
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
                $write_p(filename, &format!("{}/x", varname), &self.x[0])?;
                $write_p(filename, &format!("{}/dx", varname), &self.x[0])?;
                $write_p(filename, &format!("{}/v", varname), &self.v)?;
                $write_s(filename, &format!("{}/vhat", varname), &self.vhat)?;
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
impl_read_write_field1!(Field1<f64, S>, f64, f64, read_from_hdf5, write_to_hdf5, write_to_hdf5);
impl_read_write_field1!(
    Field1<Complex<f64>, S>,
    f64,
    Complex<f64>,
    read_from_hdf5_complex,
    write_to_hdf5,
    write_to_hdf5_complex
);

macro_rules! impl_read_write_field2 {
    ($f: ty, $a: ty, $s: ty, $read: ident , $write_p: ident , $write_s: ident) => {
        impl<S> ReadWrite<$s> for $f
        where
            S: BaseSpace<$a, 2, Physical = $a, Spectral = $s>,
        {
            fn read(&mut self, filename: &str, varname: &str) -> Result<()> {
                let name_vhat = format!("{}/vhat", varname);
                let data = $read::<$a, Ix2>(filename, &name_vhat)?;
                if data.shape() == self.vhat.shape() {
                    self.vhat.assign(&data);
                } else {
                    interpolate_2d(&data, &mut self.vhat, &self.space);
                }
                self.backward();
                Ok(())
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
                $write_p(filename, &format!("{}/x", varname), &self.x[0])?;
                $write_p(filename, &format!("{}/dx", varname), &self.x[0])?;
                $write_p(filename, &format!("{}/y", varname), &self.x[1])?;
                $write_p(filename, &format!("{}/dy", varname), &self.x[1])?;
                $write_p(filename, &format!("{}/v", varname), &self.v)?;
                $write_s(filename, &format!("{}/vhat", varname), &self.vhat)?;
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
impl_read_write_field2!(Field2<f64, S>, f64, f64, read_from_hdf5, write_to_hdf5, write_to_hdf5);
impl_read_write_field2!(
    Field2<Complex<f64>, S>,
    f64,
    Complex<f64>,
    read_from_hdf5_complex,
    write_to_hdf5,
    write_to_hdf5_complex
);

pub(crate) use impl_read_write_field2;

/// Interpolate 2d array
#[allow(clippy::cast_precision_loss)]
pub(crate) fn interpolate_1d<T, S>(old: &Array1<T>, new: &mut Array1<T>, space: &S)
where
    T: Zero + One + Copy + From<f64> + Div<Output = T> + MulAssign,
    S: BaseSpace<f64, 1, Physical = f64, Spectral = T>,
{
    use crate::bases::BaseKind;
    let minx = std::cmp::min(old.shape()[0], new.shape()[0]);
    new.fill(T::zero());
    new.slice_mut(s![..minx]).assign(&old.slice(s![..minx]));
    // Renormalize, depending on the base
    let norm: T = match space.base_kind(0) {
        BaseKind::FourierR2c => {
            T::from((new.shape()[0] - 1) as f64) / T::from((old.shape()[0] - 1) as f64)
        }
        _ => (T::one()),
    };
    for v in new.iter_mut() {
        *v *= norm;
    }
}

/// Interpolate 2d array
#[allow(clippy::cast_precision_loss)]
pub(crate) fn interpolate_2d<T, S>(old: &Array2<T>, new: &mut Array2<T>, space: &S)
where
    T: Zero + One + Copy + From<f64> + Div<Output = T> + MulAssign,
    S: BaseSpace<f64, 2, Physical = f64, Spectral = T>,
{
    use crate::bases::BaseKind;
    let sh: Vec<usize> = old
        .shape()
        .iter()
        .zip(new.shape().iter())
        .map(|(i, j)| *std::cmp::min(i, j))
        .collect();
    new.fill(T::zero());
    new.slice_mut(s![..sh[0], ..sh[1]])
        .assign(&old.slice(s![..sh[0], ..sh[1]]));
    // Renormalize, depending on the base
    let norm: T = match space.base_kind(0) {
        BaseKind::FourierR2c => {
            T::from((new.shape()[0] - 1) as f64) / T::from((old.shape()[0] - 1) as f64)
        }
        _ => (T::one()),
    };
    for v in new.iter_mut() {
        *v *= norm;
    }
}
