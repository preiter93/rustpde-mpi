//! Implement io routines for `FieldBaseMpi`
#![cfg(not(feature = "mpio"))]
use super::{BaseSpace, BaseSpaceMpi, Field2Mpi};
use crate::io::read_write_hdf5::{read_from_hdf5, write_to_hdf5};
use crate::io::read_write_hdf5::{read_from_hdf5_complex, write_to_hdf5_complex};
use crate::io::traits::ReadWrite;
use crate::io::Result;
use ndarray::{s, Array2, Ix2};
use num_complex::Complex;
use num_traits::{One, Zero};
use std::ops::{Div, MulAssign};

macro_rules! impl_read_write_fieldmpi2 {
    ($a: ty, $s: ty, $read: ident , $write_p: ident , $write_s: ident) => {
        impl<S> ReadWrite<$s> for Field2Mpi<$s, S>
        where
            S: BaseSpace<$a, 2, Physical = $a, Spectral = $s> + BaseSpaceMpi<$a, 2>,
        {
            fn read(&mut self, filename: &str, varname: &str) -> Result<()> {
                if self.nrank() == 0 {
                    let data = $read::<$a, Ix2>(filename, &format!("{}/vhat", varname))?;
                    let mut vhat = Array2::<$s>::zeros(self.space.shape_spectral());
                    if data.shape() == vhat.shape() {
                        vhat.assign(&data);
                    } else {
                        interpolate_2d(&data, &mut vhat, &self.space);
                    }
                    self.scatter_spectral_root(&vhat);
                } else {
                    self.scatter_spectral();
                }
                self.backward_mpi();
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
                // Geometry is all globally on root
                if self.nrank() == 0 {
                    $write_p(filename, &format!("{}/x", varname), &self.x[0])?;
                    $write_p(filename, &format!("{}/dx", varname), &self.x[0])?;
                    $write_p(filename, &format!("{}/y", varname), &self.x[1])?;
                    $write_p(filename, &format!("{}/dy", varname), &self.x[1])?;
                }

                // Fields must be gathered on root
                if self.nrank() == 0 {
                    let mut v = Array2::<$a>::zeros(self.space.shape_physical());
                    self.gather_physical_root(&mut v);
                    $write_p(filename, &format!("{}/v", varname), &v)?;
                } else {
                    self.gather_physical();
                }
                if self.nrank() == 0 {
                    let mut vhat = Array2::<$s>::zeros(self.space.shape_spectral());
                    self.gather_spectral_root(&mut vhat);
                    $write_s(filename, &format!("{}/vhat", varname), &vhat)?;
                } else {
                    self.gather_spectral();
                }
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

impl_read_write_fieldmpi2!(f64, f64, read_from_hdf5, write_to_hdf5, write_to_hdf5);
impl_read_write_fieldmpi2!(
    f64,
    Complex<f64>,
    read_from_hdf5_complex,
    write_to_hdf5,
    write_to_hdf5_complex
);

/// Interpolate 2d array
#[allow(clippy::cast_precision_loss)]
fn interpolate_2d<T, S>(old: &Array2<T>, new: &mut Array2<T>, space: &S)
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
