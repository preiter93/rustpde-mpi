//! `Hdf5` functions to write ndarrays
use super::H5Type;
use super::Result;
use ndarray::{Array, ArrayBase, ArrayD, Dimension};
use num_complex::Complex;
use num_traits::Num;
use std::path::Path;

/// Retrieve size of dimenion from an hdf5 file
///
/// # Errors
/// When file or variable does not exists.
///
/// # Panics
/// Panics when requested dimension is not of dimensionality
/// 1 (i.e. not a scalar).
pub fn hdf5_get_size_dimension<P: AsRef<Path>>(filename: P, name: &str) -> hdf5::Result<usize> {
    let file = hdf5::File::open(filename)?;
    let dset = file.dataset(name)?;

    assert!(
        dset.shape().len() == 1,
        "Dimension must be of size 1, but is of size {}",
        dset.shape().len()
    );

    Ok(dset.shape()[0])
}

/// Read scalar from hdf5
///
/// # Errors
/// When file or variable does not exists.
///
/// # Panics
/// Panics when requested field is not of dimensionality
/// 1 (i.e. not a scalar).
pub fn read_scalar_from_hdf5<T>(filename: &str, name: &str) -> hdf5::Result<T>
where
    T: H5Type + Clone + Copy,
{
    let file = hdf5::File::open(filename)?;
    let dset = file.dataset(name)?;

    assert!(
        dset.shape().len() == 1,
        "Dimension must be of size 1, but is of size {}",
        dset.shape().len()
    );

    let scalar: ndarray::Array1<T> = dset.read()?;

    Ok(scalar[0])
}

/// Interface to wrtie scalar to hdf5 file
///
/// # Errors
/// When file does not exist.
pub fn write_scalar_to_hdf5<T>(filename: &str, name: &str, scalar: T) -> hdf5::Result<()>
where
    T: H5Type + Copy,
{
    use ndarray::Array1;
    let x = Array1::<T>::from_elem(1, scalar);
    write_to_hdf5(filename, name, &x)?;
    Ok(())
}

/// Read ndarray from hdf5 file
///
/// # Errors
/// Errors when file/variable does not exist and
/// when array is not supported by ndarrays
///
/// # Panics
/// Panics when  array is not supported by ndarrays
/// `into_dimensionality`.
pub fn read_from_hdf5<A, D>(filename: &str, varname: &str) -> Result<Array<A, D>>
where
    A: H5Type,
    D: Dimension,
{
    // Open file
    let file = hdf5::File::open(filename)?;

    //Read dataset
    let data = file.dataset(varname)?;
    let y: ArrayD<A> = data.read_dyn::<A>()?;

    // Dyn to static
    let x = y.into_dimensionality::<D>().unwrap();
    Ok(x)
}

/// Read complex typed ndarray from hdf5 file
///
/// # Errors
/// Errors when file/variable does not exist and
/// when array is not supported by ndarrays
///
/// # Panics
/// Panics when  array is not supported by ndarrays
/// `into_dimensionality`.
pub fn read_from_hdf5_complex<A, D>(filename: &str, varname: &str) -> Result<Array<Complex<A>, D>>
where
    A: H5Type + Num + Clone,
    D: Dimension,
{
    // Read real part
    let name_re = format!("{}_re", varname);
    let r = read_from_hdf5::<A, D>(filename, &name_re)?;

    // Read imaginary part
    let name_im = format!("{}_im", varname);
    let i = read_from_hdf5::<A, D>(filename, &name_im)?;

    // Add to array
    let mut array = Array::<Complex<A>, D>::zeros(r.raw_dim());
    let Complex { mut re, mut im } = array.view_mut().split_complex();
    re.assign(&r);
    im.assign(&i);
    Ok(array)
}

/// Write ndarray to hdf5 file
///
/// # Errors
/// When file does not exist or when file and
/// variable exists, but variable has different
/// shape than input array (assign new value will fail).
pub fn write_to_hdf5<A, S, D>(
    filename: &str,
    varname: &str,
    array: &ArrayBase<S, D>,
) -> hdf5::Result<()>
where
    A: H5Type,
    S: ndarray::Data<Elem = A>,
    D: ndarray::Dimension,
{
    // Open file
    let file = if Path::new(filename).exists() {
        hdf5::File::append(filename)?
    } else {
        hdf5::File::create(filename)?
    };

    //Write dataset
    let dset = match file.dataset(varname) {
        Ok(dset) => {
            // Overwrite
            dset
        }
        std::prelude::v1::Err(..) => {
            // Create new dataset
            file.new_dataset::<A>()
                .no_chunk()
                .shape(array.shape())
                .create(varname)?
        }
    };
    dset.write(&array.view())?;
    Ok(())
}

/// Write complex typed ndarray to hdf5 file
///
/// # Errors
/// When file does not exist or when file and
/// variable exists, but variable has different
/// shape than input array (assign new value will fail).
pub fn write_to_hdf5_complex<A, S, D>(
    filename: &str,
    varname: &str,
    array: &ArrayBase<S, D>,
) -> hdf5::Result<()>
where
    A: H5Type + Copy,
    S: ndarray::Data<Elem = Complex<A>>,
    D: ndarray::Dimension,
{
    // Write real part
    let name_re = format!("{}_re", varname);
    write_to_hdf5(filename, &name_re, &array.map(|x| x.re))?;
    // Write imag part
    let name_im = format!("{}_im", varname);
    write_to_hdf5(filename, &name_im, &array.map(|x| x.im))?;
    Ok(())
}
