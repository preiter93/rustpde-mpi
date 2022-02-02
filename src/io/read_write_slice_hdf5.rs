//! `Hdf5` functions to write ndarrays
use num_complex::Complex;

/// Write slice of ndarray dataset to hdf5 file.
/// Supply the full shape of the dataset, and
/// the slice where to put it. Array, and slices
/// shapes must be smaller than the full shape.
///
/// Overwrite dataset if it already exists
/// in the hdf5 file.
///
/// # Errors
/// File does not exist or file and dataset exist,
/// but a shape mismatch occurs.
///
/// # Panics
/// Mismatch of number of dimensions between array
/// and full shape.
pub fn write_slice<A, S, D, T, Sh>(
    filename: &str,
    varname: &str,
    array: &ndarray::ArrayBase<S, D>,
    slice: ndarray::SliceInfo<T, D, D>,
    full_shape: Sh,
) -> hdf5::Result<()>
where
    A: hdf5::H5Type,
    S: ndarray::Data<Elem = A>,
    D: ndarray::Dimension,
    T: AsRef<[ndarray::SliceInfoElem]>,
    Sh: Into<hdf5::Extents>,
{
    use std::convert::TryFrom;

    let full_shape = full_shape.into();
    assert!(
        array.ndim() == full_shape.ndim(),
        "Dimension mismatch of array and full_shape, {} vs. {}",
        array.ndim(),
        full_shape.ndim()
    );

    // Open file
    let file = if std::path::Path::new(filename).exists() {
        hdf5::File::append(filename)?
    } else {
        hdf5::File::create(filename)?
    };

    //Write dataset
    let dset = match file.dataset(varname) {
        // Overwrite
        Ok(dset) => dset,
        // Create new dataset
        std::prelude::v1::Err(..) => file
            .new_dataset::<A>()
            .no_chunk()
            .shape(full_shape)
            .create(varname)?,
    };
    dset.write_slice(array, hdf5::Hyperslab::try_from(slice)?)?;
    Ok(())
}

/// Write slice of ndarray dataset to hdf5 file.
/// Supply the full shape of the dataset, and
/// the slice where to put it. Array, and slices
/// shapes must be smaller than the full shape.
///
/// Overwrite dataset if it already exists
/// in the hdf5 file.
///
/// # Errors
/// File does not exist or file and dataset exist,
/// but a shape mismatch occurs.
///
/// # Panics
/// Mismatch of number of dimensions between array
/// and full shape.
pub fn write_slice_complex<A, S, D, T, Sh>(
    filename: &str,
    varname: &str,
    array: &ndarray::ArrayBase<S, D>,
    slice: ndarray::SliceInfo<T, D, D>,
    full_shape: Sh,
) -> hdf5::Result<()>
where
    A: hdf5::H5Type + Copy,
    S: ndarray::Data<Elem = Complex<A>>,
    D: ndarray::Dimension,
    T: AsRef<[ndarray::SliceInfoElem]>,
    Sh: Into<hdf5::Extents> + Copy,
    ndarray::SliceInfo<T, D, D>: Copy,
{
    // Write real part
    let name_re = format!("{}_re", varname);
    write_slice(filename, &name_re, &array.map(|x| x.re), slice, full_shape)?;
    // Write imag part
    let name_im = format!("{}_im", varname);
    write_slice(filename, &name_im, &array.map(|x| x.im), slice, full_shape)?;
    Ok(())
}

/// Read slice of ndarray from hdf5 file.
///
/// # Errors
/// Non-existing file or dataset.
///
/// # Panics
/// Panics when array is not supported by ndarrays
/// `into_dimensionality`.
pub fn read_slice<A, D, T>(
    filename: &str,
    varname: &str,
    slice: ndarray::SliceInfo<T, D, D>,
) -> hdf5::Result<ndarray::Array<A, D>>
where
    A: hdf5::H5Type,
    D: ndarray::Dimension,
    T: AsRef<[ndarray::SliceInfoElem]>,
{
    // Open file
    let file = hdf5::File::open(filename)?;
    //Read dataset
    let data = file.dataset(varname)?;
    let y: ndarray::ArrayD<A> = data.read_slice(slice)?;
    // Dyn to static
    let x = y.into_dimensionality::<D>().unwrap();
    Ok(x)
}

/// Read slice of ndarray from hdf5 file.
///
/// # Errors
/// Non-existing file or dataset.
///
/// # Panics
/// Panics when array is not supported by ndarrays
/// `into_dimensionality`.
pub fn read_slice_complex<A, D, T>(
    filename: &str,
    varname: &str,
    slice: ndarray::SliceInfo<T, D, D>,
) -> hdf5::Result<ndarray::Array<Complex<A>, D>>
where
    A: hdf5::H5Type + Clone + num_traits::Num,
    D: ndarray::Dimension,
    T: AsRef<[ndarray::SliceInfoElem]>,
    ndarray::SliceInfo<T, D, D>: Copy,
{
    // Read real part
    let name_re = format!("{}_re", varname);
    let r = read_slice::<A, D, T>(filename, &name_re, slice)?;

    // Read imaginary part
    let name_im = format!("{}_im", varname);
    let i = read_slice::<A, D, T>(filename, &name_im, slice)?;

    // Add to array
    let mut array = ndarray::Array::<Complex<A>, D>::zeros(r.raw_dim());
    let Complex { mut re, mut im } = array.view_mut().split_complex();
    re.assign(&r);
    im.assign(&i);
    Ok(array)
}
