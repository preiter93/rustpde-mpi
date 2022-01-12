//! Solve banded system with diagonals-offsets: -2, 1, 0, 1, 2, 3, 4
///
/// # References
/// <https://www.hindawi.com/journals/mpe/2015/232456/\>
///
/// Slightly modified, with 2 more upper diagonals
use super::Solve;
use super::{diag, SolverScalar};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, DataMut, Dimension, Ix1, RemoveAxis, Zip};
use std::ops::{Add, Div, Mul};

/// Solve banded system with diagonals-offsets: -2, 1, 0, 1, 2, 3, 4
///
/// # References
/// <https://www.hindawi.com/journals/mpe/2015/232456/\>
///
/// Slightly modified, with 2 more upper diagonals
#[derive(Debug, Clone)]
pub struct PdmaPlus2<T> {
    /// Size of matrix (= size of main diagonal)
    pub n: usize,
    /// Upper diagonal (+1) after sweep
    pub al: Array1<T>,
    /// Upper diagonal (+2) after sweep
    pub be: Array1<T>,
    /// Upper diagonal (+3) after sweep
    pub ga: Array1<T>,
    /// Upper diagonal (+4) after sweep
    pub de: Array1<T>,
    /// Work array
    pub l2: Array1<T>,
    /// Work array
    pub ka: Array1<T>,
    /// Work array
    pub mu: Array1<T>,
}

impl<T> PdmaPlus2<T>
where
    T: SolverScalar,
{
    /// Initialize `PdmaPlus2` from matrix.
    /// Extracts the diagonals.
    /// Precomputes the forward sweep.
    pub fn from_matrix(a: &Array2<T>) -> Self {
        let n = a.shape()[0];

        // Extract diagonals
        let l2 = diag(a, -2);
        let l1 = diag(a, -1);
        let d0 = diag(a, 0);
        let u1 = diag(a, 1);
        let u2 = diag(a, 2);
        let u3 = diag(a, 3);
        let u4 = diag(a, 4);

        // Allocate arrays
        let mut al = Array1::<T>::zeros(n);
        let mut be = Array1::<T>::zeros(n);
        let mut ga = Array1::<T>::zeros(n);
        let mut de = Array1::<T>::zeros(n);
        let mut ka = Array1::<T>::zeros(n);
        let mut mu = Array1::<T>::zeros(n);

        // Precompute sweep
        mu[0] = d0[0];
        al[0] = u1[0] / mu[0];
        be[0] = u2[0] / mu[0];
        ga[0] = u3[0] / mu[0];
        de[0] = u4[0] / mu[0];

        ka[1] = l1[0];
        mu[1] = d0[1] - al[0] * ka[1];
        al[1] = (u1[1] - be[0] * ka[1]) / mu[1];
        be[1] = (u2[1] - ga[0] * ka[1]) / mu[1];
        ga[1] = (u3[1] - de[0] * ka[1]) / mu[1];
        de[1] = u4[1] / mu[1];

        for i in 2..n - 4 {
            ka[i] = l1[i - 1] - al[i - 2] * l2[i - 2];
            mu[i] = d0[i] - be[i - 2] * l2[i - 2] - al[i - 1] * ka[i];
            al[i] = (u1[i] - ga[i - 2] * l2[i - 2] - be[i - 1] * ka[i]) / mu[i];
            be[i] = (u2[i] - de[i - 2] * l2[i - 2] - ga[i - 1] * ka[i]) / mu[i];
            ga[i] = (u3[i] - de[i - 1] * ka[i]) / mu[i];
            de[i] = u4[i] / mu[i];
        }

        ka[n - 4] = l1[n - 5] - al[n - 6] * l2[n - 6];
        mu[n - 4] = d0[n - 4] - be[n - 6] * l2[n - 6] - al[n - 5] * ka[n - 4];
        al[n - 4] = (u1[n - 4] - ga[n - 6] * l2[n - 6] - be[n - 5] * ka[n - 4]) / mu[n - 4];
        be[n - 4] = (u2[n - 4] - de[n - 6] * l2[n - 6] - ga[n - 5] * ka[n - 4]) / mu[n - 4];
        ga[n - 4] = (u3[n - 4] - de[n - 5] * ka[n - 4]) / mu[n - 4];

        ka[n - 3] = l1[n - 4] - al[n - 5] * l2[n - 5];
        mu[n - 3] = d0[n - 3] - be[n - 5] * l2[n - 5] - al[n - 4] * ka[n - 3];
        al[n - 3] = (u1[n - 3] - ga[n - 5] * l2[n - 5] - be[n - 4] * ka[n - 3]) / mu[n - 3];
        be[n - 3] = (u2[n - 3] - de[n - 5] * l2[n - 5] - ga[n - 4] * ka[n - 3]) / mu[n - 3];

        ka[n - 2] = l1[n - 3] - al[n - 4] * l2[n - 4];
        mu[n - 2] = d0[n - 2] - be[n - 4] * l2[n - 4] - al[n - 3] * ka[n - 2];
        al[n - 2] = (u1[n - 2] - ga[n - 4] * l2[n - 4] - be[n - 3] * ka[n - 2]) / mu[n - 2];

        ka[n - 1] = l1[n - 2] - al[n - 3] * l2[n - 3];
        mu[n - 1] = d0[n - 1] - be[n - 3] * l2[n - 3] - al[n - 2] * ka[n - 1];

        PdmaPlus2 {
            n,
            al,
            be,
            ga,
            de,
            l2,
            ka,
            mu,
        }
    }

    fn solve_lane<S1, A>(&self, rhs: &mut ArrayBase<S1, Ix1>)
    where
        S1: DataMut<Elem = A>,
        A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A>,
    {
        let n = self.n;
        let al = &self.al;
        let be = &self.be;
        let ga = &self.ga;
        let de = &self.de;
        let l2 = &self.l2;
        let ka = &self.ka;
        let mu = &self.mu;
        let mut ze = Array1::<A>::zeros(n);

        // Forward step
        ze[0] = rhs[0] / mu[0];
        ze[1] = (rhs[1] - ze[0] * ka[1]) / mu[1];
        for i in 2..n - 3 {
            ze[i] = (rhs[i] - ze[i - 2] * l2[i - 2] - ze[i - 1] * ka[i]) / mu[i];
        }
        ze[n - 3] = (rhs[n - 3] - ze[n - 5] * l2[n - 5] - ze[n - 4] * ka[n - 3]) / mu[n - 3];
        ze[n - 2] = (rhs[n - 2] - ze[n - 4] * l2[n - 4] - ze[n - 3] * ka[n - 2]) / mu[n - 2];
        ze[n - 1] = (rhs[n - 1] - ze[n - 3] * l2[n - 3] - ze[n - 2] * ka[n - 1]) / mu[n - 1];

        // Backward substitution
        rhs[n - 1] = ze[n - 1];
        rhs[n - 2] = ze[n - 2] - rhs[n - 1] * al[n - 2];
        rhs[n - 3] = ze[n - 3] - rhs[n - 2] * al[n - 3] - rhs[n - 1] * be[n - 3];
        rhs[n - 4] =
            ze[n - 4] - rhs[n - 3] * al[n - 4] - rhs[n - 2] * be[n - 4] - rhs[n - 1] * ga[n - 4];

        for i in (0..n - 4).rev() {
            rhs[i] = ze[i]
                - rhs[i + 1] * al[i]
                - rhs[i + 2] * be[i]
                - rhs[i + 3] * ga[i]
                - rhs[i + 4] * de[i];
        }
    }
}

impl<T, A, D> Solve<A, D> for PdmaPlus2<T>
where
    T: SolverScalar,
    A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A> + From<T>,
    D: Dimension + RemoveAxis,
{
    fn solve<S1: Data<Elem = A>, S2: Data<Elem = A> + DataMut>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) {
        output.assign(input);
        Zip::from(output.lanes_mut(Axis(axis))).for_each(|mut out| {
            self.solve_lane(&mut out);
        });
    }

    fn solve_par<S1: Data<Elem = A>, S2: Data<Elem = A> + DataMut>(
        &self,
        input: &ArrayBase<S1, D>,
        output: &mut ArrayBase<S2, D>,
        axis: usize,
    ) {
        output.assign(input);
        Zip::from(output.lanes_mut(Axis(axis))).par_for_each(|mut out| {
            self.solve_lane(&mut out);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn approx_eq<S, D>(result: &ArrayBase<S, D>, expected: &ArrayBase<S, D>)
    where
        S: Data<Elem = f64>,
        D: Dimension,
    {
        let dif = 1e-3;
        for (a, b) in expected.iter().zip(result.iter()) {
            if (a - b).abs() > dif {
                panic!("Large difference of values, got {} expected {}.", b, a)
            }
        }
    }

    #[test]
    fn test_pdma_dim1() {
        let nx = 6;
        type Ty = f64;
        let mut data = Array1::<Ty>::zeros(nx);
        let mut result = Array1::<Ty>::zeros(nx);
        let mut matrix = Array2::<Ty>::zeros((nx, nx));
        for (i, v) in data.iter_mut().enumerate() {
            *v = i as f64;
        }
        for i in 0..nx {
            let j = (i + 1) as f64;
            matrix[[i, i]] = 0.5 * j;
            if i > 1 {
                matrix[[i, i - 2]] = 10. * j;
            }
            if i > 0 {
                matrix[[i, i - 1]] = 4. * j;
            }
            if i < nx - 1 {
                matrix[[i, i + 1]] = 1.5 * j;
            }
            if i < nx - 2 {
                matrix[[i, i + 2]] = 3.5 * j;
            }
            if i < nx - 3 {
                matrix[[i, i + 3]] = 4.5 * j;
            }
            if i < nx - 4 {
                matrix[[i, i + 4]] = 2.5 * j;
            }
        }
        let solver = PdmaPlus2::from_matrix(&matrix);
        solver.solve(&data, &mut result, 0);
        let recover: Array1<Ty> = matrix.dot(&result);
        approx_eq(&recover, &data);
    }
}
