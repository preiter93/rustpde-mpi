//! Four-diagonal matrix solver
use super::Solve;
use super::{diag, SolverScalar, Tdma};
use ndarray::prelude::*;
use ndarray::ScalarOperand;
use ndarray::{Data, DataMut, RemoveAxis, Zip};
use std::ops::{Add, Div, Mul};

/// Solve banded system with diagonals-offsets: -2, 0, 2, 4
#[derive(Debug, Clone)]
pub struct FdmaPar<T> {
    /// Size of matrix (= size of main diagonal)
    pub n: usize,
    /// Lower diagonal (-2)
    pub low: Array1<T>,
    /// Main diagonal
    pub dia: Array1<T>,
    /// Upper diagonal (+2)
    pub up1: Array1<T>,
    /// Upper diagonal (+4)
    pub up2: Array1<T>,
    /// ensure forward sweep is performed before solve
    sweeped: bool,
}

impl<T> FdmaPar<T>
where
    T: SolverScalar,
{
    /// Initialize Fdma from matrix.
    /// Extracts the diagonals.
    /// Precomputes the forward sweep.
    pub fn from_matrix(a: &Array2<T>) -> Self {
        let mut fdma = Self::from_matrix_raw(a);
        fdma.sweep();
        fdma
    }

    /// Initialize Fdma from matrix.
    /// Extracts only diagonals; no forward sweep is performed.
    /// Note that self.solve, for performance reasons, does not
    /// do the `forward_sweep` itself. So, if `from_matrix_raw`
    /// is used, this step must be executed manually before solve
    pub fn from_matrix_raw(a: &Array2<T>) -> Self {
        Self {
            n: a.shape()[0],
            low: diag(a, -2),
            dia: diag(a, 0),
            up1: diag(a, 2),
            up2: diag(a, 4),
            sweeped: false,
        }
    }

    /// Initialize `Fdma` from diagonals
    /// Precomputes the forward sweep.
    pub fn from_diags(low: &Array1<T>, dia: &Array1<T>, up1: &Array1<T>, up2: &Array1<T>) -> Self {
        let mut fdma = Self {
            n: dia.len(),
            low: low.to_owned(),
            dia: dia.to_owned(),
            up1: up1.to_owned(),
            up2: up2.to_owned(),
            sweeped: false,
        };
        fdma.sweep();
        fdma
    }

    /// Precompute forward sweep.
    /// The Arrays l,m,u1,u2 will deviate from the
    /// diagonals of the original matrix.
    pub fn sweep(&mut self) {
        for i in 2..self.n {
            self.low[i - 2] /= self.dia[i - 2];
            self.dia[i] -= self.low[i - 2] * self.up1[i - 2];
            if i < self.n - 2 {
                self.up1[i] -= self.low[i - 2] * self.up2[i - 2];
            }
        }
        self.sweeped = true;
    }

    fn solve_lane<A>(&self, input: &mut ArrayViewMut1<A>)
    where
        A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A>,
    {
        self.fdma(input);
    }

    /// Banded matrix solver
    ///     Ax = b
    /// where A is banded with diagonals in offsets -2, 0, 2, 4
    ///
    /// l:  sub-diagonal (-2)
    /// m:  main-diagonal (0)
    /// u1: sub-diagonal (+2)
    /// u2: sub-diagonal (+2)
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::assign_op_pattern)]
    pub fn fdma<A>(&self, x: &mut ArrayViewMut1<A>)
    where
        A: SolverScalar + Div<T, Output = A> + Mul<T, Output = A> + Add<T, Output = A>,
    {
        let n = self.n;

        for i in 2..n {
            x[i] = x[i] - x[i - 2] * self.low[i - 2];
        }

        x[n - 1] = x[n - 1] / self.dia[n - 1];
        x[n - 2] = x[n - 2] / self.dia[n - 2];
        x[n - 3] = (x[n - 3] - x[n - 1] * self.up1[n - 3]) / self.dia[n - 3];
        x[n - 4] = (x[n - 4] - x[n - 2] * self.up1[n - 4]) / self.dia[n - 4];
        for i in (0..n - 4).rev() {
            x[i] = (x[i] - x[i + 2] * self.up1[i] - x[i + 4] * self.up2[i]) / self.dia[i];
        }
    }
}

impl<T, A, D> Solve<A, D> for FdmaPar<T>
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
        assert!(
            self.sweeped,
            "Fdma: Forward sweep must be performed for solve! Abort."
        );
        output.assign(input);
        Zip::from(output.lanes_mut(Axis(axis))).par_for_each(|mut out| {
            self.solve_lane(&mut out);
        });
    }
}

/// Addition : Fdma + Fdma
impl<'a, 'b, T: SolverScalar> Add<&'b FdmaPar<T>> for &'a FdmaPar<T> {
    type Output = FdmaPar<T>;

    fn add(self, other: &'b FdmaPar<T>) -> FdmaPar<T> {
        assert!(!self.sweeped, "Add only unsweeped Fdma!");
        FdmaPar {
            n: self.n,
            low: &self.low + &other.low,
            dia: &self.dia + &other.dia,
            up1: &self.up1 + &other.up1,
            up2: &self.up2 + &other.up2,
            sweeped: false,
        }
    }
}

/// Addition : Fdma + Tdma
impl<'a, 'b, T: SolverScalar> Add<&'b Tdma<T>> for &'a FdmaPar<T> {
    type Output = FdmaPar<T>;

    fn add(self, other: &'b Tdma<T>) -> FdmaPar<T> {
        assert!(!self.sweeped, "Add only unsweeped Fdma!");
        FdmaPar {
            n: self.n,
            low: &self.low + &other.low,
            dia: &self.dia + &other.dia,
            up1: &self.up1 + &other.upp,
            up2: self.up2.to_owned(),
            sweeped: false,
        }
    }
}

/// Elementwise multiplication with scalar
impl<'a, T: SolverScalar + ScalarOperand> Mul<T> for &'a FdmaPar<T> {
    type Output = FdmaPar<T>;

    fn mul(self, other: T) -> FdmaPar<T> {
        assert!(!self.sweeped, "Mul only unsweeped Fdma!");
        FdmaPar {
            n: self.n,
            low: &self.low * other,
            dia: &self.dia * other,
            up1: &self.up1 * other,
            up2: &self.up2 * other,
            sweeped: false,
        }
    }
}
