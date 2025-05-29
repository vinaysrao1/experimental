use nalgebra::{DMatrix, DVector, linalg::SVD};

/// Computes the Singular Value Decomposition (SVD) of a matrix.
///
/// # Arguments
///
/// * `matrix` - The matrix to decompose.
///
/// # Returns
///
/// An `Option` containing a tuple `(U, S, V_t)` if successful, where:
/// - `U` is the matrix of left singular vectors.
/// - `S` is the vector of singular values.
/// - `V_t` is the transpose of the matrix of right singular vectors.
/// Returns `None` if U or V_t could not be computed.
pub fn perform_svd(matrix: DMatrix<f64>) -> Option<(DMatrix<f64>, DVector<f64>, DMatrix<f64>)> {
    // Compute SVD, requesting U and V_t
    let svd = SVD::new(matrix, true, true);

    // Extract U, S, and V_t
    // .u and .v_t return Option<&DMatrix<T>>, so we need to check if they are Some.
    // We clone them to return owned matrices.
    match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => {
            let s = svd.singular_values;
            Some((u.clone(), s, v_t.clone()))
        }
        _ => None, // If U or V_t is None, return None
    }
}

// The #[cfg(test)] mod tests { ... } block has been removed.
// Tests are now in tests/svd_integration_tests.rs
