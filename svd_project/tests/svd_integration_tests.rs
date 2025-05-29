use nalgebra::{DMatrix, DVector};
use svd_project::perform_svd;

// Helper function for comparing matrices
fn matrices_are_close(a: &DMatrix<f64>, b: &DMatrix<f64>, epsilon: f64) -> bool {
    if a.shape() != b.shape() {
        eprintln!("Matrix shapes differ: {:?} vs {:?}", a.shape(), b.shape());
        return false;
    }
    let diff = (a - b).abs().max();
    if diff >= epsilon {
        eprintln!("Matrix difference {} is not < epsilon {}. Diff matrix: {}", diff, epsilon, (a-b));
        return false;
    }
    true
}

// Helper function to create S_diag from singular values vector s
// U is m x k, V_t is k x n, s is k-vector. S_diag must be k x k for U * S_diag * V_t.
// nalgebra's from_diagonal will create a square matrix of size s.len() x s.len().
fn create_s_diag(s: &DVector<f64>) -> DMatrix<f64> {
    // Ensure s is not empty, otherwise from_diagonal might behave unexpectedly or panic.
    // However, SVD should not produce empty singular values for non-empty matrices.
    // If the matrix itself is 0x0 or similar, s could be empty.
    // For typical SVD, k = min(m,n), so s.len() is k.
    // The S_diag matrix should be k x k.
    DMatrix::from_diagonal(s)
}


#[cfg(test)]
mod tests {
    use super::*; // To get helper functions

    #[test]
    fn test_rectangular_matrix_properties() { // Renamed from test_svd_simple
        let original_matrix = DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]);
        let (m, n) = original_matrix.shape();
        let k = m.min(n);

        let svd_result = perform_svd(original_matrix.clone());
        assert!(svd_result.is_some(), "SVD computation failed for rectangular matrix");

        if let Some((u, s, v_t)) = svd_result {
            assert_eq!(u.nrows(), m, "U nrows mismatch");
            assert_eq!(u.ncols(), k, "U ncols mismatch");
            assert_eq!(s.len(), k, "S length mismatch");
            assert_eq!(v_t.nrows(), k, "V_t nrows mismatch");
            assert_eq!(v_t.ncols(), n, "V_t ncols mismatch");
        }
    }

    #[test]
    fn test_square_matrix_properties() { // Renamed from test_svd_square_matrix
        let original_matrix = DMatrix::from_row_slice(2, 2, &[
            2.0, 0.0,
            0.0, -1.0,
        ]);
        let (m, n) = original_matrix.shape();
        let k = m.min(n);
        
        let svd_result = perform_svd(original_matrix.clone());
        assert!(svd_result.is_some(), "SVD computation failed for square matrix");

        if let Some((u, s, v_t)) = svd_result {
            assert_eq!(u.shape(), (m, k), "U shape mismatch");
            assert_eq!(s.len(), k, "S length mismatch");
            assert_eq!(v_t.shape(), (k, n), "V_t shape mismatch");
        }
    }

    #[test]
    fn test_svd_handles_option_correctly() { // Renamed from test_svd_fail_case_not_really
        // This test ensures the function structure handles the Option correctly.
        // Since we request U and V_t, it should always return Some for valid inputs.
        let matrix = DMatrix::from_row_slice(1, 1, &[1.0]);
        let svd_result = perform_svd(matrix);
        assert!(svd_result.is_some(), "SVD result should be Some for a valid 1x1 matrix");
    }

    #[test]
    fn test_reconstruction_rectangular() {
        let original_matrix = DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]);
        let svd_result = perform_svd(original_matrix.clone());
        assert!(svd_result.is_some(), "SVD failed");
        let (u, s, v_t) = svd_result.unwrap();

        let s_diag = create_s_diag(&s);
        let reconstructed_matrix = u * s_diag * v_t;
        
        assert!(matrices_are_close(&original_matrix, &reconstructed_matrix, 1e-9),
                "Reconstructed rectangular matrix not close to original. \nOriginal:\n{}\nReconstructed:\n{}", original_matrix, reconstructed_matrix);
    }

    #[test]
    fn test_reconstruction_square() {
        let original_matrix = DMatrix::from_row_slice(3, 3, &[
            2.0, 1.0, 0.5,
            1.0, 3.0, 1.5,
            0.5, 1.5, 4.0
        ]);
        let svd_result = perform_svd(original_matrix.clone());
        assert!(svd_result.is_some(), "SVD failed");
        let (u, s, v_t) = svd_result.unwrap();
        
        let s_diag = create_s_diag(&s);
        let reconstructed_matrix = u * s_diag * v_t;

        assert!(matrices_are_close(&original_matrix, &reconstructed_matrix, 1e-9),
                "Reconstructed square matrix not close to original. \nOriginal:\n{}\nReconstructed:\n{}", original_matrix, reconstructed_matrix);
    }

    #[test]
    fn test_tall_matrix_reconstruction() {
        let original_matrix = DMatrix::from_row_slice(4, 2, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
        ]);
        let svd_result = perform_svd(original_matrix.clone());
        assert!(svd_result.is_some(), "SVD failed");
        let (u, s, v_t) = svd_result.unwrap();

        let s_diag = create_s_diag(&s);
        let reconstructed_matrix = u * s_diag * v_t;

        assert!(matrices_are_close(&original_matrix, &reconstructed_matrix, 1e-9),
                "Reconstructed tall matrix not close to original. \nOriginal:\n{}\nReconstructed:\n{}", original_matrix, reconstructed_matrix);
    }

    #[test]
    fn test_wide_matrix_reconstruction() {
        let original_matrix = DMatrix::from_row_slice(2, 4, &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]);
        let svd_result = perform_svd(original_matrix.clone());
        assert!(svd_result.is_some(), "SVD failed");
        let (u, s, v_t) = svd_result.unwrap();

        let s_diag = create_s_diag(&s);
        let reconstructed_matrix = u * s_diag * v_t;
        
        assert!(matrices_are_close(&original_matrix, &reconstructed_matrix, 1e-9),
                "Reconstructed wide matrix not close to original. \nOriginal:\n{}\nReconstructed:\n{}", original_matrix, reconstructed_matrix);
    }
}
