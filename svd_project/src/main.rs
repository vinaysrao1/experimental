use nalgebra::{DMatrix};
use svd_project::perform_svd; // Assuming lib is named svd_project

fn main() {
    // Create a sample DMatrix<f64>
    let matrix = DMatrix::from_row_slice(3, 2, &[
        1.0, 2.0, 
        3.0, 4.0, 
        5.0, 6.0
    ]);
    println!("Original Matrix:\n{}", matrix);

    // Perform SVD
    if let Some((u, s, v_t)) = perform_svd(matrix.clone()) {
        println!("\nU:\n{}", u);
        println!("\nS (Singular Values):\n{}", s);
        println!("\nV_t:\n{}", v_t);

        // Reconstruct the original matrix
        // S is a vector of singular values. U is m x k, V_t is k x n, where k = min(m,n).
        // S_diag should be k x k.
        let k = s.len();
        let mut s_diag_matrix = DMatrix::<f64>::zeros(k, k);
        for i in 0..k {
            s_diag_matrix[(i, i)] = s[i];
        }
        
        // U (m x k) * S_diag (k x k) * V_t (k x n) = A (m x n)
        let reconstructed_matrix = u * s_diag_matrix * v_t;
        println!("\nReconstructed Matrix:\n{}", reconstructed_matrix);

        // Verify reconstruction (optional, but good for demonstration)
        let diff = (&reconstructed_matrix - &matrix).abs().max();
        println!("\nMaximum absolute difference in reconstruction: {}", diff);
        if diff < 1e-10 {
            println!("Reconstruction successful within tolerance.");
        } else {
            println!("Reconstruction difference is larger than tolerance.");
        }

    } else {
        println!("SVD computation failed.");
    }
}
