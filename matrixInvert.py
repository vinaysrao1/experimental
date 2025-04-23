import numpy as np

# ... existing code ...

def invert_matrix(matrix):
    """
    Inverts an nxn matrix using NumPy.
    Args:
        matrix: A square numpy array or nested list
    Returns:
        The inverted matrix as a numpy array
    """
    try:
        # Convert to numpy array if not already
        matrix = np.array(matrix)
        # Check if matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")
        # Calculate inverse
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular and cannot be inverted")

def svd_decomposition(matrix):
    """
    Performs Singular Value Decomposition on a matrix.
    Args:
        matrix: A numpy array or nested list
    Returns:
        U: Left singular vectors
        S: Singular values
        Vh: Right singular vectors (transposed)
    """
    try:
        # Convert to numpy array if not already
        matrix = np.array(matrix)
        # Calculate SVD
        U, S, Vh = np.linalg.svd(matrix)
        return U, S, Vh
    except np.linalg.LinAlgError:
        raise ValueError("SVD computation did not converge")

def main():
    # ... existing main code ...

    # Example usage of matrix inversion
    test_matrix = [
        [4, 7],
        [2, 6]
    ]
    try:
        inverse = invert_matrix(test_matrix)
        print("\nOriginal matrix:")
        print(np.array(test_matrix))
        print("\nInverted matrix:")
        print(inverse)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()