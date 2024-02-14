import numpy as np
import scipy.linalg


def get_gaussian_kernel(sigma: float, kernel_size: int=None) -> np.ndarray:
    """
    Get a 1D Gaussian kernel of size kernel_size and standard deviation sigma.
    If kernel_size is not provided, it is set to 4*sigma.
    """

    if kernel_size is None:
        kernel_size = int(4 * sigma)

    t = sigma**2
    x = np.arange(-kernel_size, kernel_size+1, 1)

    kernel = np.exp(- x**2 / (2 * t)) / np.sqrt(2 * t * np.pi)

    return kernel / np.sum(kernel)


def get_gaussian_kernel_1st_order(sigma: float, kernel_size: int=None) -> np.ndarray:
    """
    Get a 1D 1st order Gaussian kernel of size kernel_size and standard deviation sigma.
    If kernel_size is not provided, it is set to 4*sigma.
    """

    if kernel_size is None:
        kernel_size = int(4 * sigma)

    x = np.arange(-kernel_size, kernel_size+1, 1)

    kernel = - x * np.exp(- x**2 / (2 * sigma**2)) / (sigma**3 * np.sqrt(2 * np.pi))

    return kernel / np.sum(kernel)


def get_gaussian_kernel_2nd_order(sigma: float, kernel_size: int=None) -> np.ndarray:
    """
    Get a 1D 2nd order Gaussian kernel of size kernel_size and standard deviation sigma.
    If kernel_size is not provided, it is set to 4*sigma.
    """

    if kernel_size is None:
        kernel_size = int(4 * sigma)

    x = np.arange(-kernel_size, kernel_size+1, 1)

    kernel = (x**2 / sigma**4 - 1 / sigma**2) * np.exp(- x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    return kernel / np.sum(kernel)


def get_segmentation_boundary(img: np.ndarray) -> np.ndarray:
    """
    Get the boundary of a binary image.
    """

    diff_horizontal = (img[1:, :] != img[:-1, :])[:, :-1]
    diff_vertical = (img[:, 1:] != img[:, :-1])[:-1, :]

    return diff_horizontal | diff_vertical


def get_segmentation_boundary_length(img: np.ndarray) -> int:
    """
    Get the length of the boundary of a binary image.
    """

    diff_horizontal = (img[1:, :] != img[:-1, :])
    diff_vertical = (img[:, 1:] != img[:, :-1])

    return np.sum(diff_horizontal) + np.sum(diff_vertical)


def simple_smooth_curve(curve: np.ndarray, _lambda: float) -> np.ndarray:
    """
    Basic smoothing of a curve
    """

    N, _ = curve.shape
    L = np.zeros(N)
    L[0] = -2
    L[1] = L[-1] = 1

    L = scipy.linalg.circulant(_lambda * L)

    return (np.eye(N) + L) @ curve


def smooth_curve(curve: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    More advanced smoothing of a curve
    """

    N, _ = curve.shape
    A = np.zeros(N)
    A[0] = -2
    A[1] = A[-1] = 1

    B = np.zeros(N)
    B[0] = -6
    B[1] = B[-1] = 4
    B[2] = B[-2] = -1

    L = scipy.linalg.circulant(alpha * A + beta * B)

    return np.linalg.solve(np.eye(N) - L, curve)


def predict_afine_transform(points: np.ndarray, qoints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the transformation matrix from points to qoints
    which minimizes the least squares error
    where `qoints = s * (R @ points.T).T + t`

    returns:
    s: scale factor
    R: rotation matrix
    t: translation vector
    """
    mu_p = points.mean(axis=0)
    mu_q = qoints.mean(axis=0)

    p_zero = points - mu_p
    q_zero = qoints - mu_q

    s = np.linalg.norm(q_zero) / np.linalg.norm(p_zero)

    C = (q_zero).T @ (p_zero)
    U, Sigma, V = np.linalg.svd(C)
    R_hat = V.T @ U.T

    D = np.eye(2); D[1, 1] = np.linalg.det(R_hat)
    R = R_hat @ D

    t = mu_q - s * R_hat @ mu_p
    return s, R, t