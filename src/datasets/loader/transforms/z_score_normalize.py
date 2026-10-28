import numpy as np


class ZScoreNormalize:
    """
    The ZScoreNormalize class performs Z-score normalization on the input NumPy array.

    Z-score normalization standardizes data by subtracting the mean and dividing by the standard deviation for each element.
    This results in data with a mean of 0 and a standard deviation of 1.

    Attributes
    ----------
    None

    Methods
    -------
    __call__(x: np.ndarray) -> np.ndarray
        Returns the Z-score normalized array for the input x.
        x: NumPy array to be normalized
        Returns: Z-score normalized NumPy array
    """

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        mu = x.mean()
        sigma = x.std() + 1e-8
        new_x = (x - mu) / sigma
        return new_x, y
