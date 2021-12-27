import numpy as np
from numpy.fft import fft, fft2, ifft2, ifft, ifftshift, fftshift


# fft: one-dimensional discrete Fourier Transform
# ifft: one-dimensional inverse discrete Fourier Transform
# fft2: 2-dimensional discrete Fourier Transform
# ifft2: 2-dimensional inverse discrete Fourier Transform
# fftshift: shift the zero-frequency component to the center of the spectrum
#           i.e., from [0, N] to [-N/2, N/2-1] or [-(N-1)/2, (N-1)/2]
# ifftshift: the inverse of fftshift

def fftc(x, axis=-1, norm='ortho'):
    """
    get fftshifed input
    :param x: input array, expect x as m*n matrix
    :param axis: axis over which to compute the FFT. If not given, the last axis is used
    :param norm: {“backward”, “ortho”, “forward”}
    :return: complex ndarray
    """

    return fftshift(
        fft(
            ifftshift(x, axes=axis),
            axis=axis,
            norm=norm
        ),
        axes=axis
    )


def ifftc(x, axis=-1, norm='ortho'):
    """
    get ifftshifed input
    :param x:input array, expect x as m*n matrix
    :param axis: Axis over which to compute the inverse DFT. If not given, the last axis is used.
    :param norm: {“backward”, “ortho”, “forward”}
    :return: complex ndarray
    """

    return fftshift(
        ifft(
            ifftshift(x, axes=axis),
            axis=axis,
            norm=norm
        ),
        axes=axis
    )


def fft2c(x):
    """
    Centered fft, fft2 applies fft to last 2 axes by default
    :param x: 2D onwards, if its 3d, x.shape=(n, row, col); if 4d: x.shape=(n, slice, row, col)
    :return:
    """

    # get last 2 axes
    axes = (-2, -1)

    return fftshift(
        fft2(
            ifftshift(x, axes=axes),
            norm='ortho'
        ),
        axes=axes
    )


def ifft2c(x):
    """
    Centered ifft, ifft2 applies ifft to last 2 axes by default
    :param x: 2D onwards, if its 3d, x.shape=(n, row, col); if 4d: x.shape=(n, slice, row, col)
    :return:
    """

    # get last 2 axes
    axes = (-2, -1)

    return fftshift(
        ifft2(
            ifftshift(x, axes=axes),
            norm='ortho'
        ),
        axes=axes
    )


def fourier_matrix(rows, cols):
    """

    :param rows: number of rows
    :param cols: number of columns
    :return: unitary (rows x cols) fourier matrix
    """

    col_range = np.arange(cols)
    row_range = np.arange(rows)
    scale = 1 / np.sqrt(cols)

    # compute the outer product of two vector
    coeffs = np.outer(row_range, col_range)
    print(coeffs)

    return np.exp(coeffs * (-2. * np.pi * 1j / cols)) * scale


def inverse_fourier_matrix(rows, cols):
    return np.array(
        # Returns the (complex) conjugate transpose of `self`
        np.matrix(fourier_matrix(rows, cols)).getH()
    )


# flip function can be replaced by np.flip()

def rot90_nd(x, axes=(-2, -1), k=1):
    """
    Rotates selected axes
    :param x: input array
    :param axes: axes to be rorated
    :param k: different modes
    :return:
    """

    def flipud(x):
        return np.flip(x, axes[0])

    def fliplr(x):
        return np.flip(x, axes[1])

    # Convert the input to an ndarray, but pass ndarray subclasses through.
    x = np.asanyarray(x)

    if x.ndim < 2:
        raise ValueError("Input mast >= 2d.")

    k = k % 4

    if k == 0:
        return x
    elif k == 1:
        return fliplr(x).swapaxes(*axes)
    elif k == 2:
        return fliplr(flipud(x))
    else:
        return fliplr(x.swapaxes(*axes))


if __name__ == '__main__':
    print(fourier_matrix(3, 3))
