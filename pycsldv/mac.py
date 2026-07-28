"""
Modal assurance criterion (MAC) for comparing deflection shapes.
"""

import numpy as np

__all__ = ["mac"]


def mac(shape_1, shape_2):
    """
    Modal assurance criterion between two (possibly complex) shapes.

    :param shape_1: first shape (any array shape; flattened internally)
    :param shape_2: second shape, same number of elements
    :return: MAC value in ``[0, 1]``
    """
    shape_1 = np.ravel(np.asarray(shape_1))
    shape_2 = np.ravel(np.asarray(shape_2))
    numerator = np.abs(np.vdot(shape_1, shape_2)) ** 2
    denominator = np.vdot(shape_1, shape_1).real * np.vdot(shape_2, shape_2).real
    return numerator / denominator
