import numpy as np
from .constants import btype
from .error import SFXM


class Boundary:
    """
    A class representing a boundary cell in a domain.

    Parameters
    ----------
    x : float
        The x-coordinate of the boundary cell.
    _btype : btype
        The type of boundary (left or right).
    value : numpy.ndarray, optional
        The values at the boundary cell. Default is an empty NumPy array.
    xmin : float, optional
        The minimum x-value of the domain. Default is 0.0.
    xmax : float, optional
        The maximum x-value of the domain. Default is 1.0.

    Raises
    ------
    SFXM
        If an inappropriate boundary type is given or if the x-value is an interior value.
    """

    def __init__(
        self, x, _btype, value=np.array([]), xmin: float = 0.0, xmax: float = 1.0
    ):
        """
        Initialize a `Boundary` object.
        """
        # Check if correct type specified
        if x < xmin and _btype == btype.RIGHT or x > xmax and _btype == btype.LEFT:
            raise SFXM("Inappropriate boundary type given")

        self._value = value
        self._type = _btype
        # X co-ordinate
        self._x = x

    def x(self):
        """
        Return the x-coordinate of the boundary cell.

        Returns
        -------
        float
            The x-coordinate of the boundary cell.
        """
        return self._x

    def values(self):
        """
        Return the values at the boundary cell.

        Returns
        -------
        numpy.ndarray
            The values at the boundary cell.
        """
        return self._value

    def value(self, i: int):
        """
        Return the value at the boundary cell for the given component index.

        Parameters
        ----------
        i : int
            The index of the component to retrieve.

        Returns
        -------
        float
            The value at the boundary cell for the given component index.
        """
        return self._value[i]

    # Note cell doesn't have set_x
    def set_x(self, x: float, xmin=0.0, xmax=1.0):
        """
        Set the x-coordinate of the boundary cell.

        Parameters
        ----------
        x : float
            The new x-coordinate for the boundary cell.
        xmin : float, optional
            The minimum x-value of the domain. Default is 0.0.
        xmax : float, optional
            The maximum x-value of the domain. Default is 1.0.

        Raises
        ------
        SFXM
            If the new x-value is an interior value or if an inappropriate boundary type is given.
        """
        if x >= xmin and x <= xmax:
            raise SFXM("Boundary cell cannot have interior X-value")

        # Check if correct type specified
        if (x < xmin and self._type == btype.RIGHT) or (
            x > xmax and self._type == btype.LEFT
        ):
            raise SFXM("Inappropriate boundary type given")
        self._x = x

    def set_value(self, i: int, val):
        """
        Set the value at the boundary cell for the given component index.

        Parameters
        ----------
        i : int
            The index of the component to set.
        val : float
            The new value for the given component.
        """
        self._value[i] = val
