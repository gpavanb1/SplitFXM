import numpy as np


class Cell:
    """
    A class representing a cell in a domain.

    Parameters
    ----------
    x : float, optional
        The x-coordinate of the cell. Default is `None`.
    value : numpy.ndarray, optional
        The values at the cell. Default is an empty NumPy array.
    """

    def __init__(self, x=None, value=np.array([])):
        """
        Initialize a `Cell` object.
        """
        self._value = value
        # X co-ordinate
        self._x = x

        # AMR deletion flag
        self.to_delete = False

    # Creating operators for sorting
    def __eq__(self, other):
        """
        Check if two cells have the same x-coordinate.

        Parameters
        ----------
        other : Cell
            The cell to compare to.

        Returns
        -------
        bool
            `True` if the cells have the same x-coordinate, `False` otherwise.
        """
        return self._x == other._x

    def __lt__(self, other):
        """
        Check if the x-coordinate of this cell is less than the x-coordinate of the other cell.

        Parameters
        ----------
        other : Cell
            The cell to compare to.

        Returns
        -------
        bool
            `True` if the x-coordinate of this cell is less than the x-coordinate of the other cell, `False` otherwise.
        """
        return self._x < other._x

    def x(self):
        """
        Return the x-coordinate of the cell.

        Returns
        -------
        float
            The x-coordinate of the cell.
        """
        return self._x

    def values(self):
        """
        Return the values at the cell.

        Returns
        -------
        numpy.ndarray
            The values at the cell.
        """
        return self._value

    def value(self, i: int):
        """
        Return the value at the cell for the given component index.
        Used primarily in AMR.

        Parameters
        ----------
        i : int
            The index of the component to retrieve.

        Returns
        -------
        float
            The value at the cell for the given component index.
        """
        return self._value[i]

    def set_value(self, i: int, val):
        """
        Set the value at the cell for the given component index.
        Used primarily in initial and boundary condition definitions.

        Parameters
        ----------
        i : int
            The index of the component to set.
        val : float
            The new value for the given component.
        """
        self._value[i] = val

    def set_values(self, l):
        """
        Set the values at the cell.

        Parameters
        ----------
        l : numpy.ndarray
            The new values for the cell.
        """
        self._value = l

    # Note boundary does not have update
    def update(self, dt, residual):
        """
        Update the values at the cell using the given time step and residual.

        Parameters
        ----------
        dt : float
            The time step to use in the update.
        residual : numpy.ndarray
            The residual to use in the update.
        """
        self._value += dt * residual
