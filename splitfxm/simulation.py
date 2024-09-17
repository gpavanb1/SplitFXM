import numpy as np
import numdifftools as nd
from scipy.integrate import solve_ivp
from scipy.optimize import OptimizeResult
from splitnewton.newton import newton
from splitnewton.split_newton import split_newton

from .constants import btype
from .domain import Domain
from .schemes import stencil_sizes
from .error import SFXM
from .system import System
from .refine import Refiner
from .model import Model

# ICs and BCs
from .bc import apply_BC, extend_band, get_periodic_bcs
from .initialize import set_initial_condition


# Sparse matrix for Jacobian
from scipy.sparse import lil_matrix

# Methods that require a Jacobian
JAC_REQUIRED = ["Radau", "BDF"]

# Basic class to assign solution for Euler timestep


class Solution(OptimizeResult):
    pass


def array_list_reshape(l, shape):
    """
    Reshape a list into a list of 1D NumPy arrays.

    Parameters
    ----------
    l : list
        The list to reshape.
    shape : tuple
        The shape to reshape the list into.

    Returns
    -------
    reshaped_list : list of ndarray
        The reshaped list.
    """

    # Reshape to list of 1D numpy arrays
    return [np.array(x) for x in np.reshape(l, shape).tolist()]


class Simulation:
    """
    A class representing a simulation.

    Parameters
    ----------
    d : Domain
        The domain on which to perform the simulation.
    m : Model
        The model to use in the simulation.
    ics : dict
        The initial conditions to apply to the domain.
    bcs : dict
        The boundary conditions to apply to the domain.
    ss : dict, optional
        The steady-state solver settings to use in the simulation.
    """

    def __init__(self, d: Domain, m: Model, ics: dict, bcs: dict, scheme: dict, ss: dict = {}):
        """
        Initialize a Simulation object.
        """
        self._d = d
        self._s = System(m, scheme)
        self._r = Refiner()
        self._bcs = bcs

        # Steady-state solver settings
        self._ss = ss

        # Set initial conditions
        for c, ictype in ics.items():
            set_initial_condition(self._d, c, ictype)

        # Fill BCs
        for c, bctype in self._bcs.items():
            apply_BC(self._d, c, bctype)

        # Check if stencil size matches
        boundary_cells = self._d.boundaries()[0] + self._d.boundaries()[1]
        if stencil_sizes.get(self._s._scheme) != len(boundary_cells) + 1:
            raise SFXM("Stencil size does not match boundary conditions")

    ############
    # List related methods
    ############

    def get_shape_from_list(self, l):
        """
        Get the shape of the list when reshaped into a NumPy array.

        Parameters
        ----------
        l : list
            The list to get the shape for.

        Returns
        -------
        num_points : int
            The number of points in the reshaped array.
        nv : int
            The number of components per point in the reshaped array.
        """

        nv = self._d.num_components()

        if len(l) % nv != 0:
            raise SFXM("List length not aligned with interior size")

        num_points = len(l) // nv

        return num_points, nv

    def initialize_from_list(self, l, split=False, split_loc=None):
        """
        Initialize the domain from a list of values.

        Parameters
        ----------
        l : list
            The list of values to initialize the domain with.
        split : bool, optional
            Whether to split the values into outer and inner blocks. Defaults to False.
        split_loc : int, optional
            The location to split the values at. Required if `split` is True.
        """

        # Just demarcate every nv entries as a block
        # This gives block-diagonal structure in unsplit case
        num_points, nv = self.get_shape_from_list(l)

        if split:

            if split_loc is None:
                raise SFXM("Split location must be specified in this case")

            if split_loc > nv or split_loc < 0:
                raise SFXM("Split location must be between 0 and nv-1")

            # Same as SplitNewton convention
            # Outer system will be excluding `loc`
            outer_block = array_list_reshape(
                l[: split_loc * num_points], (-1, split_loc)
            )
            inner_block = array_list_reshape(
                l[split_loc * num_points:], (-1, nv - split_loc)
            )

            block = []
            for i in range(num_points):
                block.append(np.concatenate((outer_block[i], inner_block[i])))
        else:
            # Reshape list
            block = array_list_reshape(l, (num_points, nv))

        # Assign values to cells in domain
        cells = self._d.interior()
        for i, b in enumerate(cells):
            b.set_values(block[i])

    def get_residuals_from_list(self, l, split=False, split_loc=None):
        """
        Get the residuals for the domain given a list of values.

        Parameters
        ----------
        l : list
            The list of values to get the residuals for.
        split : bool, optional
            Whether to split the residuals into outer and inner blocks. Defaults to False.
        split_loc : int, optional
            The location to split the residuals at. Required if `split` is True.

        Returns
        -------
        residual_list : list
            The list of residual values.
        """

        # Assign values from list
        # Note domain already exists and we preserve distances
        self.initialize_from_list(l, split, split_loc)

        # Fill BCs
        for c, bctype in self._bcs.items():
            apply_BC(self._d, c, bctype)

        interior_residual_block = self._s.residuals(self._d)

        if split:
            outer_block = [x[:split_loc] for x in interior_residual_block]
            inner_block = [x[split_loc:] for x in interior_residual_block]
            outer_list = np.array(outer_block).flatten()
            inner_list = np.array(inner_block).flatten()
            residual_list = np.concatenate((outer_list, inner_list))
        else:
            # Reshape residual block in list order
            residual_list = np.array(interior_residual_block).flatten()

        return residual_list

    def extend_bounds(self, bounds, num_points, nv, split=False, split_loc=None):
        """
        Extends the provided input bounds based on whether there is a split or not.

        Parameters
        ----------
        bounds : list of list
            A list containing two lists, each of size `nv`, representing the lower and upper bounds.
        num_points : int
            The number of points to extend each bound to.
        nv : int
            The number of variables, indicating the length of each bound list.
        split : bool, optional
            A flag indicating whether to split the bounds at a specific location. Default is `False`.
        split_loc : int, optional
            The index at which to split the bounds if `split` is `True`. Default is `None`.

        Returns
        -------
        list of list
            A list containing the extended lower and upper bounds.
        """
        # Check if bounds is a 2-list, each of size nv
        if bounds is None:
            return None

        if len(bounds) != 2:
            raise SFXM("Bounds must be a list of 2 lists")
        else:
            if len(bounds[0]) != nv or len(bounds[1]) != nv:
                raise SFXM(
                    "Each list in bounds must be of length - number of variables")

        if not split:
            return [bounds[0] * num_points, bounds[1] * num_points]
        else:
            if split_loc is None:
                raise SFXM("split_loc must be provided if split is True")
            return [bounds[0][:split_loc] * num_points + bounds[0][split_loc:] * num_points,
                    bounds[1][:split_loc] * num_points + bounds[1][split_loc:] * num_points]

    ############
    # Solution related methods
    ############

    def jacobian(self, l, split=False, split_loc=None, epsilon=1e-8):
        """
        Calculate the Jacobian of the system using finite differences.

        Parameters
        ----------
        l : list
            The list of values to calculate the Jacobian for.
        split : bool, optional
            Whether to split the Jacobian into outer and inner blocks. Defaults to False.
        split_loc : int, optional
            The location to split the Jacobian at. Required if `split` is True.
        epsilon : float, optional
            The finite difference step size. Defaults to 1e-8.

        Returns
        -------
        jac : scipy.sparse.lil_matrix
            The Jacobian of the system.
        """
        # Initialize domain from the provided list and apply boundary conditions
        self.initialize_from_list(l, split, split_loc)
        for c, bctype in self._bcs.items():
            apply_BC(self._d, c, bctype)

        # Find the variables with periodic BCs and their directions
        periodic_bcs_dict = get_periodic_bcs(self._bcs, self._d)

        # Get the number of points and variables
        num_points, nv = self.get_shape_from_list(l)
        n = num_points * nv

        # Create a sparse matrix for the Jacobian
        jac = lil_matrix((n, n))

        # Retrieve cells and boundary parameters
        cells = self._d.cells()
        nb_left, nb_right, ilo, ihi = self._d.nb(btype.LEFT), self._d.nb(
            btype.RIGHT), self._d.ilo(), self._d.ihi()

        # Use same point iteration as system residuals
        # Iterating over interior points
        for i in range(ilo, ihi + 1):
            # Define the neighborhood and band around the current cell
            cell_sub = [cells[i + offset]
                        for offset in range(-nb_left, nb_right + 1)]
            # Indices of points that affect the Jacobian (or part of the band)
            band = list(range(max(ilo, i - nb_left),
                        min(ihi + 1, i + nb_right + 1)))

            # Calculate unperturbed residuals
            rhs = np.concatenate([eq.residuals(cell_sub, self._s._scheme)
                                  for eq in self._s._model.equations()])

            # Perturb each variable and compute the Jacobian columns
            for j in range(nv):
                # Extend the band if required for that variable
                # Only required for periodic BC for now
                if j in periodic_bcs_dict.keys():
                    dirs = periodic_bcs_dict[j]
                    band = extend_band(band, dirs, i, self._d)

                for loc in band:
                    # Compared to center_cell, what is the index of the cell
                    # to be perturbed
                    cell = cells[loc]
                    current_value = cell.value(j)

                    # Perturb the current variable and compute perturbed residuals
                    cell.set_value(j, current_value + epsilon)

                    # Apply BC again if cell is adjacent to boundary
                    if ilo in band or ihi in band:
                        for c, bctype in self._bcs.items():
                            apply_BC(self._d, c, bctype)

                    # Calculate updated residual
                    rhs_pert = np.concatenate([eq.residuals(cell_sub, self._s._scheme)
                                               for eq in self._s._model.equations()])

                    # Reset the value
                    cell.set_value(j, current_value)
                    # Apply BC again if cell is adjacent to boundary
                    if ilo in band or ihi in band:
                        for c, bctype in self._bcs.items():
                            apply_BC(self._d, c, bctype)

                    # Compute the difference and assign to the Jacobian
                    col = (rhs_pert - rhs) / epsilon

                    # Assign the calculated column to the Jacobian
                    # Blocks of nv*nv matrices are computed
                    # row_idx gives the start of the column vector
                    # First point starts at 0, second point starts at nv and so on
                    # For col_idx, loc - ilo gives shift from 0 for the block in multiples of nv
                    if not split:
                        row_idx = (i - ilo) * nv
                        col_idx = (loc - ilo) * nv + j
                        jac[row_idx:row_idx + nv, col_idx] = col
                    else:
                        # Split location will be checked in initialize_from_list
                        # Sizes of the sub-Jacobians
                        na, nc = split_loc, (nv - split_loc)
                        # Jumps of block_offset instead of nv here
                        block_offset = num_points * na

                        # First part of residuals
                        # Same as previous but use na instead
                        row_idx = (i - ilo) * na
                        # col_idx jumps to right part of Jacobian
                        # depending on the variable if pre- or post-split
                        if j < na:
                            col_idx = (loc - ilo) * na + j
                        else:
                            # Jumps of nc in post-split parts
                            # j-na to ensure post-split variables start afresh
                            col_idx = block_offset + \
                                (loc - ilo) * nc + (j - na)

                        jac[row_idx:row_idx + na, col_idx] = col[:na]

                        # Second part of residuals
                        # col_idx remains same as previous
                        row_idx = block_offset + (i - ilo) * nc
                        jac[row_idx:row_idx + nc, col_idx] = col[na:]

        return jac

    def evolve(self, t_diff: float, split=False, split_loc=None, method='RK45', rtol=1e-3, atol=1e-6, max_step=np.inf):
        """
        Evolve the system in time using an ODE solver for a given time step.

        Parameters
        ----------
        t_diff : float
            The time advancement to be made.
        split : bool, optional
            If True, applies a domain splitting technique. Defaults to False.
        split_loc : optional
            Specifies the location or configuration for domain splitting. Default is None.
        method : str, optional
            The integration method to use for solving the system. Defaults to 'RK45'.
            Possible options include 'RK45', 'RK23', 'DOP853', etc. 
            Refer to the scipy documentation for a full list of supported methods.
        rtol : float, optional
            The relative tolerance for the solver. Defaults to 1e-6.
        atol : float, optional
            The absolute tolerance for the solver. Defaults to 1e-6.

        Notes
        -----
        This method uses `scipy.integrate.solve_ivp` to solve the system of residuals in time.
        The system's state is updated using the computed solution at the end of the time step.

        The `get_residuals_from_list` function is used to obtain the system's residuals, and
        `initialize_from_list` updates the domain's state after solving.

        Returns
        -------
        None
            The system's state is updated in-place.
        """

        def f(_, y): return self.get_residuals_from_list(y, split, split_loc)

        # Get the initial state from the domain
        y0 = self._d.listify_interior(split, split_loc)

        # Compute Jacobian if required
        jac = None
        if method in JAC_REQUIRED:
            jac = self.jacobian(y0, split, split_loc)

        # Use solve_ivp to evolve the system
        # Implement basic Euler as part of same wrapper
        sol = Solution()
        if method == "Euler":
            delta_t = min(t_diff, max_step)
            sol.y = y0 + delta_t * f(0., y0)
        else:
            sol = solve_ivp(f, (0, t_diff), y0, method=method,
                            t_eval=[t_diff], jac=jac, max_step=max_step)

        # Update the values of the domain
        self.initialize_from_list(sol.y, split, split_loc)

    def steady_state(
        self, split=False, split_loc=None, sparse=True, dt0=0.0, dtmax=1.0, armijo=False, bounds=None
    ):
        """
        Solve for the steady state of the system.

        Parameters
        ----------
        split : bool, optional
            Whether to split the solution into outer and inner blocks. Defaults to False.
        split_loc : int, optional
            The location to split the solution at. Required if `split` is True.
        sparse : bool, optional
            Whether to use a sparse Jacobian. Defaults to True.
        dt0 : float, optional
            The initial time step to use in pseudo-time. Defaults to 0.0.
        dtmax : float, optional
            The maximum time step to use  in pseudo-time. Defaults to 1.0.
        armijo : bool, optional
            Whether to use the Armijo rule for line searches. Defaults to False.

        Returns
        -------
        iter : int
            The number of iterations performed.
        """

        def _f(u): return self.get_residuals_from_list(u, split, split_loc)
        def _jac(u): return self.jacobian(u, split, split_loc)

        x0 = self._d.listify_interior(split, split_loc)
        num_points, nv = self.get_shape_from_list(x0)

        # Extend bounds based on input
        ext_bounds = self.extend_bounds(
            bounds, num_points, nv, split, split_loc)

        if not split:
            xf, _, iter = newton(
                _f, _jac, x0, sparse=sparse, dt0=dt0, dtmax=dtmax, armijo=armijo,
                bounds=ext_bounds)
        else:
            # Split location will be checked in initialize_from_list
            loc = num_points * split_loc
            xf, _, iter = split_newton(
                _f, _jac, x0, loc, sparse=sparse, dt0=dt0, dtmax=dtmax, armijo=armijo, bounds=ext_bounds
            )

        self.initialize_from_list(xf, split, split_loc)
        return iter
