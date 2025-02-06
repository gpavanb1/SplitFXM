import numpy as np
import numdifftools as nd
from scipy.integrate import solve_ivp

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

# Import from simhelper
from .simhelper import Solution, array_list_reshape, create_bound_events, apply_bounds

# Sparse matrix for Jacobian
from scipy.sparse import lil_matrix

# Methods that require a Jacobian
JAC_REQUIRED = ["Radau", "BDF"]


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
    scheme : Schemes
        The discretization scheme to use for the simulation.
    scheme_opts : dict, optional
        A dictionary of options for the scheme.
    ss : dict, optional
        The steady-state solver settings to use in the simulation.
    """

    def __init__(self, d: Domain, m: Model, ics: dict, bcs: dict, scheme, scheme_opts: dict = {}, ss: dict = {}):
        """
        Initialize a Simulation object.
        """
        self._d = d
        self._s = System(m, scheme, scheme_opts)
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

    def initialize_from_list(self, l, split=False, split_locs=None):
        """
        Initialize the domain from a list of values.

        Parameters
        ----------
        l : list
            The list of values to initialize the domain with.
        split : bool, optional
            Whether to split the values into outer and inner blocks. Defaults to False.
        split_locs : list[int], optional
            The location to split the values at. Required if `split` is True.
        """

        # Just demarcate every nv entries as a row in the 2D array
        num_points, nv = self.get_shape_from_list(l)

        if split:

            if split_locs is None:
                raise SFXM("Split location must be specified in this case")

            # Sort and remove duplicates to maintain consistency
            split_locs = sorted(set(split_locs))

            if split_locs[-1] > nv or split_locs[0] < 0:
                raise SFXM("Split locations must be between 0 and nv-1")

            # Same as SplitNewton convention
            # 1,2,1,2,3,3,4,5,4,5 for example for splits [2, 3]
            split_locs_full = [i*num_points for i in split_locs]
            l_split = np.split(l, split_locs_full)
            block = np.hstack([array_list_reshape(segment, (num_points, -1))
                               for segment in l_split])
        else:
            # No need to split, just use the values_array directly
            block = array_list_reshape(l, (num_points, nv))

        # Assign values to cells in domain
        cells = self._d.interior()
        for i, b in enumerate(cells):
            b.set_values(block[i, :])

    def get_residuals_from_list(self, l, split=False, split_locs=None):
        """
        Get the residuals for the domain given a list of values.

        Parameters
        ----------
        l : list
            The list of values to get the residuals for.
        split : bool, optional
            Whether to split the residuals into outer and inner blocks. Defaults to False.
        split_locs : list[int], optional
            The locations to split the residuals at. Required if `split` is True.

        Returns
        -------
        residual_list : list
            The list of residual values.
        """

        # Assign values from list
        # Note domain already exists and we preserve distances
        self.initialize_from_list(l, split, split_locs)

        # Fill BCs
        for c, bctype in self._bcs.items():
            apply_BC(self._d, c, bctype)

        interior_residual_block = self._s.residuals(self._d)

        if split:
            if split_locs is None:
                raise SFXM("Split locations must be specified in this case")

            # Sort and remove duplicates to maintain consistency
            split_locs = sorted(set(split_locs))

            # Ensure split locations are valid
            num_vars = interior_residual_block.shape[1]
            if split_locs[-1] > num_vars or split_locs[0] < 0:
                raise SFXM("Split locations must be between 0 and num_vars-1")

            # Split residuals into multiple blocks
            split_blocks = np.split(
                interior_residual_block, split_locs, axis=1)

            # Concatenate the flattened split blocks
            residual_list = np.concatenate(
                [block.flatten() for block in split_blocks])
        else:
            # Flatten the entire residual block
            residual_list = interior_residual_block.flatten()

        return np.array(residual_list, dtype=np.float64)

    def extend_bounds(self, bounds, num_points, nv, split=False, split_locs=None):
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
        split_locs : list[int], optional
            The indices at which to split the bounds if `split` is `True`. Default is `None`.

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
            if split_locs is None:
                raise SFXM("split_locs must be provided if split is True")

            # Sort and ensure unique split locations
            split_locs = sorted(set(split_locs))

            # Validate split locations
            if split_locs[-1] > nv or split_locs[0] < 0:
                raise SFXM("Split locations must be between 0 and nv-1")

            # Split bounds at specified locations
            lower_splits = np.split(bounds[0], split_locs)
            upper_splits = np.split(bounds[1], split_locs)

            # Extend each split block for num_points
            lower_extended = [np.tile(segment, num_points)
                              for segment in lower_splits]
            upper_extended = [np.tile(segment, num_points)
                              for segment in upper_splits]

            # Concatenate the extended bounds
            lower_bounds = np.concatenate(lower_extended).tolist()
            upper_bounds = np.concatenate(upper_extended).tolist()

            return [lower_bounds, upper_bounds]

    ############
    # Solution related methods
    ############

    def jacobian(self, l, split=False, split_locs=None, epsilon=1e-8):
        """
        Calculate the Jacobian of the system using finite differences.

        Parameters
        ----------
        l : list
            The list of values to calculate the Jacobian for.
        split : bool, optional
            Whether to split the Jacobian into multiple parts. Defaults to False.
        split_locs : list[int], optional
            A list of indices specifying where to split the Jacobian. Default is None.
        epsilon : float, optional
            The finite difference step size. Defaults to 1e-8.

        Returns
        -------
        jac : scipy.sparse.lil_matrix
            The Jacobian of the system.
        """
        # Initialize domain from the provided list and apply boundary conditions
        self.initialize_from_list(l, split, split_locs)
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

        # Iterating over interior points
        for i in range(ilo, ihi + 1):
            # Define the neighborhood and band around the current cell
            cell_sub = [cells[i + offset]
                        for offset in range(-nb_left, nb_right + 1)]
            # Indices of points that affect the Jacobian (or part of the band)
            band = list(range(max(ilo, i - nb_left),
                        min(ihi + 1, i + nb_right + 1)))

            # Calculate unperturbed residuals
            rhs = self._s._model.equation().residuals(cell_sub, self._s._scheme)

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
                    rhs_pert = self._s._model.equation().residuals(cell_sub, self._s._scheme)

                    # Reset the value
                    cell.set_value(j, current_value)
                    if ilo in band or ihi in band:
                        for c, bctype in self._bcs.items():
                            apply_BC(self._d, c, bctype)

                    # Compute the difference and assign to the Jacobian
                    col = (rhs_pert - rhs) / epsilon

                    # Determine how to split the Jacobian
                    if not split:
                        row_idx = (i - ilo) * nv
                        col_idx = (loc - ilo) * nv + j
                        jac[row_idx:row_idx + nv, col_idx] = col
                    else:
                        if split_locs is None:
                            raise SFXM(
                                "split_locs must be provided if split is True")

                        # Compute sizes of the split regions
                        split_sizes = [split_locs[0]] + [split_locs[k] - split_locs[k - 1]
                                                         for k in range(1, len(split_locs))] + [nv - split_locs[-1]]
                        offsets = [sum(split_sizes[:k])
                                   for k in range(len(split_sizes))]

                        # Compute row and col indices for each split section
                        for k, size in enumerate(split_sizes):
                            if offsets[k] <= j < offsets[k] + size:
                                # Determine the column index
                                # Only contributes to the k-th split
                                # if the variable is in the k-th split
                                offset = offsets[k] * num_points
                                col_idx = offset + \
                                    (loc - ilo) * size + (j - offsets[k])
                            else:
                                continue

                            # Determine the row index
                            # The residual from cell i is to be
                            # distributed among the k splits,
                            # each with their own row column offsets
                            offset = offsets[k] * num_points
                            row_idx = offset + (i - ilo) * size

                            jac[row_idx:row_idx + size,
                                col_idx] = col[offsets[k]:offsets[k] + size]

        return jac

    def evolve(self, t_diff: float, split=False, split_locs=None, method='RK45', rtol=1e-3, atol=1e-6, max_step=np.inf, bounds=None):
        """
        Evolve the system in time using an ODE solver for a given time step.

        Parameters
        ----------
        t_diff : float
            The time advancement to be made.
        split : bool, optional
            If True, applies a domain splitting technique. Defaults to False.
        split_locs : list[int], optional
            A list of indices specifying where to split the domain. Default is None.
        method : str, optional
            The integration method to use for solving the system. Defaults to 'RK45'.
            Possible options include 'RK45', 'RK23', 'DOP853', etc. 
            Refer to the scipy documentation for a full list of supported methods.
        rtol : float, optional
            The relative tolerance for the solver. Defaults to 1e-3.
        atol : float, optional
            The absolute tolerance for the solver. Defaults to 1e-6.
        max_step : float, optional
            The maximum time step to use in the solver. Defaults to np.inf.
        bounds: list, optional
            A list of lists representing the bounds of the domain. Defaults to None.

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

        def f(_, y): return self.get_residuals_from_list(y, split, split_locs)

        # Get the initial state from the domain
        y0 = self._d.listify_interior(split, split_locs)

        # Compute Jacobian if required
        jac = None
        if method in JAC_REQUIRED:
            jac = self.jacobian(y0, split, split_locs)

        # Get the shape of the initial state
        num_points, nv = self.get_shape_from_list(y0)

        # Construct bounds to be used
        ext_bounds = self.extend_bounds(
            bounds, num_points, nv, split, split_locs)

        # Use solve_ivp to evolve the system
        # Implement basic Euler as part of same wrapper
        sol = Solution()
        if method == "Euler":
            t_current = 0.0
            sol.y = y0
            while t_current < t_diff:
                delta_t = min(t_diff - t_current, max_step)
                sol.y += delta_t * f(t_current, sol.y)
                if bounds is not None:
                    apply_bounds(sol.y, ext_bounds, t_current)
                t_current += delta_t
        else:
            # Create bound events for the IVP solver
            events = create_bound_events(y0, ext_bounds)

            sol = solve_ivp(f, (0, t_diff), y0, method=method,
                            t_eval=[t_diff], jac=jac, max_step=max_step, events=events)

        # Update the values of the domain
        self.initialize_from_list(sol.y, split, split_locs)

    def steady_state(
        self, split=False, split_locs=None, sparse=True, dt0=0.0, dtmax=1.0, armijo=False, bounds=None
    ):
        """
        Solve for the steady state of the system.

        Parameters
        ----------
        split : bool, optional
            Whether to split the solution into outer and inner blocks. Defaults to False.
        split_locs : list[int], optional
            A list of indices specifying where to split the domain. Default is None.
        sparse : bool, optional
            Whether to use a sparse Jacobian. Defaults to True.
        dt0 : float, optional
            The initial time step to use in pseudo-time. Defaults to 0.0.
        dtmax : float, optional
            The maximum time step to use  in pseudo-time. Defaults to 1.0.
        armijo : bool, optional
            Whether to use the Armijo rule for line searches. Defaults to False.
        bounds: list, optional
            A list of lists representing the bounds of the domain. Defaults to None.


        Returns
        -------
        iter : int
            The number of iterations performed.
        """

        def _f(u): return self.get_residuals_from_list(u, split, split_locs)
        def _jac(u): return self.jacobian(u, split, split_locs)

        x0 = self._d.listify_interior(split, split_locs)
        num_points, nv = self.get_shape_from_list(x0)

        # Extend bounds based on input
        ext_bounds = self.extend_bounds(
            bounds, num_points, nv, split, split_locs)

        if not split:
            xf, _, iter = newton(
                _f, _jac, x0, sparse=sparse, dt0=dt0, dtmax=dtmax, armijo=armijo,
                bounds=ext_bounds)
        else:
            # Split location will be checked in initialize_from_list
            locs = [num_points * i for i in split_locs]
            xf, _, iter = split_newton(
                _f, _jac, x0, locs, sparse=sparse, dt0=dt0, dtmax=dtmax, armijo=armijo, bounds=ext_bounds
            )

        self.initialize_from_list(xf, split, split_locs)
        return iter
