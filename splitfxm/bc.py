from .domain import Domain
from .constants import btype, btype_map
from .error import SFXM


def apply_BC(d: Domain, v: str, bc: dict = {"left": "periodic", "right": "periodic"}, xmin=0.0, xmax=1.0):
    """
    Apply boundary conditions to the given domain.

    Parameters
    ----------
    d : Domain
        The domain to apply the boundary conditions to.
    v : str
        The name of the variable to apply the boundary conditions to.
    bc : str, optional
        The type of boundary condition to apply. Acceptable values are "periodic" and "outflow". Default is "periodic".
    xmin : float, optional
        The minimum x-value of the domain. Default is 0.0.
    xmax : float, optional
        The maximum x-value of the domain. Default is 1.0.

    Raises
    ------
    NotImplementedError
        If an unsupported boundary condition is specified.
    """

    # Get cells
    cells = d.cells()

    # Find index of component
    idx = d.component_index(v)

    # Common values used in all types
    ilo = d.ilo()
    ihi = d.ihi()

    lb, rb = d.boundaries()

    # Check only required directions specified
    bc_keys = sorted(list(bc.keys()))
    if bc_keys != [btype_map[btype.LEFT], btype_map[btype.RIGHT]]:
        raise SFXM("Incorrect boundary directions specified")

    # Iterate over left and right BCs
    for dir in bc_keys:
        bc_type = bc[dir]

        if bc_type == "periodic":
            if dir == btype_map[btype.LEFT]:
                # left boundary
                # Ghost cells are (right to left) the rightmost elements (same order)
                for i, b in enumerate(lb):
                    shift = xmax - cells[ihi - (i + 1)].x()
                    b.set_x(xmin - shift)
                    b.set_value(idx, cells[ihi - i].value(idx))

            elif dir == btype_map[btype.RIGHT]:
                # right boundary
                # Ghost cells are (left to right) the leftmost elements (same order)
                for i, b in enumerate(rb):
                    shift = cells[(i + 1) + ilo].x() - xmin
                    b.set_x(xmax + shift)
                    b.set_value(idx, cells[i + ilo].value(idx))

        elif bc_type == "outflow":
            if dir == btype_map[btype.LEFT]:
                # left boundary
                for i, b in enumerate(lb):
                    # The shift mirrors the interior on the same side
                    shift = cells[(i + 1) + ilo].x() - xmin
                    b.set_x(xmin - shift)
                    # Value same as extrapolated from interior
                    dy = cells[ilo + 1].value(idx) - cells[ilo].value(idx)
                    dx = cells[ilo + 1].x() - cells[ilo].x()
                    delta_x = cells[ilo - i].x() - cells[ilo - (i + 1)].x()

                    b.set_value(
                        idx, cells[ilo - i].value(idx) - (dy/dx) * delta_x)

            elif dir == btype_map[btype.RIGHT]:
                # right boundary
                for i, b in enumerate(rb):
                    # The shift mirrors the interior on the same side
                    shift = xmax - cells[ihi - (i + 1)].x()
                    b.set_x(xmax + shift)
                    # Value same as extrapolated from interior
                    dy = cells[ihi].value(idx) - cells[ihi - 1].value(idx)
                    dx = cells[ihi].x() - cells[ihi - 1].x()
                    delta_x = cells[ihi + (i + 1)].x() - cells[ihi + i].x()

                    b.set_value(
                        idx, cells[ihi + i].value(idx) + (dy/dx) * delta_x)

        # Dictionary-based boundary conditions
        # Dirichlet and Neumann BCs require additional values also
        elif isinstance(bc_type, dict) and len(bc_type.keys()) == 1:
            bc_data = list(bc_type.values())[0]
            bc_type = list(bc_type.keys())[0]

            # Check if BC data is valid
            if not isinstance(bc_data, (float, int)):
                raise SFXM(
                    "Incorrect data specified for dictionary-type boundaries")

            if bc_type == "neumann":
                if dir == btype_map[btype.LEFT]:
                    # left boundary
                    neumann_value = bc_data
                    for i, b in enumerate(lb):
                        # The shift mirrors the interior on the same side
                        shift = cells[(i + 1) + ilo].x() - xmin
                        b.set_x(xmin - shift)

                        # Cell width to the left
                        dx = cells[ilo - i].x() - cells[ilo - (i + 1)].x()
                        b.set_value(
                            idx, cells[ilo - i].value(idx) - neumann_value * dx)

                elif dir == btype_map[btype.RIGHT]:
                    # right boundary
                    neumann_value = bc_data
                    for i, b in enumerate(rb):
                        # The shift mirrors the interior on the same side
                        shift = xmax - cells[ihi - (i + 1)].x()
                        b.set_x(xmax + shift)

                        # Cell width to the right
                        dx = cells[ihi + (i + 1)].x() - cells[ihi + i].x()
                        b.set_value(
                            idx, cells[ihi + i].value(idx) + neumann_value * dx)

            elif bc_type == "dirichlet":
                if dir == btype_map[btype.LEFT]:
                    # left boundary
                    dirichlet_value = bc_data
                    for i, b in enumerate(lb):
                        # The shift mirrors the interior on the same side
                        shift = cells[(i + 1) + ilo].x() - xmin
                        b.set_x(xmin - shift)

                        b.set_value(idx, dirichlet_value)

                elif dir == btype_map[btype.RIGHT]:
                    # right boundary
                    dirichlet_value = bc_data
                    for i, b in enumerate(rb):
                        # The shift mirrors the interior on the same side
                        shift = xmax - cells[ihi - (i + 1)].x()
                        b.set_x(xmax + shift)

                        b.set_value(idx, dirichlet_value)

            else:
                raise SFXM(
                    "Incorrect data specified for dictionary-type boundaries")
        else:
            raise SFXM("Boundary type not implemented")


def get_periodic_bcs(bc_dict: dict, d: Domain):
    """
    Extracts periodic boundary conditions from a boundary condition dictionary.

    Parameters
    ----------
    bc_dict : dict
        A dictionary where keys are component names and values are dictionaries of boundary conditions. 
    d : Domain
        An instance of the Domain class used to get the component index.

    Returns
    -------
    dict
        A dictionary where keys are component indices and values are lists of boundary sides 
        ('left', 'right') that have periodic boundary conditions. Only components with periodic 
        conditions are included in the output dictionary.
    """
    periodic_bcs = {}

    for component, bcs in bc_dict.items():
        idx = d.component_index(component)
        sides = []
        for side, bc_type in bcs.items():
            if bc_type == "periodic":
                sides.append(side)

        if sides:
            periodic_bcs[idx] = sides

    return periodic_bcs


def extend_band(band, dirs: list, i: int, d: Domain):
    """
    Extends a list of band indices to include points affected by periodic boundary conditions.
    It handles overflow points for periodic boundaries by adding indices from the left or right domain as needed.

    Parameters
    ----------
    band : list
        A list of current band indices to be extended.
    dirs : list
        A list of boundary directions ('left', 'right') that determine how to extend the band 
        based on periodic boundary conditions.
    i : int
        The current point index within the domain where the band is being extended.
    d : Domain
        An instance of the Domain class used to obtain boundary parameters such as `nb`, `ilo`, and `ihi`

    Returns
    -------
    list
        A new list containing the union of the original band and additional indices 
        affected by periodic boundary conditions.
    """
    nb_left = d.nb(btype.LEFT)
    nb_right = d.nb(btype.RIGHT)

    ilo = d.ilo()
    ihi = d.ihi()

    # Iterate over left and right BCs
    addl_points = []
    for dir in dirs:
        # If right is periodic, check how much overflow there is
        # and add the left domain points to the band
        if dir == btype_map[btype.RIGHT]:
            overflow = max(0, nb_right - (ihi - i))
            addl_points.extend(list(range(ilo, ilo + overflow)))
        # If left is periodic, check how much overflow there is
        # and add the right domain points to the band
        elif dir == btype_map[btype.LEFT]:
            overflow = max(0, nb_left - (i - ilo))
            addl_points.extend(list(range(ihi - overflow + 1, ihi + 1)))
        else:
            raise SFXM("Incorrect boundary direction encountered")

    return list(set(band).union(set(addl_points)))
