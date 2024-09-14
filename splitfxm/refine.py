import math
from .cell import Cell
from .domain import Domain
from .error import SFXM


eps = math.ulp(1.0)


class Refiner:
    """
    Class for refining grids using slope, curve, and prune.

    Raises
    ------
    SFXM
        If the `ratio` is less than 2.0, if `slope` or `curve` is not between 0.0 and 1.0, or if `prune` is not less than `curve` and `slope`.
    """

    def __init__(self):
        # Default values
        # Borrowed from Cantera
        self._ratio = 10.0
        self._slope = 0.8
        self._curve = 0.8
        # Negative prune factor disables it
        self._prune = -0.001

        # Maximum points in grid
        self._npmax = 1000

        # Minimum range span factor
        self._min_range = 0.01

        # Minimum grid spacing
        self._min_grid = 1e-10

    def set_criteria(self, ratio, slope, curve, prune):
        """
        Sets the refinement criteria.

        Parameters
        ----------
        ratio : float
            The desired refinement ratio.
        slope : float
            The slope tolerance.
        curve : float
            The curve tolerance.
        prune : float
            The prune tolerance.

        Raises
        ------
        SFXM
            If the `ratio` is less than 2.0, if `slope` or `curve` is not between 0.0 and 1.0, or if `prune` is not less than `curve` and `slope`.
        """

        if ratio < 2.0:
            raise SFXM(
                f"ratio must be greater than 2.0 ({ratio} was specified).")
        elif slope < 0.0 or slope > 1.0:
            raise SFXM(
                f"slope must be between 0.0 and 1.0 ({slope} was specified).")
        elif curve < 0.0 or curve > 1.0:
            raise SFXM(
                f"curve must be between 0.0 and 1.0 ({curve} was specified).")
        elif prune > curve or prune > slope:
            raise SFXM(
                f"prune must be less than 'curve' and 'slope' ({prune} was specified)."
            )

        self._ratio = ratio
        self._slope = slope
        self._curve = curve
        self._prune = prune

    def set_max_points(self, npmax):
        """
        Sets the maximum number of points in the grid.

        Parameters
        ----------
        npmax : int
            Maximum number of points in the grid.

        """
        self._npmax = npmax

    def refine(self, d: Domain):
        """
        Refines the grid using given criteria.

        Parameters
        ----------
        d : Domain
            Domain object to be refined.

        Raises
        ------
        SFXM
            If maximum number of points in the grid is exceeded.
        """

        # https://cantera.org/documentation/docs-2.5/doxygen/html/dd/d3c/refine_8cpp_source.html
        # Using only slope, curve and prune
        cells = d.interior()
        n = len(cells)

        # Keep map
        # 1 means cell stays and -1 means it goes
        # Loc map
        # 1 means add a point there
        # c map
        # Addition due to that variable
        keep = {}
        c = {}
        loc = {}

        # Preserve border points
        keep[0] = 1
        keep[n - 1] = 1

        if len(cells) > self._npmax:
            raise SFXM("Exceeded maximum number of points")

        z = [cells[i].x() for i in range(n)]
        dz = [cells[i + 1].x() - cells[i].x() for i in range(n - 1)]
        # nv -> Number of variables
        nv = len(cells[1].values())

        for i in range(nv):
            name = d.component_name(i)

            # Get components at all points
            v = [cells[j].value(i) for j in range(n)]

            # Slopes (s) for component i
            s = [
                (cells[j + 1].value(i) - cells[j].value(i)) / (z[j + 1] - z[j])
                for j in range(n - 1)
            ]

            # Range of values
            vmin = min(v)
            vmax = max(v)
            # Range of slopes
            smin = min(s)
            smax = max(s)

            # Max absolute values of values and slopes
            aa = max(abs(vmin), abs(vmax))
            ss = max(abs(smin), abs(smax))

            # refine based on component i only if the range of v is
            # greater than a fraction 'min_range' of max |v|. This
            # eliminates components that consist of small fluctuations
            # on a constant background.
            if (vmax - vmin) > self._min_range * aa:
                # maximum allowable difference in value between adjacent
                # points.
                dmax = self._slope * (vmax - vmin) + eps
                for j in range(n - 1):
                    r = abs(v[j + 1] - v[j]) / dmax
                    if r > 1.0 and dz[j] >= 2 * self._min_grid:
                        loc[j] = 1
                        c[name] = 1

                    if r >= self._prune:
                        keep[j] = 1
                        keep[j + 1] = 1
                    elif j not in keep:
                        keep[j] = -1

            # refine based on the slope of component i only if the
            # range of s is greater than a fraction 'min_range' of max
            # |s|. This eliminates components that consist of small
            # fluctuations on a constant slope background.
            if (smax - smin) > self._min_range * ss:
                # maximum allowable difference in slope between
                # adjacent points
                dmax = self._curve * (smax - smin)
                for j in range(n - 2):
                    r = abs(s[j + 1] - s[j]) / (dmax + (eps / dz[j]))
                    if (
                        r > 1.0
                        and dz[j] >= 2 * self._min_grid
                        and dz[j + 1] >= 2 * self._min_grid
                    ):
                        c[name] = 1
                        loc[j] = 1
                        loc[j + 1] = 1

                    if r >= self._prune:
                        keep[j + 1] = 1
                    elif (j + 1) not in keep:
                        keep[j + 1] = -1

        # Refine based on properties of the grid itself
        for j in range(1, n - 1):
            # Add a new point if the ratio with left interval is too large
            if dz[j] > self._ratio * dz[j - 1]:
                loc[j] = 1
                c[f"point {j}"] = 1
                keep[j - 1] = 1
                keep[j] = 1
                keep[j + 1] = 1
                keep[j + 2] = 1

            # Add a point if the ratio with right interval is too large
            if dz[j] < dz[j - 1] / self._ratio:
                loc[j - 1] = 1
                c[f"point {max(j-1, 0)}"] = 1
                keep[j - 2] = 1
                keep[j - 1] = 1
                keep[j] = 1
                keep[j + 1] = 1

            # Keep the point if removing would make the ratio with the left
            # interval too large.
            if j > 1 and z[j + 1] - z[j - 1] > self._ratio * dz[j - 2]:
                keep[j] = 1

            # Keep the point if removing would make the ratio with the right
            # interval too large.
            if j < n - 2 and z[j + 1] - z[j - 1] > self._ratio * dz[j + 1]:
                keep[j] = 1

        # Don't allow pruning to remove multiple adjacent grid points
        # in a single pass.
        for j in range(2, n - 1):
            if j in keep and j - 1 in keep and keep[j] == -1 and keep[j - 1] == -1:
                keep[j] = 1

        # Finalize AMR changes
        self.show_changes(loc, c, keep)
        self.perform_changes(d, loc, keep)

    def perform_changes(self, d, loc, keep):
        """
        Perform changes to the domain d according to the changes specified in
        loc and keep.

        Parameters
        ----------
        d : Domain
            The domain to perform changes on.
        loc : Dict[int, int]
            A dictionary with locations where insertions are to happen
        keep : Dict[int, int]
            A dictionary that indicates whether a cell is to be kept or deleted. -1 indicates deletion and 1 indicates preservation
        """
        #######
        # AMR
        # Need to mark for deletion before deleting
        # as cell addition indices need to make sense
        #######
        cells = d.interior()
        # Iterate over keep and remove points
        for i, cell in enumerate(cells):
            if i in keep and keep[i] == -1:
                cell.to_delete = True

        # Add cells at loc
        # Create separate list of cells and merge
        to_merge = []
        for i in loc.keys():
            x = 0.5 * (cells[i + 1].x() + cells[i].x())
            value = 0.5 * (cells[i + 1].values() + cells[i].values())
            to_merge.append(Cell(x, value))

        # Delete marked cells
        for i, cell in enumerate(cells):
            if cell.to_delete:
                del cells[i]

        # Merge cells and sort by x
        cells += to_merge
        cells.sort()

        # Set interior to cells
        d.set_interior(cells)

    def show_changes(self, loc, c, keep):
        """
        Print information about changes made to the domain.
        """
        print("#" * 78)
        # Show additions
        if len(loc) != 0:
            print("Refining grid...")
            print("New points inserted after grid points ")
            for i in loc.keys():
                print(i, end=" ")
            print("    to resolve ", end="")
            for name in c.keys():
                print(name, end=" ")
            print("")
        else:
            print("No new points needed")

        # Show deletions
        num_deleted = list(keep.values()).count(-1)
        if num_deleted != 0:
            print("Deleted points at ")
            for i in keep.keys():
                if keep[i] == -1:
                    print(i, end=" ")
            print("")
        else:
            print("No new points deleted")
        print("#" * 78)
