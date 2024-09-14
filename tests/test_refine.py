import pytest
import math

from splitfxm.domain import Domain
from splitfxm.error import SFXM
from splitfxm.refine import Refiner


# Tolerance for floating point comparison
eps = math.ulp(1.0)


@pytest.fixture
def domain():
    """Fixture to create a domain."""
    nx = 5
    nb_left = 1
    nb_right = 1
    components = ["comp1"]
    d = Domain.from_size(nx, nb_left, nb_right, components)

    # Manually set values to the cells for testing
    for i, cell in enumerate(d.interior()):
        cell.set_value(0, i * 0.1)

    return d


def test_refiner_initialization():
    """Test Refiner initialization and default values."""
    refiner = Refiner()
    assert refiner._ratio == 10.0
    assert refiner._slope == 0.8
    assert refiner._curve == 0.8
    assert refiner._prune == -0.001
    assert refiner._npmax == 1000
    assert refiner._min_range == 0.01
    assert refiner._min_grid == 1e-10


def test_set_criteria_valid():
    """Test setting valid criteria."""
    refiner = Refiner()
    refiner.set_criteria(2.5, 0.5, 0.3, 0.2)
    assert refiner._ratio == 2.5
    assert refiner._slope == 0.5
    assert refiner._curve == 0.3
    assert refiner._prune == 0.2


def test_set_criteria_invalid():
    """Test invalid values for set_criteria."""
    refiner = Refiner()

    # Ratio less than 2.0
    with pytest.raises(SFXM):
        refiner.set_criteria(1.5, 0.5, 0.3, 0.2)

    # Slope out of range
    with pytest.raises(SFXM):
        refiner.set_criteria(2.5, 1.5, 0.3, 0.2)

    # Curve out of range
    with pytest.raises(SFXM):
        refiner.set_criteria(2.5, 0.5, 1.5, 0.2)

    # Prune greater than curve or slope
    with pytest.raises(SFXM):
        refiner.set_criteria(2.5, 0.5, 0.3, 0.6)


def test_set_max_points():
    """Test setting maximum number of points."""
    refiner = Refiner()
    refiner.set_max_points(500)
    assert refiner._npmax == 500


def test_refine_exceed_max_points(domain):
    """Test refining with domain exceeding maximum points."""
    refiner = Refiner()
    refiner.set_max_points(4)  # Set low limit to trigger exception

    with pytest.raises(SFXM):
        refiner.refine(domain)


def test_refine_domain(domain):
    """Test refining the domain with valid values."""
    refiner = Refiner()
    refiner.set_criteria(10.0, 0.5, 0.3, -0.001)  # Valid criteria

    # Refining should succeed without exceptions
    refiner.refine(domain)


def test_perform_changes(domain):
    """Test the perform_changes method."""
    refiner = Refiner()
    loc = {1: 1, 2: 1}
    keep = {0: 1, 1: -1, 2: 1}  # Keep and delete some cells
    refiner.perform_changes(domain, loc, keep)

    # Check that interior has been modified
    # Original was 5, one deleted, two added
    assert len(domain.interior()) == 6


def test_refine_slope_and_curve(domain):
    """Test refinement triggered by slope and curve conditions."""
    refiner = Refiner()
    refiner.set_criteria(3.0, 0.2, 0.2, -0.01)

    # Adjust cell values to create slopes and curves that should trigger refinement
    cells = domain.interior()
    cells[1].set_value(0, 0.9)
    cells[2].set_value(0, 0.95)

    refiner.refine(domain)

    # Verify that new points were inserted due to slope and curve
    # Ensure the grid has changed by checking the number of cells
    assert len(domain.interior()) > 5


def test_refine_ratio_trigger(domain):
    """Test refinement triggered by grid spacing ratio."""
    refiner = Refiner()
    refiner.set_criteria(2.0, 0.2, 0.2, -0.01)

    # Adjust domain grid spacing to trigger ratio-based refinement
    cells = domain.interior()
    cells[2]._x = cells[1].x() + 0.5 * (cells[2].x() - cells[1].x())

    refiner.refine(domain)

    # Verify that points were inserted due to ratio condition
    assert len(domain.interior()) > 5


def test_refiner_prune(capsys, domain):
    """Test refinement with pruning logic."""
    refiner = Refiner()

    # Set prune lower than slope/curve to trigger pruning
    prune_threshold = 0.1
    refiner.set_criteria(ratio=2.0, slope=0.8, curve=0.8,
                         prune=prune_threshold)

    # Adjust cell values to trigger pruning
    # Create small fluctuations in the middle of the domain
    cells = domain.interior()
    cells[2].set_value(0, 0.01)
    cells[3].set_value(0, 0.011)
    cells[4].set_value(0, 0.01)

    refiner.refine(domain)

    # Capture and check the output
    captured = capsys.readouterr()
    print(captured.out)
    assert "Deleted points at" in captured.out


def test_refiner_non_uniform_grid(domain):
    refiner = Refiner()

    # Set criteria for ratio-based refinement
    refiner.set_criteria(ratio=2.0, slope=0.5, curve=0.5, prune=0.0)

    cells = domain.interior()

    # Set non-uniform x-values to simulate a non-uniform grid
    # Use larger differences between some points to trigger the ratio-based conditions
    non_uniform_x = [0.0, 0.1, 0.3, 0.9, 1.5, 3.0, 6.0]
    for i, cell in enumerate(cells):
        # Accessing private variables only for test purpose
        # Cell doesn't have set_x
        if i < len(non_uniform_x):
            cell._x = non_uniform_x[i]

    initial_num_cells = len(cells)

    # Perform refinement
    refiner.refine(domain)

    refined_cells = domain.interior()
    final_num_cells = len(refined_cells)

    # Ensure new points were added based on the ratio criteria
    assert final_num_cells > initial_num_cells, "Expected new points to be added based on non-uniform grid refinement."

    # Do the same but in reverse order
    non_uniform_x = [0.0, 0.01, 0.1, 0.3, 0.9, 1, 1.1, 1.2]

    for i, cell in enumerate(cells):
        # Accessing private variables only for test purpose
        # Cell doesn't have set_x
        if i < len(non_uniform_x):
            cell._x = non_uniform_x[i]

    initial_num_cells = len(cells)

    # Perform refinement
    refiner.refine(domain)

    refined_cells = domain.interior()
    final_num_cells = len(refined_cells)

    # Ensure new points were added based on the ratio criteria
    assert final_num_cells > initial_num_cells, "Expected new points to be added based on non-uniform grid refinement."


def test_perform_changes(domain):
    """Test border point pruning."""
    refiner = Refiner()
    loc = {1: 1, 2: 1}
    keep = {0: 1, 1: -1, 2: 1}  # Keep and delete some cells
    refiner.perform_changes(domain, loc, keep)

    # Check that interior has been modified
    # Original was 5, one deleted, two added
    assert len(domain.interior()) == 6


def test_show_changes(capsys, domain):
    """Test the show_changes method for output correctness."""
    refiner = Refiner()
    refiner.set_criteria(3.0, 0.2, 0.2, -0.01)

    # Manually create loc and keep dictionaries for testing
    loc = {1: 1}
    c = {"comp1": 1}
    keep = {0: 1, 1: -1, 2: 1}

    refiner.show_changes(loc, c, keep)

    # Capture and check the output
    captured = capsys.readouterr()
    assert "New points inserted after grid points" in captured.out
    assert "Deleted points at" in captured.out
