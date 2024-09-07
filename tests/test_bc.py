# Generate tests for splitfxm.bc

import pytest
from splitfxm.bc import apply_BC, extend_band, get_periodic_bcs
from splitfxm.constants import btype
from splitfxm.domain import Domain
from splitfxm.error import SFXM
from splitfxm.initialize import set_initial_condition


def test_apply_BC():
    # Test periodic BCs
    d = Domain.from_size(10, 1, 1, ["u", "v", "w"])
    idx = d.component_index("u")

    set_initial_condition(d, "u", "gaussian")
    bcs = {"left": "periodic", "right": "periodic"}
    apply_BC(d, "u", bcs)
    assert d.boundaries()[0][0].value(idx) == d.interior()[-1].value(idx)
    assert d.boundaries()[1][0].value(idx) == d.interior()[0].value(idx)

    # Test outflow BCs
    d = Domain.from_size(10, 1, 1, ["u", "v", "w"])
    set_initial_condition(d, "u", "gaussian")
    bcs = {"left": "outflow", "right": "outflow"}
    apply_BC(d, "u", bcs)

    # Check left outflow condition
    dy = d.interior()[1].value(idx) - d.interior()[0].value(idx)
    dx = d.interior()[1].x() - d.interior()[0].x()
    delta_x = d.boundaries()[0][0].x() - d.interior()[0].x()
    assert d.boundaries()[0][0].value(idx) == d.interior()[
        0].value(idx) + (dy/dx) * delta_x

    # Check right outflow condition
    dy = d.interior()[-1].value(idx) - d.interior()[-2].value(idx)
    dx = d.interior()[-1].x() - d.interior()[-2].x()
    delta_x = d.boundaries()[1][0].x() - d.interior()[-1].x()
    assert d.boundaries()[1][0].value(
        idx) == d.interior()[-1].value(idx) + (dy/dx) * delta_x

    # Test dictionary-based BCs
    d = Domain.from_size(10, 1, 1, ["u", "v", "w"])
    set_initial_condition(d, "u", "gaussian")
    bcs = {"left": {"dirichlet": 2.0}, "right": {"neumann": 1.0}}
    apply_BC(d, "u", bcs)

    # Check Dirichlet condition
    assert d.boundaries()[0][0].value(idx) == 2.0

    # Check Neumann condition
    difference = (d.boundaries()[1][0].value(
        idx) - d.interior()[-1].value(idx))
    cell_width = d.boundaries()[1][0].x() - d.interior()[-1].x()
    assert difference/cell_width == pytest.approx(1.0)

    # Now flip it and try again
    bcs = {"left": {"neumann": 2.0}, "right": {"dirichlet": 1.0}}
    apply_BC(d, "u", bcs)

    # Check Neumann condition
    difference = (d.interior()[0].value(idx) - d.boundaries()[0][0].value(
        idx))
    cell_width = d.interior()[0].x() - d.boundaries()[0][0].x()
    assert difference/cell_width == pytest.approx(2.0)

    # Check Dirichlet condition
    assert d.boundaries()[1][0].value(idx) == 1.0

    # Test unsupported BCs
    with pytest.raises(SFXM):
        d = Domain.from_size(10, 1, 1, ["u", "v", "w"])
        set_initial_condition(d, "u", "gaussian")
        bcs = {"left": "unsupported", "right": "periodic"}
        apply_BC(d, "u", bcs)


def test_get_periodic_bcs():
    d = Domain.from_size(10, 1, 1, ["u", "v", "w"])
    bcs = {
        "u": {"left": "periodic", "right": "periodic"},
        "v": {"left": "dirichlet", "right": "neumann"},
        "w": {"left": "periodic", "right": "outflow"}
    }

    periodic_bcs = get_periodic_bcs(bcs, d)

    assert periodic_bcs[d.component_index("u")] == ["left", "right"]
    assert periodic_bcs[d.component_index("w")] == ["left"]
    assert d.component_index("v") not in periodic_bcs


def test_extend_band():
    d = Domain.from_size(10, 1, 2, ["u", "v", "w"])
    nb_left = d.nb(btype.LEFT)
    nb_right = d.nb(btype.RIGHT)

    ilo = d.ilo()
    ihi = d.ihi()

    # Test extending band with left periodic boundary
    band = [ilo, ilo + 1]
    extended_band = extend_band(band, ["left"], ilo, d)
    expected_extension = list(range(ihi - nb_left + 1, ihi + 1))
    assert set(extended_band) == set(band + expected_extension)

    # Test extending band with right periodic boundary
    band = [ihi - 1, ihi]
    extended_band = extend_band(band, ["right"], ihi, d)
    expected_extension = list(range(ilo, ilo + nb_right))
    assert set(extended_band) == set(band + expected_extension)


def test_apply_BC_invalid_bc_type():
    # Test invalid boundary condition type
    with pytest.raises(SFXM):
        d = Domain.from_size(10, 1, 1, ["u", "v", "w"])
        set_initial_condition(d, "u", "gaussian")
        bcs = {"left": "invalid_bc_type", "right": "periodic"}
        apply_BC(d, "u", bcs)


def test_apply_BC_invalid_direction():
    # Test invalid boundary direction in dictionary-based BCs
    with pytest.raises(SFXM):
        d = Domain.from_size(10, 1, 1, ["u", "v", "w"])
        set_initial_condition(d, "u", "gaussian")
        bcs = {"left": {"dirichlet": 2.0},
               "invalid_direction": {"dirichlet": 1.0}}
        apply_BC(d, "u", bcs)


def test_apply_BC_invalid_data():
    # Test invalid boundary direction in dictionary-based BCs
    with pytest.raises(SFXM):
        d = Domain.from_size(10, 1, 1, ["u", "v", "w"])
        set_initial_condition(d, "u", "gaussian")
        bcs = {"left": {"dirichlet": 2.0},
               "right": {"invalid": 2.0}}
        apply_BC(d, "u", bcs)


def test_extend_band_invalid_direction():
    # Test invalid direction in extend_band function
    with pytest.raises(SFXM):
        d = Domain.from_size(10, 1, 2, ["u", "v", "w"])
        band = [0, 1]
        extend_band(band, ["invalid_direction"], 0, d)
