# Generate tests for splitfxm.bc

import pytest
from splitfxm.bc import apply_BC
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

    # Test unsupported BCs
    with pytest.raises(SFXM):
        d = Domain.from_size(10, 1, 1, ["u", "v", "w"])
        set_initial_condition(d, "u", "gaussian")
        bcs = {"left": "unsupported", "right": "periodic"}
        apply_BC(d, "u", bcs)
