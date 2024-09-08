# Import the constants from constants.py (adjust the import path as necessary)
from splitfxm.constants import btype, btype_map


def test_btype_enum():
    assert btype.LEFT.name == "LEFT"
    assert btype.RIGHT.name == "RIGHT"


def test_btype_map():
    assert btype_map[btype.LEFT] == "left"
    assert btype_map[btype.RIGHT] == "right"
