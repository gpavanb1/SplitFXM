from enum import Enum

btype = Enum("btype", "LEFT RIGHT")
btype_map = {
    btype.LEFT: "left",
    btype.RIGHT: "right"
}
