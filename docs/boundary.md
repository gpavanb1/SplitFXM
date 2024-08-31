# Boundary

A boundary is a special type of cell that is used to define the boundary conditions of the system.

Note that this does NOT inherit from the `Cell` class, but instead implements the `Boundary` interface.
This ensures that some of the interior cell methods are not unintentionally called on a boundary cell.

:::splitfxm.boundary