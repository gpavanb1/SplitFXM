# SplitFXM

[![Downloads](https://pepy.tech/badge/splitfxm)](https://pepy.tech/project/splitfxm)

![img](images/logo.jpg)

1D [Finite-Difference](https://en.wikipedia.org/wiki/Finite_difference_method) or [Finite-Volume](https://en.wikipedia.org/wiki/Finite_volume_method) with [adaptive mesh refinement](https://en.wikipedia.org/wiki/Adaptive_mesh_refinement) and steady-state solver using Newton and [Split-Newton](https://github.com/gpavanb1/SplitNewton) approach

## What does 'split' mean?

The system is divided into two and for ease of communication, let's refer to first set of variables as "outer" and the second as "inner".

* Holding the outer variables fixed, Newton iteration is performed till convergence using the sub-Jacobian

* One Newton step is performed for the outer variables with inner held fixed (using its sub-Jacobian)

* This process is repeated till convergence criterion is met for the full system (same as in Newton)

## Why SplitFXM?

The combination of adaptive mesh refinement+multiple boundary conditions+various finite-difference/finite-volume schemes in 1D is crucial for accurately capturing steep gradients and complex phenomena in various physical systems. A [SplitNewton](http://github.com/gpavanb1/SplitNewton) solver further increases robustness by efficiently handling non-linearities and stiff reactions, ensuring convergence in challenging scenarios.

Some of the applications where challenging 1D problems exist include:

- **Compressible Flows**: Shock waves, boundary layers.
- **Flamelet Problems**: Premixed and non-premixed combustion.
- **Batteries**: Electrochemical reactions, solid-electrolyte interfaces. 
- **Phase Changes**: Solidification, melting fronts.
- **Heat Transfer**: Thermal gradients in thin layers.
- **Chemical Kinetics**: Reaction fronts, ignition processes.
- **Acoustics**: Wave propagation in media with varying density.
- **Plasma Physics**: Sheaths, boundary layers in plasma.
- **Magnetohydrodynamics (MHD)**: Magnetic reconnection, shock structures.
 

## GitHub Repository

The link to the repository can be found [here](http://github.com/gpavanb1/SplitFXM)

## Whom to contact?

Please direct your queries to [gpavanb1](http://github.com/gpavanb1)
for any questions.

## Acknowledgements

Special thanks to [Cantera](https://github.com/Cantera/cantera) and [WENO-Scalar](https://github.com/comp-physics/WENO-scalar) for serving as an inspiration for code architecture.
