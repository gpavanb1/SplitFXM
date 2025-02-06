# SplitFXM

[![Downloads](https://pepy.tech/badge/splitfxm)](https://pepy.tech/project/splitfxm)

![img](images/logo.jpg)

1D [Finite-Difference](https://en.wikipedia.org/wiki/Finite_difference_method) or [Finite-Volume](https://en.wikipedia.org/wiki/Finite_volume_method) using asymmetric stencils with [adaptive mesh refinement](https://en.wikipedia.org/wiki/Adaptive_mesh_refinement) and steady-state solver using Newton and [Split-Newton](https://github.com/gpavanb1/SplitNewton) approach

## What does 'split' mean?

The system is divided into multiple segments, and for ease of communication, let’s refer to the first segment of variables as "outer" and the remaining as "inner".

* Holding the outer variables fixed, Newton iteration is performed recursively for the inner variables, using the sub-Jacobian associated with them, until convergence is reached.

* One Newton step is then performed for the outer variables, while the inner variables are kept fixed, using the sub-Jacobian for the outer subsystem.

* This process is repeated, alternating between solving the inner and outer subsystems, until the convergence criterion for the entire system (similar to standard Newton) is met.

Consider a system of 5 variables, with the split locations at indices [1, 4]. This results in the following segments:

  * `a1` (variables from 0 to 1)
  * `a2 a3 a4` (variables from 1 to 4)
  * `a5` (variable at index 4)

1. First, the innermost segment `a5` is solved recursively using Newton's method while holding the variables `a1` and `a2 a3 a4`) fixed. This step is repeated until the convergence criterion for `a5` is met.

2. Next, one Newton step is taken for the segment `a2 a3 a4`, with `a5` held fixed. This step is followed by solving `a5` again till convergence.

3. This alternating process repeats: solving for `a5` until convergence, then one step for `a2 a3 a4`, and so on, until all subsystems converge.

Finally, one Newton step is performed for `a1`, with the other segments fixed. This completes one cycle of the split Newton process.

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

## Citing

If you are using `SplitFXM` in any scientific work, please make sure to cite as follows
```
@software{pavan_b_govindaraju_2024_13882261,
  author       = {Pavan B Govindaraju},
  title        = {gpavanb1/SplitFXM: v0.4.0},
  month        = oct,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.0},
  doi          = {10.5281/zenodo.13882261},
  url          = {https://doi.org/10.5281/zenodo.13882261}
}
```
