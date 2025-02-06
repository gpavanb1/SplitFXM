# SplitFXM

[![Downloads](https://pepy.tech/badge/splitfxm)](https://pepy.tech/project/splitfxm)
![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13882261.svg)](https://doi.org/10.5281/zenodo.13882261)

![img](https://github.com/gpavanb1/SplitFXM/blob/main/assets/logo.jpg)

1D [Finite-Difference](https://en.wikipedia.org/wiki/Finite_difference_method) or [Finite-Volume](https://en.wikipedia.org/wiki/Finite_volume_method) using asymmetric stencils with [adaptive mesh refinement](https://en.wikipedia.org/wiki/Adaptive_mesh_refinement) and steady-state solver using Newton and [Split-Newton](https://github.com/gpavanb1/SplitNewton) approach

## What does 'split' mean?

The system is divided into multiple segments, and for ease of communication, let’s refer to the first segment of variables as "outer" and the remaining as "inner".

* Holding the outer variables fixed, Newton iteration is performed recursively for the inner variables, using the sub-Jacobian associated with them, until convergence is reached.

* One Newton step is then performed for the outer variables, while the inner variables are kept fixed, using the sub-Jacobian for the outer subsystem.

* This process is repeated, alternating between solving the inner and outer subsystems, until the convergence criterion for the entire system (similar to standard Newton) is met.

### Example:

Consider a system of 5 variables, with the split locations at indices [1, 4]. This results in the following segments:

  * `a1` (variables from 0 to 1)
  * `a2 a3 a4` (variables from 1 to 4)
  * `a5` (variable at index 4)

1. First, the innermost segment `a5` is solved recursively using Newton's method while holding the variables `a1` and `a2 a3 a4`) fixed. This step is repeated until the convergence criterion for `a5` is met.

2. Next, one Newton step is taken for the segment `a2 a3 a4`, with `a5` held fixed. This step is followed by solving `a5` again till convergence.

3. This alternating process repeats: solving for `a5` until convergence, then one step for `a2 a3 a4`, and so on, until all subsystems converge.

Finally, one Newton step is performed for `a1`, with the other segments fixed. This completes one cycle of the split Newton process.

## How to install and execute?

Just run 
```
pip install splitfxm
```

There is an [examples](https://github.com/gpavanb1/SplitFXM/tree/main/splitfxm/models) folder that contains a test model - [Advection-Diffusion](https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation)

You can define your own equations by simply creating a derived class from `Model` and adding to the `_equations` using existing or custom equations!

A basic driver program is as follows
```
from splitfxm.domain import Domain
from splitfxm.simulation import Simulation
from splitfxm.models.advection_diffusion import AdvectionDiffusion
from splitfxm.schemes import default_scheme
from splitfxm.visualize import draw
import matplotlib.pyplot as plt

# Define the problem
method = 'FVM'
m = AdvectionDiffusion(c=0.2, nu=0.001, method=method)
# nx, nb_left, nb_right, variables
d = Domain.from_size(20, 1, 1, ["u", "v", "w"])
ics = {"u": "gaussian", "v": "rarefaction", "w": "tophat"}
bcs = {
    "u": {
        "left": "periodic",
        "right": "periodic"
    },
    "v": {
        "left": {"dirichlet": 3},
        "right": {"dirichlet": 4}
    },
    "w": {
        "left": {"dirichlet": 2},
        "right": "periodic"
    }
}
s = Simulation(d, m, ics, bcs, default_scheme(method))


# Advance in time or to steady state
s.evolve(t_diff=0.1)
bounds = [[-1., -2., -2.], [5., 4., 3.]]
iter = s.steady_state(split=True, split_locs=[1, 2], bounds=bounds)

# Visualize
draw(d, "label")
plt.show()
```

## How to build from source?

Since v0.4.0, SplitFXM utilizes Cython for accelerated computation. To build from source, you will need to install Cython and run the following command:
```
python setup.py build_ext --inplace
```

## Run benchmark
There is a benchmark that is included, which compares the time it takes to generate both a sparse and dense Jacobian. The results are as follows:

For N=250, 

| Method    | Time       | 
|-----------|------------|
| Dense   |    45 seconds |
| Sparse |  ~0.6 seconds  |

The benchmark can be executed from the parent folder using the command

`python -m pytest -s benchmark`

## How to run tests?

To run the tests, execute the following command from the parent folder:
```
python -m pytest tests
```

You can use the `-s` flag to show `print` outputs of the tests

## How to get coverage?

To get coverage, execute the following command from the parent folder:
```
python -m pytest --cov=splitfxm --cov-report <option> tests
```

The `option` can be related to showing covered/missed lines or specifying the output format of the report. For example, to get a line-by-line report, use the following command:
```
python -m pytest --cov=splitfxm --cov-report term-missing tests
```

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
