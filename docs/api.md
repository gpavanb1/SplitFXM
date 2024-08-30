# API Documentation

The first step to solving your system of equations is to implement a `Model`.
Define a class that inherits from it and assign `_equations` attribute for it.
An example is provided in the `examples` folder

::: splitfxm.model

You can construct a `Domain` object using the `from_size` class method

::: splitfxm.domain

Certain pre-defined profiles can be set for your variables using the `ics` argument

::: splitfxm.initialize


