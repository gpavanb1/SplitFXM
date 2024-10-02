# Finite-Difference Schemes

This section contains details related to the implementation of finite-difference schemes. Note that the solver can utilize any finite-difference scheme, including asymmetric stencils. Additionally, the derivative-related methods are implemented using Cython.

::: splitfxm.derivatives
    options:
      allow_inspection: false
      members:
        - Dx
        - D2x
        - dx
        - d2x
