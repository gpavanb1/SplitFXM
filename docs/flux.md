# Finite-Volume Schemes

This section contains details related to the implementation of finite-volume schemes. Note that the flux-related methods are implemented using Cython.

::: splitfxm.flux
    options:
        allow_inspection: false
        members:
            - fluxes
            - diffusion_fluxes

### Solving Equations in Characteristic Space

To solve hyperbolic systems of equations, we use an upwind scheme in characteristic space. The process begins by computing the Jacobian matrix \( \mathbf{A} = \frac{\partial \mathbf{F}}{\partial \mathbf{U}} \), where \( \mathbf{F} \) is the flux function and \( \mathbf{U} \) represents the conservative variables. The Jacobian \( \mathbf{A} \) is evaluated at the central state \( \mathbf{U}_c \).

#### Step 1: Eigenvalue Decomposition

Next, the Jacobian matrix \( \mathbf{A} \) is decomposed into its eigenvalues and eigenvectors:

\[
\mathbf{A} = \mathbf{R} \mathbf{\Lambda} \mathbf{R}^{-1}
\]

where \( \mathbf{\Lambda} \) is the diagonal matrix of eigenvalues \( \lambda_i \), and \( \mathbf{R} \) is the matrix of right eigenvectors.

#### Step 2: Transformation to Characteristic Variables

The conservative variables \( \mathbf{U}_l \) (left state), \( \mathbf{U}_c \) (central state), and \( \mathbf{U}_r \) (right state) are transformed into characteristic variables using the inverse of the eigenvector matrix:

\[
\mathbf{W}_l = \mathbf{R}^{-1} \mathbf{U}_l, \quad \mathbf{W}_c = \mathbf{R}^{-1} \mathbf{U}_c, \quad \mathbf{W}_r = \mathbf{R}^{-1} \mathbf{U}_r
\]

where \( \mathbf{W} \) represents the characteristic variables.

#### Step 3: Scheme Application - Upwind Fluxes in Characteristic Space

Once the characteristic variables are computed, the fluxes are computed using any specified scheme. For the sake of disucssion, we will use the upwind scheme.

For each characteristic, the direction of the wave is determined by the sign of the corresponding eigenvalue \( \lambda_i \):

- If \( \lambda_i > 0 \), the wave moves to the right, and the flux at the west (left) side is determined using the left state, while the flux at the east (right) side is determined using the central state. This corresponds to the 'upwind' direction.
- If \( \lambda_i < 0 \), the wave moves to the left, and the flux at the west side is determined using the central state, while the flux at the east side is determined using the right state. As the wave moves to the left in this case, the upwind direction is to the right.

The characteristic fluxes are computed as:

\[
\mathbf{F}_w^\text{char} = \mathbf{F}(\mathbf{W}_l) \quad \text{if} \quad \lambda_i > 0, \quad \mathbf{F}_w^\text{char} = \mathbf{F}(\mathbf{W}_c) \quad \text{if} \quad \lambda_i < 0
\]
\[
\mathbf{F}_e^\text{char} = \mathbf{F}(\mathbf{W}_c) \quad \text{if} \quad \lambda_i > 0, \quad \mathbf{F}_e^\text{char} = \mathbf{F}(\mathbf{W}_r) \quad \text{if} \quad \lambda_i < 0
\]

#### Step 4: Transformation Back to Conservative Variables

After computing the fluxes in characteristic space, they are transformed back to the original conservative variable space using the eigenvector matrix \( \mathbf{R} \):

\[
\mathbf{F}_w = \mathbf{R} \mathbf{F}_w^\text{char}, \quad \mathbf{F}_e = \mathbf{R} \mathbf{F}_e^\text{char}
\]

Thus, the west (left) and east (right) fluxes \( \mathbf{F}_w \) and \( \mathbf{F}_e \) are obtained in the original space.

This method allows the fluxes to be accurately computed by considering wave propagation direction, ensuring stability and correctness in solving hyperbolic systems.
