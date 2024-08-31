# Sparse Jacobian Benchmark

There is a benchmark that is included (also written as a `pytest` test), which compares the time it takes to generate both a sparse and dense Jacobian. The results are as follows:

For N=250, 

| Method    | Time       | 
|-----------|------------|
| Dense   |    20 seconds |
| Sparse |  ~0.6 seconds  |

The benchmark can be executed from the parent folder using the command

`python -m pytest -s benchmark`

The sparsity pattern of the Jacobian is as follows. Both `split` parameter and boundary conditions lead to wrap-around and reshuffle of the sparse entries, which are correctly captured and match with the dense Jacobian entries. Note that the image is shown for N=20 for better representation. Also, 

![img](images/benchmark.jpg)