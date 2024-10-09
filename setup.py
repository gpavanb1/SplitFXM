from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

with open("README.md", "r") as fh:
    long_description = fh.read()

extensions = [
    Extension(
        name="splitfxm.derivatives",
        sources=["splitfxm/derivatives.pyx"],
        include_dirs=[np.get_include()],  # Include NumPy headers
        # Disable deprecated API warnings
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    ),
    Extension(
        "splitfxm.flux",  # Assuming `flux.pyx` is in `splitfxm` package
        sources=["splitfxm/flux.pyx"],
        include_dirs=[np.get_include()],
        # Disable deprecated API warnings
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    )
]

# Common compile arguments for all extensions
# Disable fallthrough warnings
common_compile_args = ['-Wno-unreachable-code-fallthrough', '-fopenmp']
common_link_args = ['-fopenmp']
# Apply common compile arguments globally
for ext in extensions:
    ext.extra_compile_args = common_compile_args
    ext.extra_link_args = common_link_args

setup(
    name="SplitFXM",
    version="0.4.5",
    description="1D Finite-Difference/Volume Split Newton Solver",
    url="https://github.com/gpavanb1/SplitFXM",
    author="gpavanb1",
    author_email="gpavanb@gmail.com",
    license="CC BY-NC 4.0 for non-commercial use, commercial license available",
    packages=["splitfxm", "splitfxm.equations", "splitfxm.models"],
    include_package_data=True,
    install_requires=["numpy", "numdifftools", "matplotlib", "splitnewton"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="amr newton python finite-difference armijo optimization pseudotransient splitting",
    project_urls={  # Optional
        "Bug Reports": "https://github.com/gpavanb1/SplitFXM/issues",
        "Source": "https://github.com/gpavanb1/SplitFXM/",
    },
    zip_safe=False,
    ext_modules=cythonize(extensions, compiler_directives={
                          'language_level': "3"}),
)
