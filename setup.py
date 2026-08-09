from setuptools import find_packages
from setuptools import setup

with open('README.md') as f:
  readme = f.read()

with open('LICENSE') as f:
  lic = f.read()

# Runtime dependencies only; test and optional-backend dependencies are in
# extras_require below (pip install tangent[tf,jax,symbolic,viz,test]).
install_requires = [
    'autograd>=1.2',
    'future',
    'gast>=0.6.0,<0.8.0',
    'numpy',
]

extras_require = {
    'test': ['pytest>=6.0'],
    # TensorFlow and JAX support. Wheels are platform-specific; on aarch64
    # (e.g. Grace/DGX Spark) TensorFlow installs as a CPU build.
    'tf': ['tensorflow>=2.16'],
    'jax': ['jax>=0.4.30'],
    # PyTorch support. Wheels are platform-specific; on aarch64 the default
    # PyPI wheel is a CPU build.
    'torch': ['torch>=2.0'],
    # Keras 3 support (backend-agnostic ops; requires one of the Keras
    # backends - tensorflow, jax or torch).
    'keras': ['keras>=3.0'],
    # Symbolic optimizations (tangent.optimizations.algebraic_simplification
    # and the straight-line coarsening prototype in
    # tangent.optimizations.coarsening)
    'symbolic': ['sympy>=1.12'],
    # Visualization tools (tangent.visualization)
    'viz': ['matplotlib>=3.7', 'networkx>=3.0'],
}
extras_require['all'] = sorted(
    dep for deps in extras_require.values() for dep in deps)

setup(
    name='tangent',
    version='0.1.9',
    description=('Automatic differentiation using source code transformation '
                 'for Python'),
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Google Inc.',
    author_email='alexbw@google.com',
    url='https://github.com/google/tangent',
    license=lic,
    packages=find_packages(exclude=('tests')),
    package_data={'': ['README.md', 'LICENSE']},
    keywords=[
        'autodiff', 'automatic-differentiation', 'machine-learning',
        'deep-learning'
    ],
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
