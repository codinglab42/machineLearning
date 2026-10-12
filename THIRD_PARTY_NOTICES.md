# Third-party notices

This project uses the following third-party components. Their licenses apply
to those components and are not replaced by the license for this project.

## Included in compiled artifacts

- **Eigen** (Eigen3, version 3.3 or later): its MPL-2.0 license text is in
  [`licenses/Eigen-MPL-2.0.txt`](licenses/Eigen-MPL-2.0.txt). Project targets
  define `EIGEN_MPL2_ONLY` so Eigen rejects inclusion of its non-MPL2 code.
- **pybind11** (version 2.10 or later): its BSD 3-Clause license text is in
  [`licenses/pybind11-BSD-3-Clause.txt`](licenses/pybind11-BSD-3-Clause.txt).
  pybind11 is used to build the Python extension; it is not vendored in this
  repository.

## Dependencies installed separately

- **NumPy** (version 1.20 or later) is a Python runtime dependency, not bundled
  with this project. Its license and notices are provided by the NumPy
  distribution installed by the package manager.
- **GoogleTest** is used only to build and run the C++ tests. It is not part of
  the installed library or Python extension.
- **pytest** is used only to run the Python tests. It is not part of the
  installed library or Python extension.

The build may use system-provided versions of Eigen and GoogleTest. Consult
those distributions for their exact version and corresponding notices.
