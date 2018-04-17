# Streamline Mapping of Landscape Structure: slm  #

The **slm** package provides a set of tools to map landscape structure using topographic streamline tracing.
The main repository is located here; companion repos house [Jupyter notebooks](https:://github.com/cstarknyc/slmnb) and [DTM  data](https:://github.com/cstarknyc/slmdata) (lidar digital terrain model).

***proviso:*** *this is a work in progress *

   - [**slm** hub](https://cstarknyc.github.io/slm)
      - explains how **slm** works, linking to Jupyter notebook examples
      - provides documentation of the **Python** portion of the code
   - [OpenCL docs](https://cstarknyc.github.io/slm/base)
      - documents the OpenCL kernels and related functions used in **slm** 
      - generated with Doxygen 


**slm** has a code base founded on:
   - [Python 3](https://docs.python.org/3/)
      - development is with [version 3.6.x](https://docs.python.org/3/)
      - several packages are required, notably [Numba](http://numba.pydata.org/), which is used to accelerate preprocessing steps
   - [OpenCL](https://www.khronos.org/opencl/) 
      - accessed from Python using [PyOpenCL](https://documen.tician.de/pyopencl/index.html)
      - development is with [version 1.2](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/) on AMD and NVIDIA GPUs
   - [OCaml](https://ocaml.org/)
       - intended to be a fast replacement for the Python component of **slm**
       - works with common OpenCL code base
   
