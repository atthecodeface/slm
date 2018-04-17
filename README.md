# Streamline Mapping of Landscape Structure: slm  #

The **slm** package provides a set of tools to map landscape structure using topographic streamline tracing. These tools are implemented in Python, OpenCL and OCaml.

***proviso:*** *this is a work in progress *

## What **slm** can do

Capabilities available now or anticipated soon include:
   - mapping channels and identifying the locations of channel heads
   - visualization of patterns of topographic surface flow
   - measuring hillslope lengths across a DTM landscape
   - flow routing over pit-prone and divergent topography such as alluvial fans
   - GPU accelerated processing of large DTM data sets
 
 Longer-term applications include:
   - kinematic mapping of surface water flow depth
      - the main goal is to estimate channel inundation and flow geometry in lidar DTMs
      - contrasts with typical GIS methods of DTM flow routing which have no sense of channel flow geometry
   - deployment of these methods in a landscape evolution model
      - will be able to resolve hillslope-channel transitions and approximate channel flow geometry
      - speed will be a challenge

## Documentation

   - [**slm** hub](https://cstarknyc.github.io/slm)
      - core documentation of **slm** idea, implementation and example results
      - links to Jupyter notebook demos
      - documents the **Python** portion of the code
   - [OpenCL docs](https://cstarknyc.github.io/slm/base)
      - documents the OpenCL kernels and related functions used in **slm** 
      - generated with Doxygen 
   - [OCaml docs](https://cstarknyc.github.io/slm/ocaml)
      - not yet implemented
      - will document the OCaml portion of **slm**

## Code base

**slm** has a code base founded on:
   - [Python 3](https://docs.python.org/3/)
      - development is with [version 3.6.x](https://docs.python.org/3/)
      - several packages are required, notably [Numba](http://numba.pydata.org/), which is used to accelerate preprocessing steps
   - [OpenCL](https://www.khronos.org/opencl/) 
      - accessed from Python using [PyOpenCL](https://documen.tician.de/pyopencl/index.html)
      - development is with [version 1.2](https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/) on AMD and NVIDIA GPUs
   - [OCaml](https://ocaml.org/)
       - intended to be a fast replacement for the Python component of **slm**
       - porting currently underway
       - will work with common OpenCL code base
   
The main repository is located here. Companion repos house [Jupyter notebooks](https://github.com/cstarknyc/slmnb) and [DTM  data](https://github.com/cstarknyc/slmdata) (lidar digital terrain model).

