# 2D G functions

gfunc2d is a package for calculating the joint age-metallicity probability density function (2D G function) of one or more stars. The code requires a grid of stellar models saved in the HDF5 format. An example of such a grid can be found here: https://lu.box.com/s/z3jyz9metpeg42o8dqx5dqd6phvlx190 (Last checked on October 21 2021).

To test if the code works, one can run `test_gfunc2d.py` in the test directory. The examples directory includes an example input file and run script showing how the code can be run and what output is produced.

In addition to the main functions, the package includes functions for handling the HDF5 model grids (`gridtools.py`), for producing synthetic samples of stars (`mksynth.py`), and for creating an HDF5 model grid (`mkgrid.py`). The `mkgrid.py` functions are designed for a specific input format. To make a grid of PARSEC models, one should be able to combine the `makePARSEC()` with models downloaded from http://stev.oapd.inaf.it/cgi-bin/cmd.
