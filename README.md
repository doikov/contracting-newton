# contracting-newton
Implementation of Contracting Newton method for minimizing Logistic Regression model.

To plot the graphs from the paper 
"Convex optimization based on global lower second-order models" 
by N. Doikov and Yu. Nesterov
(https://arxiv.org/abs/2006.08518) 
do the following.

1. Compile the C++ code, using the command:
$ g++ main.cc optimize.cc -O2 -std=c++17 -o main

2. Download the files:
"data/w8a.txt",
"data/covtype_bin_sc"
from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/.

3. To run the basic (exact) methods, use
$ python3 demo_basic.py

4. To run the stochastic methods, use 
$ python3 demo_stochastic.py

The graphs will be placed into "output/*".
