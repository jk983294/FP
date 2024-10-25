#!/bin/bash

cd FP
Rscript -e 'devtools::build(binary=T)'
R CMD INSTALL ../FP_1.0_R_x86_64-pc-linux-gnu.tar.gz