git clone https://github.com/cvxgrp/scs.git --depth 1
cd scs
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/3rd/scs/ \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON \
      -DUSE_LAPACK=ON \
      ..
make
install