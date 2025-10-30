#include <fp_opt.h>
#include <iostream>
#include <limits>
#include <getopt.h>

int main(int argc, char** argv) {
    FP::FpOpt opt;
    opt.set_type(FP::FpOptType::Barra);
    size_t nIns = 5;
    double max_tv = 0.3;
    // double max_tv = 0;

    opt.set_verbose(true);
    opt.set_maxIter(100);
    opt.set_size(nIns, false);
    opt.set_insMinWeight(0.0);
    opt.set_insMaxWeight(0.3);
    opt.set_benchWeights({0.1, 0.2, 0.1, 0.3, 0.4});
    if (max_tv > 1e-6) {
      opt.set_oldWeights({0.2, 0.2, 0.2, 0.2, 0.2});
    }
    opt.add_constrain({0.1, 0.2, 0.1, 0.3, 0.4}, -1.3, 1.3, true);
    opt.add_constrain({0.2, 0.1, 0.1, 0.4, 0.3}, -1.4, 1.4, true);

    Eigen::VectorXd ret(nIns);
    ret << 0.07, 0.1, 0.03, -0.02, -0.03;

    opt.set_expected_return(ret);
    opt.set_tvAversion(max_tv);
    opt.solve();
    return 0;
}

