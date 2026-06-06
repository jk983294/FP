#include <fp_opt.h>
#include <getopt.h>
#include <iostream>
#include <limits>

int main(int argc, char** argv) {
  FP::FpOpt opt;
  opt.set_verbose(false);
  opt.set_type(FP::FpOptType::Barra1);
  size_t nIns = 5;

  opt.set_maxIter(100);
  opt.set_size(nIns, false);
  opt.set_insMinWeight(0.0);
  opt.set_insMaxWeight(0.3);
  opt.set_benchWeights({0.1, 0.2, 0.1, 0.3, 0.4});
  opt.set_tvAversion(0);
  opt.add_constrain({0.1, 0.2, 0.1, 0.3, 0.4}, -1.3, 1.3, true);
  opt.add_constrain({0.2, 0.1, 0.1, 0.4, 0.3}, -1.4, 1.4, true);

  Eigen::VectorXd ret(nIns);
  ret << 0.07, 0.1, 0.03, -0.02, -0.03;

  opt.set_expected_return(ret);
  opt.solve();
  std::cout << "m_result = " << opt.m_result.transpose() << std::endl;
  opt.clear_barra();
  return 0;
}
