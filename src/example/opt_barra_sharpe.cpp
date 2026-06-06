#include <fp_opt.h>
#include <getopt.h>
#include <iostream>
#include <limits>

int main(int argc, char** argv) {
  FP::FpOpt opt;
  opt.set_verbose(false);
  opt.set_type(FP::FpOptType::BarraSharpe);
  size_t nIns = 5;

  opt.set_maxIter(100);
  opt.set_size(nIns, false);
  opt.set_insMinWeight(0.0);   // long-only
  opt.set_insMaxWeight(0.3);
  opt.set_riskFreeRate(0.01);
  opt.set_benchWeights({0.1, 0.2, 0.1, 0.3, 0.4});
  opt.add_constrain({0.1, 0.2, 0.1, 0.3, 0.4}, -1.3, 1.3, true);
  opt.add_constrain({0.2, 0.1, 0.1, 0.4, 0.3}, -1.4, 1.4, true);

  // covariance matrix (5x5 symmetric positive definite)
  Eigen::MatrixXd cov(nIns, nIns);
  cov << 0.04, 0.006, 0.002, -0.001, -0.003,
         0.006, 0.09,  0.004, -0.002, -0.005,
         0.002, 0.004, 0.01,  0.001, -0.001,
        -0.001,-0.002, 0.001, 0.04,   0.008,
        -0.003,-0.005,-0.001, 0.008,  0.06;
  opt.set_covariance(cov);

  Eigen::VectorXd ret(nIns);
  ret << 0.07, 0.1, 0.03, -0.02, -0.03;

  opt.set_expected_return(ret);
  opt.solve();
  std::cout << "weight = " << opt.m_result.transpose() << std::endl;
  std::cout << "expected_return = " << opt.get_expected_return() << std::endl;
  std::cout << "variance = " << opt.get_variance() << std::endl;
  double sharpe = (opt.get_expected_return() - opt.m_riskFreeRate) / std::sqrt(opt.get_variance());
  std::cout << "sharpe = " << sharpe << std::endl;
  opt.clear_barra();
  return 0;
}
