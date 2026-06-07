#include <fp_opt.h>
#include <iostream>
#include <limits>

/**
 * Demo: Barra1 factor model QP.
 *
 * Factor model:  Sigma = B * Sigma_f * B' + D
 *
 * Instead of forming the full N x N covariance, handle_barra1()
 * reformulates the QP using only the K x K factor covariance and
 * the diagonal specific risk, introducing f = B'x as auxiliary
 * variables.  This is much cheaper when K << N.
 *
 * Dummy data: 5 instruments, 2 Barra factors (e.g. Value, Momentum).
 */
int main(int argc, char** argv) {
  FP::FpOpt opt;
  opt.set_verbose(false);
  opt.set_type(FP::FpOptType::Barra1);

  size_t nIns = 5;
  size_t nFactors = 2;

  opt.set_maxIter(100);
  opt.set_size(nIns, false);
  opt.set_insMinWeight(0.0);
  opt.set_insMaxWeight(0.3);
  opt.set_riskAversion(1.0);

  // --- Factor exposure B (N x K) ---
  // Row i = factor loadings for instrument i
  // Factor 1 (Value)
  // Factor 2 (Momentum)
  Eigen::MatrixXd B(nIns, nFactors);
  B <<  0.8, -0.2,
        0.9,  0.7,
        0.1,  0.1,
       -0.6,  0.8,
       -0.7, -0.5;
  opt.set_factor_exposure(B);

  // --- Factor covariance Sigma_f (K x K) ---
  // Factors are mildly correlated
  Eigen::MatrixXd Fcov(nFactors, nFactors);
  Fcov << 0.04,  0.005,
          0.005, 0.09;
  opt.set_factor_cov(Fcov);

  // --- Specific risk D (N x 1, diagonal) ---
  // Idiosyncratic variance per instrument
  Eigen::VectorXd D(nIns);
  D << 0.01, 0.02, 0.015, 0.025, 0.03;
  opt.set_specific_risk(D);

  Eigen::VectorXd yhat(nIns);
  yhat << 0.07, 0.10, 0.03, -0.02, -0.03;
  opt.set_expected_return(yhat);

  opt.set_benchWeights({0.1, 0.2, 0.1, 0.3, 0.4});
  opt.add_constrain({0.8, 0.9, 0.1, -0.6, -0.7}, -0.3, 0.3, true);
  opt.add_constrain({-0.2, 0.7, 0.1, 0.8, -0.5}, -0.4, 0.4, true);

  opt.solve();

  std::cout << "\n=== Results ===" << std::endl;
  std::cout << "status  = " << opt.get_status() << std::endl;
  std::cout << "weights = " << opt.m_result.transpose() << std::endl;
  std::cout << "sum(w)  = " << opt.m_result.sum() << std::endl;
  std::cout << "exp_ret = " << opt.get_expected_return() << std::endl;
  std::cout << "variance= " << opt.get_variance() << std::endl;
  std::cout << "vol     = " << std::sqrt(opt.get_variance()) << std::endl;

  // Verify variance via full Sigma for sanity
  Eigen::MatrixXd Sigma = (B * Fcov * B.transpose())
                        + D.asDiagonal().toDenseMatrix();
  double var_full = opt.m_result.transpose() * Sigma * opt.m_result;
  std::cout << "var (full Sigma check) = " << var_full << std::endl;

  opt.clear_barra();
  return 0;
}
