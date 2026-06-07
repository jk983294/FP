#include <fp_eigen.h>
#include <fp_opt.h>
#include <iostream>
#include <piqp/piqp.hpp>

namespace FP {
void FpOpt::handle_barra1() {
  /**
   * Factor model QP reformulation.
   *
   * In a Barra factor model the asset return covariance is
   *   Sigma = B * Sigma_f * B' + D
   * where
   *   B:       N x K factor exposure matrix
   *   Sigma_f: K x K factor covariance (dense, small)
   *   D:       N x N specific risk (diagonal)
   *
   * Instead of forming the full N x N Sigma, we introduce the
   * factor return f = B' x as auxiliary decision variables and
   * add the equality constraint  f = B' x.
   *
   * Original problem:
   *   min  (lambda/2) x' Sigma x  -  x' r
   *   s.t. constraints ...
   *
   * Since x' Sigma x = x' B Sigma_f B' x + x' D x
   *                  = f'  Sigma_f  f  + x' D x     with f = B'x
   *
   * Reformulated with z = [x; f],  dim = (N + K):
   *
   *   min  (1/2) z' P_aug z  +  c_aug' z
   *   s.t. [B'  -I] z = 0          (factor linking)
   *        sum(x) = 1              (fully invested)
   *        other constraints on x only
   *
   * where
   *   P_aug = [lambda*D       0    ]
   *           [0        lambda*Sigma_f]
   *
   *   c_aug = [-r;  0_K]
   *
   * The N x N specific-risk block and the K x K factor-cov block
   * are both tiny compared with the full N x N Sigma.
   */

  if (!m_factorModel) {
    throw std::runtime_error("Barra1 requires factor model: "
        "call set_factor_exposure, set_factor_cov, set_specific_risk first");
  }

  size_t K = m_nFactors;
  size_t new_n = m_n + K;  // z = [x(N), f(K)]

  // ---- P_aug (block diagonal) ----
  Eigen::MatrixXd P_aug = Eigen::MatrixXd::Zero(new_n, new_n);
  // upper-left: lambda * D (diagonal)
  for (size_t i = 0; i < m_n; i++) {
    P_aug(i, i) = m_riskAversion * m_D(i);
  }
  // lower-right: lambda * Sigma_f
  P_aug.block(m_n, m_n, K, K) = m_riskAversion * m_Fcov;

  // ---- c_aug ----
  Eigen::VectorXd c_aug = Eigen::VectorXd::Zero(new_n);
  c_aug.head(m_n) = m_c * (-1.0);  // maximize return
  // f part is zero

  // ---- Equality constraints ----
  // Row 0: sum(x) = 1
  // Rows 1..K: B'x - f = 0   =>  [B'  -I] z = 0
  // Existing m_A/m_b rows follow after
  size_t n_existing_eq = m_A.rows();
  size_t n_eq = 1 + K + n_existing_eq;
  Eigen::MatrixXd A_aug = Eigen::MatrixXd::Zero(n_eq, new_n);
  Eigen::VectorXd b_aug = Eigen::VectorXd::Zero(n_eq);

  // sum(x) = 1
  A_aug.row(0).head(m_n).setOnes();
  b_aug(0) = 1.0;

  // B'x - f = 0  =>  [B'  -I] z = 0
  A_aug.block(1, 0, K, m_n) = m_B.transpose();
  for (size_t k = 0; k < K; k++) {
    A_aug(1 + k, m_n + k) = -1.0;
  }
  // b_aug rows 1..K are already 0

  // preserve existing equality constraints, extended with zeros for f
  if (n_existing_eq > 0) {
    A_aug.block(1 + K, 0, n_existing_eq, m_n) = m_A;
    b_aug.segment(1 + K, n_existing_eq) = m_b;
  }

  // ---- Inequality constraints ----
  // Existing m_G constraints only involve x, extend with zeros for f
  size_t n_ineq = m_G.rows();
  Eigen::MatrixXd G_aug = Eigen::MatrixXd::Zero(n_ineq, new_n);
  if (n_ineq > 0) {
    G_aug.leftCols(m_n) = m_G;
  }

  // ---- Variable bounds ----
  // x: use ins weight bounds; f: unbounded
  double lb = std::isfinite(m_insMinWeight) ? m_insMinWeight : -2.0;
  double ub = std::isfinite(m_insMaxWeight) ? m_insMaxWeight : 2.0;
  Eigen::VectorXd x_lb_aug = Eigen::VectorXd::Constant(new_n, -1e6);
  Eigen::VectorXd x_ub_aug = Eigen::VectorXd::Constant(new_n, 1e6);
  x_lb_aug.head(m_n).setConstant(lb);
  x_ub_aug.head(m_n).setConstant(ub);

  // ---- Solve ----
  {
    piqp::DenseSolver<double> solver;
    solver.settings().verbose = m_verbose;
    solver.settings().compute_timings = m_verbose;
    solver.settings().max_iter = m_maxIter;

    solver.setup(P_aug, c_aug, A_aug, b_aug,
                 G_aug, m_lh, m_uh, x_lb_aug, x_ub_aug);

    piqp::Status status = solver.solve();

    m_status = status;
    m_iter = solver.result().info.iter;
    // extract only x portion
    m_result = solver.result().x.head(m_n);
  }

  // ---- Post-processing ----
  m_expected_ret = m_result.transpose()
      * Eigen::Map<Eigen::VectorXd>(m_y_hat.data(), m_n);

  // Variance via factor model: x'(B Sigma_f B' + D)x = f'Sigma_f f + x'D x
  Eigen::VectorXd f = m_B.transpose() * m_result;
  m_variance = f.transpose() * m_Fcov * f
             + m_result.cwiseProduct(m_D).dot(m_result);

  m_turnover = m_result.lpNorm<1>();

  if (m_verbose) {
    std::cout << "=== Barra1 Factor Model QP ===" << std::endl;
    std::cout << "N = " << m_n << ", K = " << K << std::endl;
    std::cout << "P_aug (diag): ";
    for (size_t i = 0; i < new_n; i++) std::cout << P_aug(i,i) << " ";
    std::cout << std::endl;
    std::cout << "c_aug = " << c_aug.transpose() << std::endl;
    std::cout << "A_aug :\n" << A_aug << std::endl;
    std::cout << "b_aug = " << b_aug.transpose() << std::endl;
    if (n_ineq > 0) {
      std::cout << "G_aug :\n" << G_aug << std::endl;
      std::cout << "lh = " << m_lh.transpose() << std::endl;
      std::cout << "uh = " << m_uh.transpose() << std::endl;
    }
    std::cout << "x_lb = " << x_lb_aug.transpose() << std::endl;
    std::cout << "x_ub = " << x_ub_aug.transpose() << std::endl;
    std::cout << "m_result = " << m_result.transpose() << std::endl;
    std::cout << "factor_exposure (B'x) = " << f.transpose() << std::endl;
    std::cout << "variance = " << m_variance << std::endl;
    tidy_info();
  }
  m_barra_pre_n = new_n;
}
}  // namespace FP
