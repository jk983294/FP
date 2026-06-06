#include <fp_eigen.h>
#include <fp_opt.h>
#include <iostream>
#include <piqp/piqp.hpp>

namespace FP {
void FpOpt::handle_barra_max_sharpe() {
  if (m_P.rows() != (Eigen::Index)m_n || m_P.cols() != (Eigen::Index)m_n) {
    throw std::runtime_error("max sharpe: expected m_P to be set");
  }

  /**
   * Max Sharpe via Cornuejols-Tutuncu variable substitution
   * with explicit kappa for per-asset weight bounds.
   *
   * Augmented variable z = [y; kappa] where y = kappa * x, kappa = sum(y).
   *
   * min  (1/2) y' * Sigma * y
   * s.t. y'(mu - rf*1) = 1
   *      sum(y) = kappa
   *      kappa * m_insMinWeight <= y_i <= kappa * m_insMaxWeight
   *      kappa * lh_j <= g_j' * y <= kappa * uh_j  (factor constraints)
   *
   * Weight bounds and factor constraints become LINEAR in (y, kappa):
   *   y_i - kappa * ub <= 0
   *  -y_i + kappa * lb <= 0
   *   g_j' * y - uh_j * kappa <= 0
   *  -g_j' * y + lh_j * kappa <= 0
   *
   * After solve: x = y / kappa
   */
  size_t new_n = m_n + 1;

  // P_aug = [Σ  0]
  //         [0  0]
  Eigen::MatrixXd P_aug = Eigen::MatrixXd::Zero(new_n, new_n);
  P_aug.block(0, 0, m_n, m_n) = m_P;

  Eigen::VectorXd c_aug = Eigen::VectorXd::Zero(new_n);

  // --- Equality constraints ---
  Eigen::VectorXd excess_ret = m_c - Eigen::VectorXd::Constant(m_n, m_riskFreeRate);
  size_t n_eq = 2 + m_A.rows();
  Eigen::MatrixXd A_aug(n_eq, new_n);
  Eigen::VectorXd b_aug(n_eq);

  // Row 0: y'(mu - rf*1) = 1
  A_aug.row(0).setZero();
  A_aug.row(0).head(m_n) = excess_ret.transpose();
  b_aug(0) = 1.0;

  // Row 1: sum(y) - kappa = 0
  A_aug.row(1).setZero();
  A_aug.row(1).head(m_n).setOnes();
  A_aug(1, m_n) = -1.0;
  b_aug(1) = 0.0;

  // Translate existing equality constraints from add_constrain:
  //   a_j' * x = b_j  →  a_j' * y = b_j * kappa  →  [a_j, -b_j] * z = 0
  for (Eigen::Index j = 0; j < m_A.rows(); j++) {
    A_aug.row(2 + j).setZero();
    A_aug.row(2 + j).head(m_n) = m_A.row(j);
    A_aug(2 + j, m_n) = -m_b(j);
    b_aug(2 + j) = 0.0;
  }

  // --- Inequality constraints ---
  double lb = std::isfinite(m_insMinWeight) ? m_insMinWeight : -2.0;
  double ub = std::isfinite(m_insMaxWeight) ? m_insMaxWeight : 2.0;
  size_t n_factor_rows = m_G.rows();
  size_t n_ineq = m_n * 2 + n_factor_rows * 2;

  Eigen::MatrixXd G_aug = Eigen::MatrixXd::Zero(n_ineq, new_n);
  Eigen::VectorXd uh_aug = Eigen::VectorXd::Zero(n_ineq);

  // Weight bound inequalities:
  //   y_i - kappa * ub <= 0   (upper bound)
  //  -y_i + kappa * lb <= 0   (lower bound)
  for (size_t i = 0; i < m_n; i++) {
    G_aug(i, i) = 1.0;
    G_aug(i, m_n) = -ub;

    G_aug(m_n + i, i) = -1.0;
    G_aug(m_n + i, m_n) = lb;
  }

  // Translate factor constraints from add_constrain:
  //   lh_j <= g_j' * x <= uh_j
  //   →  g_j' * y - uh_j * kappa <= 0  (upper)
  //      -g_j' * y + lh_j * kappa <= 0  (lower)
  for (Eigen::Index j = 0; j < (Eigen::Index)n_factor_rows; j++) {
    size_t row_upper = m_n * 2 + j * 2;
    size_t row_lower = m_n * 2 + j * 2 + 1;

    G_aug.row(row_upper).head(m_n) = m_G.row(j);
    G_aug(row_upper, m_n) = -m_uh(j);

    G_aug.row(row_lower).head(m_n) = -m_G.row(j);
    G_aug(row_lower, m_n) = m_lh(j);
  }

  // --- Variable bounds ---
  Eigen::VectorXd x_lb_aug = Eigen::VectorXd::Constant(new_n, -100.0);
  Eigen::VectorXd x_ub_aug = Eigen::VectorXd::Constant(new_n, 100.0);
  x_lb_aug(m_n) = 1e-6; // kappa > 0

  // --- Solve ---
  {
    piqp::DenseSolver<double> solver;
    solver.settings().verbose = m_verbose;
    solver.settings().compute_timings = m_verbose;
    solver.settings().max_iter = m_maxIter;

    solver.setup(P_aug, c_aug, A_aug, b_aug, G_aug, std::nullopt, uh_aug, x_lb_aug, x_ub_aug);

    piqp::Status status = solver.solve();

    m_status = status;
    m_iter = solver.result().info.iter;

    // Recover portfolio weights: x = y / kappa
    double kappa = solver.result().x(m_n);
    if (kappa < 1e-12) {
      throw std::runtime_error("max sharpe: kappa too close to zero, no solution");
    }
    m_result = solver.result().x.head(m_n) / kappa;
  }

  m_expected_ret = m_result.transpose() * Eigen::Map<Eigen::VectorXd>(m_y_hat.data(), m_n);
  m_variance = m_result.transpose() * m_P * m_result;
  m_turnover = m_result.lpNorm<1>();

  if (m_verbose) {
    std::cout << "risk_free_rate = " << m_riskFreeRate << std::endl;
    std::cout << "P_aug :\n" << P_aug << std::endl;
    std::cout << "c_aug = " << c_aug.transpose() << std::endl;
    std::cout << "A_aug :\n" << A_aug << std::endl;
    std::cout << "b_aug = " << b_aug.transpose() << std::endl;
    std::cout << "G_aug :\n" << G_aug << std::endl;
    std::cout << "uh_aug = " << uh_aug.transpose() << std::endl;
    std::cout << "x_lb = " << x_lb_aug.transpose() << std::endl;
    std::cout << "x_ub = " << x_ub_aug.transpose() << std::endl;
    std::cout << "m_result = " << m_result.transpose() << std::endl;
    double sharpe = (m_expected_ret - m_riskFreeRate) / std::sqrt(m_variance);
    std::cout << "sharpe = " << sharpe << std::endl;
    tidy_info();
  }
  m_barra_pre_n = new_n;
}
}  // namespace FP
