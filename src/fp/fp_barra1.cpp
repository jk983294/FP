#include <fp_eigen.h>
#include <fp_opt.h>
#include <iostream>
#include <piqp/piqp.hpp>

namespace FP {
void FpOpt::handle_barra1() {
  /**
   * min -x' * r
   * s.t. x' * i = 1.
   *      0. <= x <= m_insMaxWeight
   */
  m_P = Eigen::MatrixXd::Zero(m_n, m_n);
  size_t new_n = m_n;
  Eigen::VectorXd _c = m_c * (-1.);

  {
    Eigen::MatrixXd tmp_ = Eigen::MatrixXd::Constant(1, m_n, 1.0);
    append(m_A, tmp_, true);
    append(m_b, 1.0);
  }

  add_ins_weight_constrain();

  {
    piqp::DenseSolver<double> solver;
    solver.settings().verbose = m_verbose;
    solver.settings().compute_timings = m_verbose;
    solver.settings().max_iter = m_maxIter;

    solver.setup(m_P, _c, m_A, m_b, m_G, m_lh, m_uh, m_x_lb, m_x_ub);

    piqp::Status status = solver.solve();

    m_status = status;
    m_result = solver.result().x;
    m_iter = solver.result().info.iter;
  }

  m_expected_ret = m_result.transpose() * Eigen::Map<Eigen::VectorXd>(m_y_hat.data(), m_n);
  m_turnover = m_result.lpNorm<1>();

  if (m_verbose) {
    std::cout << "P :\n" << m_P << std::endl;
    std::cout << "c = " << m_c.transpose() << std::endl;
    std::cout << "A :\n" << m_A << std::endl;
    std::cout << "b = " << m_b.transpose() << std::endl;
    std::cout << "G :\n" << m_G << std::endl;
    std::cout << "h_l = " << m_lh.transpose() << std::endl;
    std::cout << "h_u = " << m_uh.transpose() << std::endl;
    std::cout << "x_lb = " << m_x_lb.transpose() << std::endl;
    std::cout << "x_ub = " << m_x_ub.transpose() << std::endl;
    std::cout << "m_result = " << m_result.transpose() << std::endl;
    tidy_info();
  }
  m_barra_pre_n = new_n;
}
}  // namespace FP