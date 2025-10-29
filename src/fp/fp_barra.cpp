#include <fp_opt.h>
#include <fp_eigen.h>
#include <piqp/piqp.hpp>
#include <iostream>

namespace FP {
void FpOpt::handle_barra() {
    if (m_bIncludeCash) throw std::runtime_error("not support m_bIncludeCash, use m_cashWeight instead!");

    /**
     * min -w^t * r
     * s.t. w^t * i = 1.
     *      0. <= w <= m_insMaxWeight
     */
    m_P = Eigen::MatrixXd::Zero(m_n, m_n);
    Eigen::VectorXd _c = m_c * (-1.);

    m_A = Eigen::MatrixXd::Constant(1, m_n, 1.0);
    m_b = Eigen::VectorXd::Constant(1, 1.0 - m_cashWeight);
    add_ins_weight_constrain();
    auto _lh = Eigen::Map<const Eigen::VectorXd>(m_lh.data(), m_n);
    auto _uh = Eigen::Map<const Eigen::VectorXd>(m_uh.data(), m_n);

    piqp::DenseSolver<double> solver;
    solver.settings().verbose = m_verbose;
    solver.settings().compute_timings = m_verbose;
    solver.settings().max_iter = m_maxIter;

    solver.setup(m_P, _c, m_A, m_b, m_G, _lh, _uh, m_x_lb, m_x_ub);

    piqp::Status status = solver.solve();

    m_status = status;
    m_result = solver.result().x;
    if (m_covConstrain) {
        m_variance = m_result.transpose() * Eigen::Map<Eigen::MatrixXd>(m_orig_cov.data(), m_n, m_n) * m_result;
    }
    m_expected_ret = m_result.transpose() * Eigen::Map<Eigen::VectorXd>(m_y_hat.data(), m_n);
    if (m_tvConstrain) {
        Eigen::Map<Eigen::VectorXd> old_w(m_oldWeights.data(), m_n);
        m_expected_ret = m_result.transpose() * (m_c - m_tvAversion * old_w);
    }
    if (m_tvConstrain) {
        m_turnover = (m_result - Eigen::Map<Eigen::VectorXd>(m_oldWeights.data(), m_n)).lpNorm<1>();
    } else {
        m_turnover = m_result.lpNorm<1>();
    }

    if (m_verbose) {
        auto real_cov = Eigen::Map<Eigen::MatrixXd>(m_orig_cov.data(), m_n, m_n);
        std::cout << "P :\n" << m_P << std::endl;
        std::cout << "c = " << m_c.transpose() << std::endl;
        std::cout << "A :\n" << m_A << std::endl;
        std::cout << "b = " << m_b.transpose() << std::endl;
        if (m_G.rows() > 0) {
            std::cout << "G :\n" << m_G << std::endl;
            std::cout << "h = " << m_h.transpose() << std::endl;
        }
        if (m_covConstrain) std::cout << "corr :\n " << cov2corr(real_cov) << std::endl;
        std::cout << "weight bound [" << m_x_lb[0] << ", " << m_x_ub[0] << "]" << std::endl;
        tidy_info();
    }
}
}