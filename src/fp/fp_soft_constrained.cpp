#include <fp_opt.h>
#include <fp_eigen.h>
#include <piqp/piqp.hpp>
#include <iostream>

namespace FP {
void FpOpt::soft_tv_constrain() {
    if (!m_tvConstrain) return;

    { // c = (c + lambda2*old_w)
        if (m_oldWeights.empty()) {
            m_oldWeights.resize(m_n, 0);
        }
        Eigen::Map<Eigen::VectorXd> old_w(m_oldWeights.data(), m_n);
        // std::cout << "m_c :" << m_c.transpose() << std::endl;
        m_c = m_c + m_tvAversion * old_w;
        // std::cout << "m_c :" << m_c.transpose() << std::endl;
    }

    if (m_covConstrain) { // P = (P + lambda2/lambda1 * I)
        // std::cout << "m_P :\n" << m_P << std::endl;
        Eigen::VectorXd new_diag = m_P.diagonal();
        new_diag = new_diag + Eigen::VectorXd::Constant(new_diag.size(), m_tvAversion / m_riskAversion);
        // std::cout << "new_diag : " << new_diag.transpose() << std::endl;
        m_P.diagonal() = new_diag;
        // std::cout << "m_P :\n" << m_P << std::endl;
    }
}

void FpOpt::handle_SoftConstrained() {
    if (m_bIncludeCash) throw std::runtime_error("not support m_bIncludeCash, use m_cashWeight instead!");

    /**
     * min 1/2 w^t * (P + lambda2/lambda1 * I) * w - (1/lambda1) * w^t * (r + lambda2*old_w)
     * s.t. w^t * i = 1. - cash_weight
     *      0. <= w <= m_insMaxWeight
     */
    if (m_covConstrain) {
        m_P = Eigen::Map<Eigen::MatrixXd>(m_orig_cov.data(), m_n, m_n);
    } else if (m_tvConstrain) {
        m_P = Eigen::MatrixXd::Identity(m_n, m_n);
    } else {
        m_P = Eigen::MatrixXd::Zero(m_n, m_n);
    }
    
    m_A = Eigen::MatrixXd::Constant(1, m_n, 1.0);
    m_b = Eigen::VectorXd::Constant(1, 1.0 - m_cashWeight);
    add_ins_weight_constrain();
    soft_tv_constrain();
    Eigen::VectorXd _c;
    if (m_covConstrain) {
        _c = m_c * (-1. / m_riskAversion);
    } else if (m_tvConstrain) {
        _c = m_c * (-1. / m_tvAversion);
    } else {
        _c = m_c * (-1.);
    }

    piqp::DenseSolver<double> solver;
    solver.settings().verbose = m_verbose;
    solver.settings().compute_timings = m_verbose;
    solver.settings().max_iter = m_maxIter;
    if (m_G.rows() > 0) {
        solver.setup(m_P, _c, m_A, m_b, m_G, std::nullopt, m_uh, m_x_lb, m_x_ub);
    } else {
        solver.setup(m_P, _c, m_A, m_b, std::nullopt, std::nullopt, std::nullopt, m_x_lb, m_x_ub);
    }

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
    // if (status == piqp::Status::PIQP_SOLVED) {
    // }

    if (m_verbose) {
        auto real_cov = Eigen::Map<Eigen::MatrixXd>(m_orig_cov.data(), m_n, m_n);
        std::cout << "P :\n" << m_P << std::endl;
        std::cout << "c = " << m_c.transpose() << std::endl;
        std::cout << "A :\n" << m_A << std::endl;
        std::cout << "b = " << m_b.transpose() << std::endl;
        if (m_G.rows() > 0) {
            std::cout << "G :\n" << m_G << std::endl;
            std::cout << "h = " << m_uh.transpose() << std::endl;
        }
        if (m_covConstrain) std::cout << "corr :\n " << cov2corr(real_cov) << std::endl;
        std::cout << "weight bound [" << m_x_lb[0] << ", " << m_x_ub[0] << "]" << std::endl;
        tidy_info();
    }
}

void FpOpt::tidy_info() const {
    std::cout << "status = " << m_status << std::endl;
    std::cout << "lambda1 = " << m_riskAversion << " lambda2 = " << m_tvAversion << std::endl;
    if (m_tvConstrain) {
        auto vec = Eigen::Map<const Eigen::VectorXd>(m_oldWeights.data(), m_n);
        std::cout << "old weight = " << vec.transpose() << std::endl;
    }
    std::cout << "weight = " << m_result.transpose() << " cash=" << m_cashWeight << std::endl;
    std::cout << "portfolio variance = " << m_variance << " mean = " << m_expected_ret << " tv = " << m_turnover << std::endl;
}
}