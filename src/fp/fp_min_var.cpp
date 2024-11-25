#include <fp_opt.h>
#include <fp_eigen.h>
#include <piqp/piqp.hpp>
#include <iostream>

namespace FP {
void FpOpt::handle_MinimumVariance() {
    /**
     * min 1/2 w^t * P * w
     * s.t. w^t * i = 1.
     */
    m_c = Eigen::VectorXd::Constant(m_n, 0.0);
    m_A = Eigen::MatrixXd::Constant(1, m_n, 1.0);
    m_b = Eigen::VectorXd::Constant(1, 1.0);

    piqp::DenseSolver<double> solver;
    solver.settings().verbose = m_verbose;
    solver.settings().compute_timings = m_verbose;
    solver.setup(m_P, m_c, m_A, m_b);

    piqp::Status status = solver.solve();

    m_status = status;
    m_result = solver.result().x;
    if (status == piqp::Status::PIQP_SOLVED) {
        m_variance = m_result.transpose() * m_P * m_result;
    }

    if (m_verbose) {
        std::cout << m_P << std::endl;
        std::cout << m_A << std::endl;
        std::cout << m_b << std::endl;
        std::cout << "corr = " << cov2corr(m_P) << std::endl;
        std::cout << "status = " << m_status << std::endl;
        std::cout << "weight = " << m_result.transpose() << std::endl;
        std::cout << "portfolio variance = " << m_variance << std::endl;
    }
}

void FpOpt::handle_MeanVariance() {
    /**
     * min 1/2 w^t * P * w - (1/lambda) * w^t * r
     * s.t. w^t * i = 1.
     */
    m_A = Eigen::MatrixXd::Constant(1, m_n, 1.0);
    m_b = Eigen::VectorXd::Constant(1, 1.0);
    auto _c = m_c * (-1. / m_riskAversion);

    piqp::DenseSolver<double> solver;
    solver.settings().verbose = m_verbose;
    solver.settings().compute_timings = m_verbose;
    solver.setup(m_P, _c, m_A, m_b);

    piqp::Status status = solver.solve();

    m_status = status;
    m_result = solver.result().x;
    if (status == piqp::Status::PIQP_SOLVED) {
        m_variance = m_result.transpose() * m_P * m_result;
    }

    if (m_verbose) {
        std::cout << "P :\n" << m_P << std::endl;
        std::cout << "c = " << m_c.transpose() << std::endl;
        std::cout << "A :\n" << m_A << std::endl;
        std::cout << "b = " << m_b.transpose() << std::endl;
        std::cout << "corr :\n " << cov2corr(m_P) << std::endl;
        std::cout << "lambda = " << m_riskAversion << std::endl;
        std::cout << "status = " << m_status << std::endl;
        std::cout << "weight = " << m_result.transpose() << std::endl;
        if (m_bIncludeCash) {
            std::cout << "cash weight = " << m_result(m_result.size() - 1) << std::endl;
        }
        std::cout << "portfolio variance = " << m_variance << std::endl;
    }
}

}