#include <fp_opt.h>
#include <fp_eigen.h>
#include <piqp/piqp.hpp>
#include <iostream>

namespace FP {
void FpOpt::handle_barra() {
    /**
     * min -x^t * r
     * s.t. x^t * i = 1.
     *      0. <= x <= m_insMaxWeight
     */
    if(!m_useSparse) {
        m_P = Eigen::MatrixXd::Zero(m_n, m_n);
    }
    size_t new_n = m_n;
    Eigen::VectorXd _c = m_c * (-1.);

    {
      Eigen::MatrixXd tmp_ = Eigen::MatrixXd::Constant(1, m_n, 1.0);
      append(m_A, tmp_, true);
      append(m_b, 1.0);
    }

    add_ins_weight_constrain();

    if (m_tvConstrain) {
        if (m_oldWeights.size() != m_n) {
            throw std::runtime_error("expect old weight!");
        }
        m_maxTurnover = m_tvAversion;
        /**
         * x - w = buy - sell
         * sigma(buy + sell) < turnover
         * 0 <= x <= m_insMaxWeight
         * 0 <= buy <= m_insMaxWeight
         * 0 <= sell <= m_insMaxWeight
         */
        new_n = m_n * 3; // x, buy_amount, sell_amount
        if(!m_useSparse) {
            m_P = Eigen::MatrixXd::Zero(new_n, new_n);
        }
        append(_c, Eigen::VectorXd::Zero(m_n * 2));
        m_A.conservativeResize(m_A.rows(), new_n);
        m_A.rightCols(m_n * 2).setZero();
        Eigen::MatrixXd splits = Eigen::MatrixXd::Zero(m_n, new_n);
        for (size_t i = 0; i < m_n; i++) { // x[i] - buy[i] + sell[i] = old_w[i]
            splits(i, i) = 1;
            splits(i, i + m_n) = -1;
            splits(i, i + m_n * 2) = 1;
        }
        append(m_A, splits, true);
        append(m_b, ToVector(m_oldWeights));
        m_G.conservativeResize(m_G.rows(), new_n);
        m_G.rightCols(m_n * 2).setZero();
        Eigen::VectorXd tv_vec = Eigen::VectorXd::Zero(new_n);
        tv_vec.tail(m_n * 2).setOnes();
        append(m_G, tv_vec, true);
        append(m_lh, 0);
        append(m_uh, m_maxTurnover);
        append(m_x_lb, Eigen::VectorXd::Zero(m_n * 2));
        append(m_x_ub, Eigen::VectorXd::Ones(m_n * 2));
    }

    if (m_useSparse) {
        bool first_time = false;
        piqp::SparseSolver<double>* solver{nullptr};
        if (m_sSolver == nullptr) {
            m_sSolver = solver = new piqp::SparseSolver<double>;
            first_time = true;
            solver->settings().verbose = m_verbose;
            solver->settings().compute_timings = m_verbose;
            solver->settings().max_iter = m_maxIter;
        } else {
            solver = reinterpret_cast<piqp::SparseSolver<double>*>(m_sSolver);
            // if (m_barra_pre_n != new_n) 
                first_time = true;
        }
        Eigen::SparseMatrix<double> _P(new_n, new_n);
        _P.makeCompressed();
        Eigen::SparseMatrix<double> _A = m_A.sparseView();
        Eigen::SparseMatrix<double> _G = m_G.sparseView();

        if (first_time) {
            solver->setup(_P, _c, _A, m_b, _G, m_lh, m_uh, m_x_lb, m_x_ub);
        } else {
            solver->update(_P, _c, _A, m_b, _G, m_lh, m_uh, m_x_lb, m_x_ub);
        }

        piqp::Status status = solver->solve();
        m_status = status;
        m_iter = solver->result().info.iter;
        m_result = solver->result().x;
    } else {
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

    if (m_tvConstrain) {
        m_result.conservativeResize(m_n); // remove buy & sell dummy variable
    }
    m_expected_ret = m_result.transpose() * Eigen::Map<Eigen::VectorXd>(m_y_hat.data(), m_n);
    if (m_tvConstrain) {
        Eigen::Map<Eigen::VectorXd> old_w(m_oldWeights.data(), m_n);
        m_expected_ret = m_result.transpose() * (m_c - old_w);
    }
    if (m_tvConstrain) {
        m_turnover = (m_result - Eigen::Map<Eigen::VectorXd>(m_oldWeights.data(), m_n)).lpNorm<1>();
    } else {
        m_turnover = m_result.lpNorm<1>();
    }

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
}