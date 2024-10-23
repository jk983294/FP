#include <fp_opt.h>
#include <fp_eigen.h>
#include <piqp/piqp.hpp>
#include <iostream>

namespace FP {
void FpOpt::add_ins_weight_constrain() {
    double lower_ = -2.0;
    double upper_ = 2.0;
    if (m_bLongOnly) {
        lower_ = 0.0;
        if (std::isfinite(m_insMaxWeight)) {
            upper_ = m_insMaxWeight;
        }
    } else {
        if (std::isfinite(m_insMaxWeight)) {
            lower_ = -std::abs(m_insMaxWeight);
            upper_ = std::abs(m_insMaxWeight);
        }
    }
    m_x_lb = Eigen::VectorXd::Constant(m_n, lower_);
    m_x_ub = Eigen::VectorXd::Constant(m_n, upper_);
}

void FpOpt::add_sector_constrain(const std::vector<int>& ins_sectors, const std::vector<int>& _sectors,
        const std::vector<double>& _sector_wgts) {
    if (ins_sectors.empty()) return;
    if (ins_sectors.size() != m_n) {
        throw std::runtime_error("expect ins_sectors.size() == " + std::to_string(m_n));
    }

    std::unordered_map<int, int> sec2cnt;
    for (auto sec : ins_sectors) sec2cnt[sec]++;
    if (_sectors.empty()) {
        std::vector<int> sectors = _sectors;
        std::transform(sec2cnt.begin(), sec2cnt.end(), std::back_inserter(sectors), 
            [](const auto &pair){return pair.first;} );
        std::sort(sectors.begin(), sectors.end());

        double wgt = 1.5/ sectors.size();
        if (!_sector_wgts.empty()) wgt = _sector_wgts.front();

        for (size_t i = 0; i < sectors.size(); i++) {
            int sec = sectors[i];
            Eigen::VectorXd vec = Eigen::VectorXd::Constant(m_n, 0.);
            for (size_t j = 0; j < m_n; j++) {
                if (ins_sectors[j] == sec) {
                    vec[j] = 1.0;
                }
            }
            append(m_G, vec, true);
            append(m_h, wgt);
            if (m_verbose) {
                std::cout << "add " << vec.transpose() << " <= " << wgt << std::endl;
            }
        }
    } else {
        double dft_wgt = 1.5 / sec2cnt.size();
        for (size_t i = 0; i < _sectors.size(); i++) {
            int sec = _sectors[i];
            if (sec2cnt[sec] <= 0) continue;
            Eigen::VectorXd vec = Eigen::VectorXd::Constant(m_n, 0.);
            for (size_t j = 0; j < m_n; j++) {
                if (ins_sectors[j] == sec) {
                    vec[j] = 1.0;
                }
            }
            append(m_G, vec, true);
            if (_sectors.size() == _sector_wgts.size()) append(m_h, _sector_wgts[i]);
            else if (!_sector_wgts.empty()) append(m_h, _sector_wgts.front());
            else append(m_h, dft_wgt);

            if (m_verbose) {
                std::cout << "add " << vec.transpose() << " <= " << m_h(m_h.size() - 1) << std::endl;
            }
        }
    }
}

void FpOpt::add_tv_constrain(const std::vector<double>& old_wgts, double tv) {
    m_oldWeights = old_wgts;
    m_maxTurnover = tv;
    m_tvConstrain = true;
    if (m_maxTurnover <= 1e-6 || m_maxTurnover >= 1.) {
        throw std::runtime_error("expect maxTurnover between [0 ~ 1]");
    }
    if (m_oldWeights.empty()) {
        m_oldWeights.resize(m_n, 0.);
    } else if (m_oldWeights.size() != m_n) {
        throw std::runtime_error("expect m_oldWeights.size() == m_n " + std::to_string(m_n));
    }
}

void FpOpt::_tv_constrain() {
    if (!m_tvConstrain) return;

    // introduce auxiliary variables z ∈ R^n, w - z <= w_old, -z - w <= w_old, sum(z) = tv
    resize(m_P, m_n * 2, m_n * 2);
    resize(m_c, m_n * 2);
    resize(m_A, m_A.rows(), m_n * 2);
    resize(m_G, m_G.rows(), m_n * 2);
    resize(m_x_lb, m_n * 2);
    resize(m_x_ub, m_n * 2);

    {
        Eigen::MatrixXd new_G = Eigen::MatrixXd::Zero(m_n * 2, m_n * 2);
        Eigen::VectorXd new_h = Eigen::VectorXd::Zero(m_n * 2);
        for (size_t i = 0; i < m_n; i++) {
            new_G(i, i) = 1.0;
            new_G(i, m_n + i) = -1.0;
            new_h[i] = m_oldWeights[i];

            new_G(m_n + i, i) = -1.0;
            new_G(m_n + i, m_n + i) = -1.0;
            new_h[i] = m_oldWeights[i];
        }
        append(m_G, new_G, true);
        append(m_h, new_h);
    }
}

void FpOpt::handle_Constrained() {
    if (m_bIncludeCash) throw std::runtime_error("not support m_bIncludeCash, use m_cashWeight instead!");

    /**
     * min 1/2 w^t * P * w - (1/lambda) * w^t * r
     * s.t. w^t * i = 1. - cash_weight
     *      0. <= w <= m_insMaxWeight
     *      ||w - old_w||_1 <= tv
     */
    m_A = Eigen::MatrixXd::Constant(1, m_n, 1.0);
    m_b = Eigen::VectorXd::Constant(1, 1.0 - m_cashWeight);
    add_ins_weight_constrain();
    _tv_constrain();
    auto _c = m_c * (-1. / m_riskAversion);

    piqp::DenseSolver<double> solver;
    solver.settings().verbose = m_verbose;
    solver.settings().compute_timings = m_verbose;
    if (m_G.rows() > 0) {
        solver.setup(m_P, _c, m_A, m_b, m_G, m_h, m_x_lb, m_x_ub);
    } else {
        solver.setup(m_P, _c, m_A, m_b, std::nullopt, std::nullopt, m_x_lb, m_x_ub);
    }

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
        if (m_G.rows() > 0) {
            std::cout << "G :\n" << m_G << std::endl;
            std::cout << "h = " << m_h.transpose() << std::endl;
        }
        std::cout << "corr :\n " << cov2corr(m_P) << std::endl;
        std::cout << "lambda = " << m_riskAversion << std::endl;
        std::cout << "weight bound [" << m_x_lb[0] << ", " << m_x_ub[0] << "]" << std::endl;
        std::cout << "status = " << m_status << std::endl;
        std::cout << "weight = " << m_result.transpose() << " cash=" << m_cashWeight << std::endl;
        std::cout << "portfolio variance = " << m_variance << std::endl;
    }
}

}