#include <fp_opt.h>
#include <fp_eigen.h>
#include <piqp/piqp.hpp>

namespace FP {
void FpOpt::clear() {
    m_status = 0;
    m_variance = NAN;
    m_expected_ret = NAN;
    m_turnover = NAN;
    m_result.resize(0);
    // m_P.resize(0, 0);
    // m_c.resize(0);
    // m_A.resize(0, 0);
    // m_b.resize(0);
    // m_G.resize(0, 0);
    // m_h.resize(0);
    m_x_lb.resize(0);
    m_x_ub.resize(0);
}

void FpOpt::clear_barra() {
    m_G.conservativeResize(0, 0);
    m_lh.resize(0);
    m_uh.resize(0);
    m_A.resize(0, 0);
    m_b.resize(0);
}

FpOpt::~FpOpt() {
    if (m_sSolver) {
        delete reinterpret_cast<piqp::SparseSolver<double>*>(m_sSolver);
    }
    if (m_dSolver) {
        delete reinterpret_cast<piqp::DenseSolver<double>*>(m_dSolver);
    }
}

void FpOpt::sanity_check() {
    if (m_bLongOnly && m_bDollarNeutral) {
        throw std::runtime_error("LongOnly and DollarNeutral cannot turn on the same time!");
    }
}
void FpOpt::solve() {
    clear();
    sanity_check();

    switch (m_optType) {
        case FpOptType::MinimumVariance: {
            handle_MinimumVariance();
            break;
        }
        case FpOptType::MeanVariance: {
            handle_MeanVariance();
            break;
        }
        case FpOptType::Constrained: {
            handle_Constrained();
            break;
        }
        case FpOptType::SoftConstrained: {
            handle_SoftConstrained();
            break;
        }
        case FpOptType::Barra: {
          handle_barra();
          break;
        }
        default:
            printf("should not print here!\n");
    }
}

void FpOpt::set_riskAversion(double v) { 
    m_riskAversion = v;
    if (m_riskAversion <= 1e-6) {
        m_covConstrain = false;
    }
}

void FpOpt::set_covariance_vec(const std::vector<double>& cov) {
    m_covConstrain = true;
    if (m_optType == FpOptType::SoftConstrained) {
        m_orig_cov = cov;
    } else {
        throw std::runtime_error("expected SoftConstrained!");
    }
}
void FpOpt::set_covariance(const Eigen::MatrixXd& cov) {
    uint32_t numRows = cov.rows();
    uint32_t numCols = cov.cols();

    if (numRows != numCols) {
        throw std::runtime_error("expected covariance is a square!");
    }
    if (numRows > m_n) {
        throw std::runtime_error("expected covariance numRows < m_n! "
            + std::to_string(numRows) + " vs " + std::to_string(m_n));
    }
    
    m_covConstrain = true;
    if (m_optType == FpOptType::SoftConstrained || m_optType == FpOptType::Barra) {
        m_orig_cov = ToVector(cov);
        return;
    }

    if (m_bIncludeCash) {
        if (m_n == numRows) {
            m_P = cov;
        } else if (numRows < m_n) {
            // set the left upper block of m_P to cov
            m_P = Eigen::MatrixXd::Zero(m_n, m_n);
            m_P.block(0, 0, numRows, numCols) = cov;
        }
    } else {
        m_P = cov;
    }
}

void FpOpt::set_expected_return_vec(const std::vector<double>& ret) {
    if (m_optType == FpOptType::SoftConstrained) {
        m_y_hat = ret;
        m_c = ToVector(ret);
    } else {
        throw std::runtime_error("expected SoftConstrained!");
    }
}

void FpOpt::set_expected_return(const Eigen::VectorXd& ret, double rf) {
    uint32_t num_ = ret.size();
    if (num_ > m_n) {
        throw std::runtime_error("expected return num_ < m_n! " 
            + std::to_string(num_) + " vs " + std::to_string(m_n));
    }

    if (m_bIncludeCash) {
        if (m_n == num_) {
            m_c = ret;
        } else if (num_ < m_n) {
            m_c = Eigen::VectorXd::Zero(m_n);
            m_c.segment(0, num_) = ret;
            m_c(m_c.size() - 1) = rf;
        }
    } else {
        m_c = ret;
    }
    m_y_hat = ToVector(m_c);
}

void FpOpt::set_type(FpOptType type) {
    m_optType = type;
}

void FpOpt::set_size(size_t nIns, bool incCash) {
    m_nIns = nIns;
    m_bIncludeCash = incCash;
    m_n = m_nIns;
    if (incCash) m_n++;
    m_P = Eigen::MatrixXd::Identity(m_n, m_n);
}

std::vector<double> FpOpt::get_result() const {
    return ToVector(m_result);
}
}