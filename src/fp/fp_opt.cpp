#include <fp_opt.h>

namespace FP {
void FpOpt::clear() {

}
void FpOpt::solve() {

}
void FpOpt::set_covariance(const Eigen::MatrixXd& cov) {
    m_P = cov;
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
}