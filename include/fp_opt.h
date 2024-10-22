#include <Eigen/Dense>
#include <vector>

namespace FP {
enum class FpOptType : int32_t {
    None,
    MinimumVariance,
    MeanVarianceWithCash,
};

struct FpOpt {
    void clear();
    void solve();
    void set_covariance(const Eigen::MatrixXd& cov);
    void set_type(FpOptType type);
    void set_size(size_t nIns, bool incCash);

public:
    bool m_bIncludeCash{false};
    FpOptType m_optType{FpOptType::None};
    size_t m_nIns{0};
    size_t m_n{0}; // if include cash, m_n = m_nIns + 1, else m_n = m_nIns
    Eigen::MatrixXd m_P;
    Eigen::VectorXd m_c;
    Eigen::MatrixXd m_A;
    Eigen::VectorXd m_b;
    Eigen::MatrixXd m_G;
    Eigen::VectorXd m_h;
    Eigen::VectorXd m_x_lb;
    Eigen::VectorXd m_x_ub;
};
}  // namespace FP