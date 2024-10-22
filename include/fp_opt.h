#include <Eigen/Dense>
#include <vector>

namespace FP {
enum class FpOptType : int32_t {
    None,
    MinimumVariance,
    MeanVariance,
};

struct FpOpt {
    FpOpt() = default;
    FpOpt(bool verbose): m_verbose{verbose} {}
    void clear();
    void solve();
    void set_covariance(const Eigen::MatrixXd& cov);
    void set_expected_return(const Eigen::VectorXd& ret, double risk_free_ret = 0.01);
    void set_type(FpOptType type);
    void set_size(size_t nIns, bool incCash);
    void set_verbose(bool flag) { m_verbose = flag; }
    void set_BetaNeutral(bool flag) { m_bBetaNeutral = flag; }
    void set_LongOnly(bool flag) { m_bLongOnly = flag; }
    void set_DollarNeutral(bool flag) { m_bDollarNeutral = flag; }
    std::vector<double> get_result() const;

private:
    void handle_MinimumVariance();
    void handle_MeanVariance();
    void sanity_check();

public:
    /**
     * if m_bIncludeCash = true, then last weight is cash's weight
     */
    bool m_bIncludeCash{false};
    bool m_bDollarNeutral{false}; // long == short
    bool m_bBetaNeutral{false}; // beta^T * weight = 0
    bool m_bLongOnly{false};
    bool m_verbose{true};
    FpOptType m_optType{FpOptType::None};
    double m_riskAversion{1.0}; // greater riskAversion, more conservative, keep more cash
    size_t m_nIns{0};
    size_t m_n{0}; // if include cash, m_n = m_nIns + 1, else m_n = m_nIns
    Eigen::MatrixXd m_P; // cov matrix
    Eigen::VectorXd m_c; // return vector
    Eigen::MatrixXd m_A; // equality constrains, Ax = b
    Eigen::VectorXd m_b;
    Eigen::MatrixXd m_G; // inequality constrains, Gx <= h
    Eigen::VectorXd m_h;
    Eigen::VectorXd m_x_lb; // bound x_lb <= x <= x_ub
    Eigen::VectorXd m_x_ub;

    int32_t m_status{0};
    double m_variance{NAN};
    Eigen::VectorXd m_result;
};
}  // namespace FP