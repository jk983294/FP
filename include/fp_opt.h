#pragma once

#include <Eigen/Dense>
#include <vector>

namespace FP {
enum class FpOptType : int32_t {
    None,
    MinimumVariance,
    MeanVariance,
    Constrained,
    Barra,
    SoftConstrained, // tv into penalty
};

struct FpOpt {
    FpOpt() = default;
    FpOpt(bool verbose): m_verbose{verbose} {}
    void clear();
    void solve();
    void set_covariance_vec(const std::vector<double>& cov);
    void set_expected_return_vec(const std::vector<double>& ret);
    void set_covariance(const Eigen::MatrixXd& cov);
    void set_expected_return(const Eigen::VectorXd& ret, double risk_free_ret = 0.01);
    void set_type(FpOptType type);
    /**
     * includeCash only for MeanVariance
     */
    void set_size(size_t nIns, bool incCash = false);
    void set_riskAversion(double v);
    void set_tvAversion(double v);
    void set_maxIter(size_t v) { m_maxIter = v; }
    void set_cashWeight(double w_) { m_cashWeight = w_; }
    void set_insMaxWeight(double w_) { m_insMaxWeight = w_; }
    void set_insMinWeight(double w_) { m_insMinWeight = w_; }
    void set_verbose(bool flag) { m_verbose = flag; }
    void set_BetaNeutral(bool flag) { m_bBetaNeutral = flag; }
    void set_LongOnly(bool flag) { m_bLongOnly = flag; }
    void set_DollarNeutral(bool flag) { m_bDollarNeutral = flag; }
    void set_oldWeights(const std::vector<double>& ows);
    void set_benchWeights(const std::vector<double>& v);
    void add_constrain(const std::vector<double>& coefs, double lb, double ub, bool againstBench=true);
    void add_sector_constrain(const std::vector<int>& ins_sectors, const std::vector<int>& sectors,
        const std::vector<double>& sector_wgts);
    void add_tv_constrain(const std::vector<double>& old_wgts, double tv = 0.2);
    std::vector<double> get_result() const;
    void tidy_info() const;
    int get_type() const { return static_cast<int>(m_status); }
    int get_status() const { return m_status; }
    double get_variance() const { return m_variance; }
    double get_expected_return() const { return m_expected_ret; }
    double get_turnover() const { return m_turnover; }

private:
    void handle_MinimumVariance();
    void handle_MeanVariance();
    void handle_Constrained();
    void handle_SoftConstrained();
    void handle_barra();
    void sanity_check();
    void add_ins_weight_constrain();
    void _tv_constrain();
    void soft_tv_constrain();

public:
    /**
     * if m_bIncludeCash = true, then last weight is cash's weight
     */
    bool m_bIncludeCash{false};
    bool m_bDollarNeutral{false}; // long == short
    bool m_bBetaNeutral{false}; // beta^T * weight = 0
    bool m_bLongOnly{false};
    bool m_verbose{false};
    bool m_tvConstrain{false};
    bool m_covConstrain{false};
    FpOptType m_optType{FpOptType::SoftConstrained};
    /**
     * lambda1
     * greater riskAversion, more conservative, keep more cash
     */
    double m_riskAversion{1.0};
    /**
     * lambda2
     * greater m_tvAversion, more conservative, do not trade much
     */
    double m_tvAversion{1.0};
    double m_cashWeight{0};
    double m_insMaxWeight{NAN}; // individual instrument max weight
    double m_insMinWeight{NAN}; // individual instrument min weight
    double m_maxTurnover{NAN};
    size_t m_maxIter{250};
    size_t m_nIns{0};
    size_t m_n{0}; // if include cash, m_n = m_nIns + 1, else m_n = m_nIns
    std::vector<double> m_oldWeights;
    std::vector<double> m_benchWeights;
    std::vector<double> m_y_hat;
    std::vector<double> m_orig_cov;
    std::vector<double> m_lh, m_uh;
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
    double m_expected_ret{NAN};
    double m_turnover{NAN};
    Eigen::VectorXd m_result;
};
}  // namespace FP