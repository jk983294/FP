#include <numeric>
#include <fp_opt.h>
#include <catch.hpp>

using namespace std;

TEST_CASE("SoftConstrained cashWeight", "[SoftConstrained]") {
    FP::FpOpt opt(false);
    opt.set_type(FP::FpOptType::SoftConstrained);
    opt.set_size(2, false);
    opt.set_LongOnly(true);
    opt.set_insMaxWeight(0.6);
    opt.set_cashWeight(0.05);
    opt.set_riskAversion(1);
    double old_weight = 0.4;
    double tv = 2;
    opt.add_tv_constrain({old_weight, old_weight}, tv);
    size_t nIns = 2;
    Eigen::MatrixXd cov(nIns, nIns);
    cov << 0.03, -0.01, -0.01, 0.05;
    Eigen::VectorXd ret(nIns);
    ret << 0.07, 0.1;

    opt.set_covariance(cov);
    opt.set_expected_return(ret);
    opt.solve();

    std::vector<double> expected_w = {0.47000000137242515, 0.47999999862907999};
    std::vector<double> result = opt.get_result();

    REQUIRE(opt.m_status == 1);
    REQUIRE(opt.m_variance == 0.013634999972609599);
    REQUIRE(opt.m_expected_ret == 0.080899999958977786);
    REQUIRE(result == expected_w);
}

TEST_CASE("SoftConstrained no cov", "[SoftConstrained]") {
    FP::FpOpt opt(false);
    opt.set_type(FP::FpOptType::SoftConstrained);
    opt.set_size(2, false);
    opt.set_LongOnly(true);
    double cash = 0.05;
    // opt.set_insMaxWeight(0.6);
    opt.set_cashWeight(cash);
    opt.set_riskAversion(0);
    double old_weight = 0.3;
    double tv = 0.3;
    opt.add_tv_constrain({old_weight, 1 - cash - old_weight}, tv);
    size_t nIns = 2;
    Eigen::VectorXd ret(nIns);
    ret << 0.07, 0.1;

    opt.set_expected_return(ret);
    opt.set_tvAversion(tv);
    opt.solve();

    std::vector<double> expected_w = {0.25000000287852914, 0.69999999712139693};
    std::vector<double> result = opt.get_result();

    REQUIRE(opt.m_status == 1);
    REQUIRE(std::isnan(opt.m_variance));
    REQUIRE(opt.m_expected_ret == 0.087499999913636745);
    REQUIRE(opt.m_turnover == 0.099999994242867873);
    REQUIRE(result == expected_w);
}