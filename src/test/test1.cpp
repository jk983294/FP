#include <numeric>
#include <fp_opt.h>
#include <catch.hpp>

using namespace std;

TEST_CASE("MinimumVariance", "[MinimumVariance]") {
    FP::FpOpt opt(false);
    opt.set_type(FP::FpOptType::MinimumVariance);
    opt.set_size(2, false);
    Eigen::MatrixXd cov(2, 2);
    cov << 0.03, -0.01, -0.01, 0.05;

    opt.set_covariance(cov);
    opt.solve();

    std::vector<double> expected_w = {0.59999999998722531, 0.400000000062814};
    std::vector<double> result = opt.get_result();

    REQUIRE(opt.m_status == 1);
    REQUIRE(opt.m_variance == 0.0140000000014011);
    REQUIRE(result == expected_w);
}

TEST_CASE("MeanVariance With Cash", "[MeanVariance]") {
    FP::FpOpt opt(false);
    opt.set_type(FP::FpOptType::MeanVariance);
    size_t nIns = 2;
    bool incCash = true;
    opt.m_riskAversion = 10;
    opt.set_size(nIns, incCash);
    Eigen::MatrixXd cov(nIns, nIns);
    cov << 0.03, -0.01, -0.01, 0.05;
    Eigen::VectorXd ret(nIns);
    ret << 0.07, 0.1;

    opt.set_covariance(cov);
    opt.set_expected_return(ret, 0.01);
    opt.solve();

    std::vector<double> expected_w = {0.27857142963032916, 0.23571428638111727, 0.48571428403712158};
    std::vector<double> result = opt.get_result();

    REQUIRE(opt.m_status == 1);
    REQUIRE(opt.m_variance == 0.003792857167566918);
    REQUIRE(result == expected_w);
}

TEST_CASE("MeanVariance Without Cash", "[MeanVariance]") {
    FP::FpOpt opt(false);
    opt.set_type(FP::FpOptType::MeanVariance);
    size_t nIns = 2;
    bool incCash = false;
    opt.m_riskAversion = 10;
    opt.set_size(nIns, incCash);
    Eigen::MatrixXd cov(nIns, nIns);
    cov << 0.03, -0.01, -0.01, 0.05;
    Eigen::VectorXd ret(nIns);
    ret << 0.07, 0.1;

    opt.set_covariance(cov);
    opt.set_expected_return(ret, 0.01);
    opt.solve();

    std::vector<double> expected_w = {0.57000000000119344, 0.4300000000493942};
    std::vector<double> result = opt.get_result();

    REQUIRE(opt.m_status == 1);
    REQUIRE(opt.m_variance == 0.014090000001591406);
    REQUIRE(result == expected_w);
}