#include <numeric>
#include <fp_opt.h>
#include <catch.hpp>

using namespace std;

TEST_CASE("Constrained cashWeight", "[Constrained]") {
    FP::FpOpt opt(false);
    opt.set_type(FP::FpOptType::Constrained);
    opt.set_size(2, false);
    opt.set_cashWeight(0.05);
    opt.set_riskAversion(1);
    size_t nIns = 2;
    Eigen::MatrixXd cov(nIns, nIns);
    cov << 0.03, -0.01, -0.01, 0.05;
    Eigen::VectorXd ret(nIns);
    ret << 0.07, 0.1;

    opt.set_covariance(cov);
    opt.set_expected_return(ret);
    opt.solve();

    std::vector<double> expected_w = {0.27000000626328541, 0.67999999373674724};
    std::vector<double> result = opt.get_result();

    REQUIRE(opt.m_status == 1);
    REQUIRE(opt.m_variance == 0.021634999624204922);
    REQUIRE(result == expected_w);
}

TEST_CASE("Constrained long only max ins weight", "[Constrained]") {
    FP::FpOpt opt(false);
    opt.set_type(FP::FpOptType::Constrained);
    opt.set_size(2, false);
    opt.set_cashWeight(0.05);
    opt.set_riskAversion(1);
    opt.set_LongOnly(true);
    opt.set_insMaxWeight(0.6);
    size_t nIns = 2;
    Eigen::MatrixXd cov(nIns, nIns);
    cov << 0.03, -0.01, -0.01, 0.05;
    Eigen::VectorXd ret(nIns);
    ret << 0.07, 0.1;

    opt.set_covariance(cov);
    opt.set_expected_return(ret);
    opt.solve();

    std::vector<double> expected_w = {0.35000023995561796, 0.59999976004440514};
    std::vector<double> result = opt.get_result();

    REQUIRE(opt.m_status == 1);
    REQUIRE(opt.m_variance == 0.017474989441959791);
    REQUIRE(result == expected_w);
}

TEST_CASE("Constrained sector weight", "[Constrained]") {
    FP::FpOpt opt(false);
    opt.set_type(FP::FpOptType::Constrained);
    opt.set_size(2, false);
    opt.set_cashWeight(0.05);
    opt.set_riskAversion(1);
    opt.add_sector_constrain({0, 1}, {}, {0.55});
    size_t nIns = 2;
    Eigen::MatrixXd cov(nIns, nIns);
    cov << 0.03, -0.01, -0.01, 0.05;
    Eigen::VectorXd ret(nIns);
    ret << 0.07, 0.1;

    opt.set_covariance(cov);
    opt.set_expected_return(ret);
    opt.solve();

    std::vector<double> expected_w = {0.40000000534643249, 0.54999999465356908};
    std::vector<double> result = opt.get_result();

    REQUIRE(opt.m_status == 1);
    REQUIRE(opt.m_variance == 0.015524999818221372);
    REQUIRE(result == expected_w);
}

TEST_CASE("Constrained sector weight full", "[Constrained]") {
    FP::FpOpt opt(false);
    opt.set_type(FP::FpOptType::Constrained);
    opt.set_size(2, false);
    opt.set_cashWeight(0.05);
    opt.set_riskAversion(1);
    opt.add_sector_constrain({0, 1}, {0, 1}, {0.55, 0.55});
    size_t nIns = 2;
    Eigen::MatrixXd cov(nIns, nIns);
    cov << 0.03, -0.01, -0.01, 0.05;
    Eigen::VectorXd ret(nIns);
    ret << 0.07, 0.1;

    opt.set_covariance(cov);
    opt.set_expected_return(ret);
    opt.solve();

    std::vector<double> expected_w = {0.40000000534643249, 0.54999999465356908};
    std::vector<double> result = opt.get_result();

    REQUIRE(opt.m_status == 1);
    REQUIRE(opt.m_variance == 0.015524999818221372);
    REQUIRE(result == expected_w);
}