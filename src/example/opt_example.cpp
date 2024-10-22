#include <fp_opt.h>
#include <iostream>
#include <limits>


void MinimumVariance_example();
void MeanVarianceWithCash_example();
void MeanVarianceWithoutCash_example();

int main() {
    // MinimumVariance_example();
    // MeanVarianceWithCash_example();
    MeanVarianceWithoutCash_example();
    return 0;
}

void MeanVarianceWithoutCash_example() {
    FP::FpOpt opt;
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
    opt.set_expected_return(ret);
    opt.solve();
}

void MeanVarianceWithCash_example() {
    FP::FpOpt opt;
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
}

void MinimumVariance_example() {
    FP::FpOpt opt;
    opt.set_type(FP::FpOptType::MinimumVariance);
    size_t nIns = 2;
    bool incCash = false;
    opt.set_size(nIns, incCash);
    Eigen::MatrixXd cov(nIns, nIns);
    cov << 0.03, -0.01, -0.01, 0.05;

    opt.set_covariance(cov);
    opt.solve();
}
