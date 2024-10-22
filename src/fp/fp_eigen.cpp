#include <fp_eigen.h>

namespace FP {
Eigen::MatrixXd cov2corr(const Eigen::MatrixXd& cov_) {
    int n = cov_.rows();

    Eigen::MatrixXd corr_ = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd sd = cov_.diagonal().array().sqrt();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            corr_(i, j) = cov_(i, j) / (sd(i) * sd(j));
        }
    }
    return corr_;
}

Eigen::MatrixXd corr2cov(const Eigen::MatrixXd& corrMatrix, const Eigen::VectorXd& sd) {
    int n = corrMatrix.rows();

    Eigen::MatrixXd cov_ = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cov_(i, j) = corrMatrix(i, j) * sd(i) * sd(j);
        }
    }
    return cov_;
}

std::vector<double> ToVector(const Eigen::VectorXd& vec) {
    std::vector<double> ret(vec.size());
    const double* ptr = vec.data();
    std::copy(ptr, ptr + vec.size(), ret.data());
    return ret;
}
}