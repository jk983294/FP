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

Eigen::VectorXd ToVector(const std::vector<double>& vec) {
    Eigen::VectorXd _vector = Eigen::Map<const Eigen::VectorXd>(vec.data(), vec.size());
    return _vector;
}

std::vector<double> ToVector(const Eigen::VectorXd& vec) {
    std::vector<double> ret(vec.size());
    const double* ptr = vec.data();
    std::copy(ptr, ptr + vec.size(), ret.data());
    return ret;
}

std::vector<double> ToVector(const Eigen::MatrixXd& m) {
    std::vector<double> ret(m.rows() * m.cols());
    const double* ptr = m.data();
    std::copy(ptr, ptr + ret.size(), ret.data());
    return ret;
}

void append(Eigen::VectorXd& target, double val) {
    target.conservativeResize(target.size() + 1);
    target(target.size() - 1) = val;
}

void append(Eigen::VectorXd& target, const Eigen::VectorXd& vec) {
    if (target.size() == 0) {
        target = vec;
    } else {
        target.conservativeResize(target.size() + vec.size());
        target.tail(vec.size()) = vec;
    }
}

void append(Eigen::MatrixXd& target, const Eigen::MatrixXd& m, bool vertically) {
    if (vertically) {
        if (target.rows() == 0) {
            target = m;
        } else {
            target.conservativeResize(target.rows() + m.rows(), target.cols());
            target.bottomRows(m.rows()) = m;
        }
    } else {
        if (target.cols() == 0) {
            target = m;
        } else {
            target.conservativeResize(target.rows(), target.cols() + m.cols());
            target.rightCols(m.cols()) = m;
        }
    }
}

void append(Eigen::MatrixXd& target, const Eigen::VectorXd& vec, bool vertically) {
    if (vertically) {
        if (target.rows() == 0) {
            target.conservativeResize(1, vec.size());
        } else {
            target.conservativeResize(target.rows() + 1, target.cols());
        }
        target.row(target.rows() - 1) = vec.transpose();
    } else {
        if (target.rows() == 0) {
            target.conservativeResize(vec.size(), 1);
        } else {
            target.conservativeResize(target.rows(), target.cols() + 1);
        }
        target.col(target.cols() - 1) = vec;
    }
}

void resize(Eigen::VectorXd& target, size_t new_size) {
    Eigen::VectorXd resized_(new_size);
    resized_.setZero();
    resized_.head(target.size()) = target;
    target.swap(resized_);
}

void resize(Eigen::MatrixXd& target, size_t new_row, size_t new_col) {
    Eigen::MatrixXd resized_(new_row, new_col);
    resized_.setZero();
    resized_.block(0, 0, target.rows(), target.cols()) = target;
    target.swap(resized_);
}
}