#include <Eigen/Dense>

namespace FP {
Eigen::MatrixXd cov2corr(const Eigen::MatrixXd& cov);
Eigen::MatrixXd corr2cov(const Eigen::MatrixXd& corr, const Eigen::VectorXd& sd);
std::vector<double> ToVector(const Eigen::VectorXd& vec);
std::vector<double> ToVector(const Eigen::MatrixXd& m);
void append(Eigen::VectorXd& target, double val);
void append(Eigen::VectorXd& target, const Eigen::VectorXd& vec);
void append(Eigen::MatrixXd& target, const Eigen::MatrixXd& m, bool vertically = true);
void append(Eigen::MatrixXd& target, const Eigen::VectorXd& vec, bool vertically = true);
void resize(Eigen::VectorXd& target, size_t new_size);
void resize(Eigen::MatrixXd& target, size_t new_row, size_t new_col);
}