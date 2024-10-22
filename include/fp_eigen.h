#include <Eigen/Dense>

namespace FP {
Eigen::MatrixXd cov2corr(const Eigen::MatrixXd& cov);
Eigen::MatrixXd corr2cov(const Eigen::MatrixXd& corr, const Eigen::VectorXd& sd);
std::vector<double> ToVector(const Eigen::VectorXd& vec);
}