#pragma once

#include <omp.h>
#include <Eigen/src/Core/util/Meta.h>

namespace piqp {
namespace dense {

template<typename MatrixType, typename VectorType>
static int64_t fast_llt_rank_update_lower(MatrixType& mat, const VectorType& vec, const typename MatrixType::RealScalar& sigma)
{
  using std::sqrt;
  using namespace Eigen::numext;
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::Matrix<Scalar,-1,1> TempVectorType;

  int64_t n = mat.cols();

  {
    TempVectorType temp = vec;
    double beta = 1;
    for(int64_t j=0; j<n; ++j)
    {
      double Ljj = Eigen::numext::real(mat.coeff(j,j));
      double dj = Eigen::numext::abs2(Ljj);
      Scalar wj = temp.coeff(j);
      double swj2 = sigma*Eigen::numext::abs2(wj);
      double gamma = dj*beta + swj2;

      double x = dj + swj2/beta;
      if (x<=double(0))
        return j;
      double nLjj = sqrt(x);
      mat.coeffRef(j,j) = nLjj;
      beta += swj2/dj;

      // Update the terms of L
      int64_t rs = n-j-1;
      if(rs) {
        temp.tail(rs) -= (wj/Ljj) * mat.col(j).tail(rs);
        if(gamma != 0)
          mat.col(j).tail(rs) = (nLjj/Ljj) * mat.col(j).tail(rs) + (nLjj * sigma*Eigen::numext::conj(wj)/gamma)*temp.tail(rs);
      }
    }
  }
  return -1;
}

struct fast_llt_inplace {
  template<typename MatrixType>
  static int64_t unblocked(MatrixType& mat) {
    using std::sqrt;

    const int64_t size = mat.rows();
    for(int64_t k = 0; k < size; ++k) {
      int64_t rs = size-k-1; // remaining size

      Eigen::Block<MatrixType,-1,1> A21(mat,k+1,k,rs,1);
      Eigen::Block<MatrixType,1,-1> A10(mat,k,0,1,k);
      Eigen::Block<MatrixType,-1,-1> A20(mat,k+1,0,rs,k);

      double x = Eigen::numext::real(mat.coeff(k,k));
      if (k>0) x -= A10.squaredNorm();
      if (x<=double(0))
        return k;
      mat.coeffRef(k,k) = x = sqrt(x);
      if (k>0 && rs>0) A21.noalias() -= A20 * A10.adjoint();
      if (rs>0) A21 /= x;
    }
    return -1;
  }

  template<typename MatrixType>
  static int64_t blocked(MatrixType& m) {
    int64_t size = m.rows();
    if(size<32)
      return unblocked(m);

    int64_t blockSize = size/8;
    blockSize = (blockSize/16)*16;
    blockSize = (std::min)((std::max)(blockSize,int64_t(8)), int64_t(128));

    for (int64_t k=0; k<size; k+=blockSize)
    {
      // partition the matrix:
      //       A00 |  -  |  -
      // lu  = A10 | A11 |  -
      //       A20 | A21 | A22
      int64_t bs = (std::min)(blockSize, size-k);
      int64_t rs = size - k - bs;
      Eigen::Block<MatrixType,-1,-1> A11(m,k,   k,   bs,bs);
      Eigen::Block<MatrixType,-1,-1> A21(m,k+bs,k,   rs,bs);
      Eigen::Block<MatrixType,-1,-1> A22(m,k+bs,k+bs,rs,rs);

      int64_t ret;
      if((ret=unblocked(A11))>=0) return k+ret;
      if(rs>0) A11.adjoint().template triangularView<Eigen::Upper>().template solveInPlace<Eigen::OnTheRight>(A21);
      if(rs>0) A22.template selfadjointView<Eigen::Lower>().rankUpdate(A21, -1); // bottleneck
    }
    return -1;
  }

  template<typename MatrixType, typename VectorType>
  static int64_t rankUpdate(MatrixType& mat, const VectorType& vec, double sigma)
  {
    return fast_llt_rank_update_lower(mat, vec, sigma);
  }
};

template <typename TMat, typename TMat1>
void fast_llt(TMat& llt, TMat1& kkt_mat, int threads) {
    auto& _llt_mat = llt.m_matrix;
    const int64_t size = kkt_mat.rows();
    _llt_mat.resize(size, size);
    if (!Eigen::internal::is_same_dense(_llt_mat, kkt_mat.derived()))
        _llt_mat = kkt_mat.derived();

    // Compute matrix L1 norm = max abs column sum.
    std::vector<double> _l1_norms(size);
    #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int64_t col = 0; col < size; ++col) {
        _l1_norms[col] = _llt_mat.col(col).tail(size - col).template lpNorm<1>() + _llt_mat.row(col).head(col).template lpNorm<1>();
    }

    double _l1_norm = 0;
    for (int64_t col = 0; col < size; ++col) {
        if (_l1_norms[col] > _l1_norm) _l1_norm = _l1_norms[col];
    }
    llt.m_l1_norm = _l1_norm;

    llt.m_isInitialized = true;

    bool ok = fast_llt_inplace::blocked(_llt_mat) == -1;
    llt.m_info = ok ? Eigen::Success : Eigen::NumericalIssue;
}
}
}