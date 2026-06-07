#include <fp_eigen.h>
#include <fp_opt.h>
#include <iostream>
#include <scs/scs.h>

namespace FP {
void FpOpt::handle_barra_te() {
  /**
   * Max Alpha @ TE — SOCP formulation via SCS.
   *
   * Factor model:  Sigma = B * Sigma_f * B' + D
   *
   * Problem:
   *   max   alpha' * x
   *   s.t.  (x - x_b)' * Sigma * (x - x_b) <= TE_bound^2
   *         sum(x) = 1
   *         B' * x = f   (factor linking)
   *         other constraints on x
   *
   * The TE constraint decomposes via the factor model as:
   *   (x - x_b)' * D * (x - x_b) + (f - f_b)' * Sigma_f * (f - f_b) <= TE^2
   *
   * With L_D = diag(sqrt(D)) and L_f = chol(Sigma_f), this becomes:
   *   || [L_D*(x-x_b); L_f*(f-f_b)] ||_2 <= TE
   *
   * which is a standard second-order cone constraint.
   *
   * SCS standard form:
   *   min   c' * z
   *   s.t.  b - A * z = s,  s in K
   *
   * Equivalently: A * z + s = b, s in K
   *
   * Decision variable z = [x(N); f(K)]
   *
   * Cone layout (rows of A, in order):
   *   1. Zero cone (z):         equality constraints  (s=0 => Az=b)
   *   2. Positive orthant (l):  inequality constraints + variable bounds (s>=0 => Az<=b)
   *   3. Second-order cone (q): TE bound
   */

  if (!m_factorModel) {
    throw std::runtime_error("BarraTE requires factor model: "
        "call set_factor_exposure, set_factor_cov, set_specific_risk first");
  }
  if (m_benchWeights.size() != m_n) {
    throw std::runtime_error("BarraTE requires benchmark weights: "
        "call set_benchWeights first");
  }

  size_t N = m_n;
  size_t K = m_nFactors;
  size_t n_vars = N + K;  // z = [x; f]

  // ---- Variable bounds ----
  double lb = std::isfinite(m_insMinWeight) ? m_insMinWeight : -2.0;
  double ub = std::isfinite(m_insMaxWeight) ? m_insMaxWeight : 2.0;

  // ---- Cholesky of factor covariance ----
  Eigen::LLT<Eigen::MatrixXd> llt(m_Fcov);
  if (llt.info() != Eigen::Success) {
    throw std::runtime_error("BarraTE: factor covariance Cholesky failed");
  }
  Eigen::MatrixXd L_f = llt.matrixL();  // K x K lower triangular

  // ---- Specific risk square root ----
  Eigen::VectorXd sqrtD = m_D.cwiseSqrt();  // N x 1

  // ---- Benchmark factor exposure ----
  Eigen::VectorXd xb = Eigen::Map<Eigen::VectorXd>(m_benchWeights.data(), N);
  Eigen::VectorXd fb = m_B.transpose() * xb;  // K x 1

  // ---- Count constraints ----
  // SCS form: b - A*z = s in K
  // Zero cone: s = 0 => A*z = b
  // Positive orthant: s >= 0 => A*z <= b
  // SOC: (s_0, s_1..) with s_0 >= ||s_1..||_2

  size_t n_existing_eq = m_A.rows();
  size_t n_eq = 1 + K + n_existing_eq;  // sum(x)=1, B'x-f=0, user eq

  size_t n_ineq_rows = m_G.rows();
  // Each double-sided ineq lh <= G*x <= uh:
  //   G*x <= uh  (upper) and  -G*x <= -lh  (lower)
  size_t n_user_pos = n_ineq_rows * 2;

  // Variable bounds: x_i >= lb => -x_i <= -lb  and  x_i <= ub
  size_t n_bound_pos = N * 2;

  size_t n_pos = n_user_pos + n_bound_pos;

  size_t soc_dim = 1 + N + K;  // [TE_bound; L_D*(x-xb); L_f*(f-fb)]

  size_t m_rows = n_eq + n_pos + soc_dim;

  // ---- Build A and b ----
  // SCS convention: b - A*z = s in K
  // So for zero cone: A*z = b
  // For positive orthant: A*z <= b  (i.e., b - A*z >= 0)
  // For SOC: b_0 - A_0*z >= ||b_{1:} - A_{1:}*z||_2

  Eigen::MatrixXd A_dense = Eigen::MatrixXd::Zero(m_rows, n_vars);
  Eigen::VectorXd b_vec = Eigen::VectorXd::Zero(m_rows);

  size_t row = 0;

  // --- Zero cone: equality constraints ---
  // A*z = b

  // sum(x) = 1
  for (size_t i = 0; i < N; i++) A_dense(row, i) = 1.0;
  b_vec(row) = 1.0;
  row++;

  // B'x - f = 0  =>  [B'  -I] z = 0
  A_dense.block(row, 0, K, N) = m_B.transpose();
  for (size_t k = 0; k < K; k++) {
    A_dense(row + k, N + k) = -1.0;
  }
  // b = 0 (already)
  row += K;

  // Existing user equality constraints: A*z = m_b
  if (n_existing_eq > 0) {
    A_dense.block(row, 0, n_existing_eq, N) = m_A;
    b_vec.segment(row, n_existing_eq) = m_b;
    row += n_existing_eq;
  }

  // --- Positive orthant: inequality constraints ---
  // A*z <= b  (b - A*z >= 0)

  // User constraints: lh <= G*x <= uh
  // Upper: G*x <= uh  =>  A_row = [G, 0], b_row = uh
  // Lower: -G*x <= -lh  =>  A_row = [-G, 0], b_row = -lh
  for (Eigen::Index j = 0; j < (Eigen::Index)n_ineq_rows; j++) {
    // Upper bound
    A_dense.row(row).head(N) = m_G.row(j);
    b_vec(row) = m_uh(j);
    row++;

    // Lower bound
    A_dense.row(row).head(N) = -m_G.row(j);
    b_vec(row) = -m_lh(j);
    row++;
  }

  // Variable bounds:
  // x_i <= ub  =>  A_row has 1 at x_i, b = ub
  // x_i >= lb  =>  -x_i <= -lb  =>  A_row has -1 at x_i, b = -lb
  for (size_t i = 0; i < N; i++) {
    // x_i <= ub
    A_dense(row, i) = 1.0;
    b_vec(row) = ub;
    row++;

    // x_i >= lb  =>  -x_i <= -lb
    A_dense(row, i) = -1.0;
    b_vec(row) = -lb;
    row++;
  }

  // --- Second-order cone: TE bound ---
  // SOC(s): s_0 >= ||s_1, ..., s_{d-1}||_2
  // We want: TE_bound >= || [L_D*(x-xb); L_f*(f-fb)] ||_2
  //
  // b - A*z = s, s in SOC
  // s_0 >= ||s_1..s_{N+K}||_2
  //
  // We want:
  //   s_0 = TE_bound  (constant)
  //   s_1..s_N = L_D * (x - xb)
  //   s_{N+1}..s_{N+K} = L_f * (f - fb)
  //
  // Since s = b - A*z:
  //   s_0 = b_0 - A_0*z = TE_bound  =>  A_0 = 0, b_0 = TE_bound
  //   s_i = b_i - A_i*z = sqrtD_i*(x_i - xb_i) = sqrtD_i*x_i - sqrtD_i*xb_i
  //       =>  A_i = -sqrtD_i * e_i^T  (row),  b_i = -sqrtD_i * xb_i
  //   Wait: s_i = b_i - A_i*z. We want s_i = sqrtD_i * x_i - sqrtD_i * xb_i
  //   So b_i - A_i*z = sqrtD_i * x_i - sqrtD_i * xb_i
  //   =>  A_i = -sqrtD_i * e_i^T  (negate to get + on x_i side)
  //   =>  b_i = -sqrtD_i * xb_i
  //   Check: b_i - A_i*z = -sqrtD_i*xb_i - (-sqrtD_i*x_i) = sqrtD_i*(x_i - xb_i) ✓

  // s_0: TE_bound (constant)
  // A row = 0, b = TE_bound
  b_vec(row) = m_teBound;
  row++;

  // s_1..s_N: L_D * (x - xb)
  for (size_t i = 0; i < N; i++) {
    A_dense(row, i) = -sqrtD(i);
    b_vec(row) = -sqrtD(i) * xb(i);
    row++;
  }

  // s_{N+1}..s_{N+K}: L_f * (f - fb)
  for (size_t k = 0; k < K; k++) {
    for (size_t j = 0; j < K; j++) {
      A_dense(row, N + j) = -L_f(k, j);
    }
    b_vec(row) = -(L_f.row(k) * fb)(0);
    row++;
  }

  // ---- Objective c: min -alpha' * x => max alpha' * x ----
  Eigen::VectorXd c_vec = Eigen::VectorXd::Zero(n_vars);
  c_vec.head(N) = -m_c;

  // ---- Convert A to CSC format ----
  size_t nnz = 0;
  for (Eigen::Index j = 0; j < (Eigen::Index)n_vars; j++) {
    for (Eigen::Index i = 0; i < (Eigen::Index)m_rows; i++) {
      if (std::abs(A_dense(i, j)) > 1e-15) nnz++;
    }
  }

  std::vector<scs_int> Ap(n_vars + 1), Ai(nnz);
  std::vector<scs_float> Ax(nnz);

  size_t idx = 0;
  for (Eigen::Index j = 0; j < (Eigen::Index)n_vars; j++) {
    Ap[j] = static_cast<scs_int>(idx);
    for (Eigen::Index i = 0; i < (Eigen::Index)m_rows; i++) {
      if (std::abs(A_dense(i, j)) > 1e-15) {
        Ai[idx] = static_cast<scs_int>(i);
        Ax[idx] = static_cast<scs_float>(A_dense(i, j));
        idx++;
      }
    }
  }
  Ap[n_vars] = static_cast<scs_int>(idx);

  // ---- Set up SCS data ----
  ScsMatrix A_scs;
  A_scs.x = Ax.data();
  A_scs.i = Ai.data();
  A_scs.p = Ap.data();
  A_scs.m = static_cast<scs_int>(m_rows);
  A_scs.n = static_cast<scs_int>(n_vars);

  ScsMatrix *P_ptr = SCS_NULL;

  std::vector<scs_float> b_scs(m_rows), c_scs(n_vars);
  for (size_t i = 0; i < m_rows; i++) b_scs[i] = static_cast<scs_float>(b_vec(i));
  for (size_t i = 0; i < n_vars; i++) c_scs[i] = static_cast<scs_float>(c_vec(i));

  ScsData d;
  d.m = static_cast<scs_int>(m_rows);
  d.n = static_cast<scs_int>(n_vars);
  d.A = &A_scs;
  d.P = P_ptr;
  d.b = b_scs.data();
  d.c = c_scs.data();

  // ---- Cone specification ----
  ScsCone k;
  k.z = static_cast<scs_int>(n_eq);
  k.l = static_cast<scs_int>(n_pos);

  scs_int q_dim = static_cast<scs_int>(soc_dim);
  k.q = &q_dim;
  k.qsize = 1;

  // Unused cones
  k.bsize = 0;
  k.bu = SCS_NULL;
  k.bl = SCS_NULL;
  k.s = SCS_NULL;
  k.ssize = 0;
  k.cs = SCS_NULL;
  k.cssize = 0;
  k.ep = 0;
  k.ed = 0;
  k.p = SCS_NULL;
  k.psize = 0;

  // ---- Settings ----
  ScsSettings stgs;
  scs_set_default_settings(&stgs);
  stgs.verbose = m_verbose ? 1 : 0;
  stgs.max_iters = static_cast<scs_int>(m_maxIter);
  stgs.eps_abs = 1e-6;
  stgs.eps_rel = 1e-6;

  // ---- Solve ----
  ScsSolution sol;
  sol.x = SCS_NULL;
  sol.y = SCS_NULL;
  sol.s = SCS_NULL;

  ScsInfo info;

  scs_int status = scs(&d, &k, &stgs, &sol, &info);

  m_status = static_cast<int>(status);
  m_iter = info.iter;

  if (status == SCS_SOLVED || status == SCS_SOLVED_INACCURATE) {
    m_result = Eigen::VectorXd(N);
    for (size_t i = 0; i < N; i++) {
      m_result(i) = sol.x[i];
    }

    // ---- Post-processing ----
    m_expected_ret = m_result.transpose()
        * Eigen::Map<Eigen::VectorXd>(m_y_hat.data(), N);

    // Variance via factor model
    Eigen::VectorXd f = m_B.transpose() * m_result;
    m_variance = f.transpose() * m_Fcov * f
               + m_result.cwiseProduct(m_D).dot(m_result);

    // Tracking error
    Eigen::VectorXd x_act = m_result - xb;
    Eigen::VectorXd f_act = f - fb;
    double te_var = x_act.cwiseProduct(m_D).dot(x_act)
                  + f_act.transpose() * m_Fcov * f_act;
    m_turnover = std::sqrt(te_var);

    if (m_verbose) {
      std::cout << "=== BarraTE Max Alpha @ TE ===" << std::endl;
      std::cout << "N = " << N << ", K = " << K << std::endl;
      std::cout << "TE_bound = " << m_teBound << std::endl;
      std::cout << "SCS status: " << info.status << std::endl;
      std::cout << "m_result = " << m_result.transpose() << std::endl;
      std::cout << "sum(w)  = " << m_result.sum() << std::endl;
      std::cout << "exp_ret = " << m_expected_ret << std::endl;
      std::cout << "variance= " << m_variance << std::endl;
      std::cout << "TE      = " << m_turnover << std::endl;
      std::cout << "factor_exposure (B'x) = " << f.transpose() << std::endl;
      tidy_info();
    }
  } else {
    m_result = Eigen::VectorXd::Zero(N);
    m_expected_ret = NAN;
    m_variance = NAN;
    m_turnover = NAN;

    if (m_verbose) {
      std::cout << "=== BarraTE FAILED ===" << std::endl;
      std::cout << "SCS status: " << info.status << " (code " << status << ")" << std::endl;
    }
  }

  // SCS allocates x, y, s internally — must free
  if (sol.x) free(sol.x);
  if (sol.y) free(sol.y);
  if (sol.s) free(sol.s);

  m_barra_pre_n = n_vars;
}
}  // namespace FP
