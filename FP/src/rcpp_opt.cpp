
#include <Rcpp.h>
#include <RcppEigen.h>
#include <fp_opt.h>

using namespace Rcpp;

// [[Rcpp::depends(RcppEigen)]]

RCPP_MODULE(FP) {
    Rcpp::class_<FP::FpOpt>("Opt")
        .constructor()
        .method("solve", &FP::FpOpt::solve)
        .method("set_covariance_vec", &FP::FpOpt::set_covariance_vec)
        .method("set_expected_return_vec", &FP::FpOpt::set_expected_return_vec)
        .method("set_covariance", &FP::FpOpt::set_covariance)
        .method("set_expected_return", &FP::FpOpt::set_expected_return)
        .method("set_size", &FP::FpOpt::set_size)
        .method("set_riskAversion", &FP::FpOpt::set_riskAversion)
        .method("set_tvAversion", &FP::FpOpt::set_tvAversion)
        .method("set_cashWeight", &FP::FpOpt::set_cashWeight)
        .method("set_insMaxWeight", &FP::FpOpt::set_insMaxWeight)
        .method("set_verbose", &FP::FpOpt::set_verbose)
        .method("set_LongOnly", &FP::FpOpt::set_LongOnly)
        .method("set_oldWeights", &FP::FpOpt::set_oldWeights)
        .method("add_sector_constrain", &FP::FpOpt::add_sector_constrain)
        .method("tidy_info", &FP::FpOpt::tidy_info)
        .method("get_type", &FP::FpOpt::get_type)
        .method("get_result", &FP::FpOpt::get_result)
        .method("get_status", &FP::FpOpt::get_status)
        .method("get_variance", &FP::FpOpt::get_variance)
        .method("get_expected_return", &FP::FpOpt::get_expected_return)
        .method("get_turnover", &FP::FpOpt::get_turnover);
}
