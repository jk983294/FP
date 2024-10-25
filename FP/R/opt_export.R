#' create OPT object
#'
#' # construct
#' opt <- new(FP::Opt)
#'
#' @name Opt
#' @export Opt

Rcpp::loadModule(module = "FP", TRUE)