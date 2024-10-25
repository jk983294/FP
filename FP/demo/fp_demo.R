library(FP)

opt <- new(FP::Opt)

nIns <- 2L
inc_cash <- FALSE
cov <- matrix(c(0.03, -0.01, -0.01, 0.05), nrow = 2, ncol = 2)
ret <- c(0.07, 0.1)
ins_sectors <- c(0L, 1L)
sectors <- c(0L, 1L)
sector_wgts <- c(0.8, 0.8)

opt$set_verbose(TRUE)
opt$set_size(nIns, inc_cash)
opt$set_LongOnly(TRUE)
opt$set_insMaxWeight(0.6)
opt$add_sector_constrain(ins_sectors, sectors, sector_wgts)
opt$set_riskAversion(1.0)
opt$set_tvAversion(0.5)
opt$set_covariance(cov)
opt$set_oldWeights(c(0.4, 0.4))
opt$set_expected_return_vec(ret)
opt$solve()
opt$tidy_info()
opt$get_status()
opt$get_result()
opt$get_variance()
opt$get_expected_return()
opt$get_turnover()