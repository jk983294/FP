#include <fp_opt.h>
#include <iostream>
#include <limits>
#include <getopt.h>

static void help() {
    std::cout << "Program options:" << std::endl;
    std::cout << "  -h                                    list help" << std::endl;
    std::cout << "  -l arg (=)                          long only" << std::endl;
    std::cout << "  -r arg (=)                          riskAversion default 2" << std::endl;
    std::cout << "  -i arg (=)                          max weight per instrument" << std::endl;
    std::cout << "  -t arg (=)                          turnover" << std::endl;
    std::cout << "  -s arg (=)                          sector" << std::endl;
    std::cout << "  -o arg (=)                          old weight" << std::endl;
    std::cout << "  -m arg (=)                          ins max weight" << std::endl;
    std::cout << "  -v arg (=false)                          verbose" << std::endl;
}

int main(int argc, char** argv) {
    FP::FpOpt opt;
    opt.set_type(FP::FpOptType::SoftConstrained);
    size_t nIns = 2;
    double riskAversion = 0;
    double maxWeight = 1;
    double cash = 0.05;
    double old_weight = NAN;
    double tv = NAN;
    double sector_max = NAN;
    bool longOnly = true;
    bool verbose = false;

    int opt1;
    while ((opt1 = getopt(argc, argv, "hvlr:i:t:s:o:m:")) != -1) {
        switch (opt1) {
            case 'r':
                riskAversion = std::stod(optarg);
                break;
            case 'i':
                opt.set_insMaxWeight(std::stod(optarg));
                break;
            case 'l':
                longOnly = false;
                break;
            case 'v':
                verbose = true;
                break;
            case 't':
                tv = std::stod(optarg);
                break;
            case 's':
                sector_max = std::stod(optarg);
                break;
            case 'o':
                old_weight = std::stod(optarg);
                break;
            case 'm':
                maxWeight = std::stod(optarg);
                break;
            case 'h':
            default:
                help();
                return 0;
        }
    }

    opt.set_verbose(verbose);
    opt.set_size(nIns, false);
    opt.set_cashWeight(cash);
    if (longOnly) opt.set_LongOnly(true);
    opt.set_insMaxWeight(maxWeight);
    if (std::isfinite(sector_max)) {
        opt.add_sector_constrain({0, 1}, {}, {sector_max});
    }
    if (std::isfinite(old_weight)) {
        opt.set_oldWeights({old_weight, 1. - cash - old_weight});
    }
    if (std::isfinite(tv)) {
        opt.set_tvAversion(tv);
    }
    Eigen::MatrixXd cov(nIns, nIns);
    cov << 0.03, -0.01, -0.01, 0.05;
    Eigen::VectorXd ret(nIns);
    ret << 0.07, 0.1;

    if (std::isfinite(tv)) {
        opt.set_covariance(cov);
    }
    opt.set_expected_return(ret);
    opt.set_riskAversion(riskAversion);
    opt.solve();
    if (!verbose) {
        opt.tidy_info();
    }
    return 0;
}

