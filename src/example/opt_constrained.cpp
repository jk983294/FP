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
}

int main(int argc, char** argv) {
    FP::FpOpt opt;
    opt.set_type(FP::FpOptType::Constrained);
    size_t nIns = 2;
    opt.m_riskAversion = 2;

    int opt1;
    while ((opt1 = getopt(argc, argv, "hlr:i:")) != -1) {
        switch (opt1) {
            case 'r':
                opt.set_riskAversion(std::stod(optarg));
                break;
            case 'i':
                opt.set_insMaxWeight(std::stod(optarg));
                break;
            case 'l':
                opt.set_LongOnly(true);
                break;
            case 'h':
            default:
                help();
                return 0;
        }
    }

    opt.set_size(nIns, false);
    opt.set_cashWeight(0.05);
    opt.set_LongOnly(true);
    opt.set_insMaxWeight(0.6);
    opt.add_sector_constrain({0, 1}, {}, {0.6});
    opt.add_tv_constrain({}, 0.2);
    Eigen::MatrixXd cov(nIns, nIns);
    cov << 0.03, -0.01, -0.01, 0.05;
    Eigen::VectorXd ret(nIns);
    ret << 0.07, 0.1;

    opt.set_covariance(cov);
    opt.set_expected_return(ret);
    opt.solve();
    return 0;
}

