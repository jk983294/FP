#include <fp_opt.h>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <random>
#include <omp.h>

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
    std::cout << "  -n arg (=10)                          nIns" << std::endl;
    std::cout << "  -x arg (=4)                          n_thread" << std::endl;
    std::cout << "./soft_constrained -o 0.4 -t 1 -r 1 -x 1 -n 1000" << std::endl;
}

int main(int argc, char** argv) {
    
    FP::FpOpt opt;
    opt.set_type(FP::FpOptType::SoftConstrained);
    size_t nIns = 10;
    int n_thread = 4;
    double riskAversion = 0;
    double maxWeight = 1;
    double cash = 0.05;
    double old_weight = NAN;
    double tv = 0.5;
    double sector_max = NAN;
    bool longOnly = true;
    bool verbose = false;

    int opt1;
    while ((opt1 = getopt(argc, argv, "hvlr:i:t:s:o:m:n:x:")) != -1) {
        switch (opt1) {
            case 'r':
                riskAversion = std::stod(optarg);
                break;
            case 'n':
                nIns = std::stoull(optarg);
                break;
            case 'x':
                n_thread = std::stoull(optarg);
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

    omp_set_num_threads(n_thread);
    opt.set_threads(n_thread);
    printf("n_thread=%d\n", n_thread);

    // std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    opt.set_verbose(verbose);
    opt.set_size(nIns, false);
    opt.set_cashWeight(cash);
    if (longOnly) opt.set_LongOnly(true);
    opt.set_insMaxWeight(maxWeight);
    if (std::isfinite(sector_max)) {
        opt.add_sector_constrain({0, 1}, {}, {sector_max});
    }
    if (std::isfinite(old_weight)) {
        std::vector<double> ows(nIns);
        double wsum = 0;
        for (size_t i = 0; i < nIns; ++i) {
            ows[i] = std::abs(dis(gen));
            wsum += ows[i];
        }
        for (size_t i = 0; i < nIns; ++i) {
            ows[i] = ows[i] / wsum * old_weight;
        }
        opt.set_oldWeights(ows);
    }
    if (std::isfinite(tv)) {
        opt.set_tvAversion(tv);
    }
    Eigen::VectorXd sd_ = Eigen::VectorXd::Random(nIns).cwiseAbs() * 0.1;
    Eigen::VectorXd ret = Eigen::VectorXd::Random(nIns) * 0.01;
    Eigen::MatrixXd cov(nIns, nIns);

    for (size_t i = 0; i < nIns; ++i) {
        for (size_t j = 0; j < nIns; ++j) {
            double cov_v = dis(gen) * sd_(i) * sd_(j);
            cov(i, j) = cov_v;
            cov(j, i) = cov_v;
        }
    }
    for (size_t i = 0; i < nIns; ++i) {
        cov(i, i) = sd_(i) * sd_(i);
    }

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
