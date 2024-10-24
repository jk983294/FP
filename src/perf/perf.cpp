#include <fp_opt.h>
#include <fp_eigen.h>
#include <iostream>
#include <random>
#include <chrono>
#include <getopt.h>

using namespace std;
using namespace std::chrono;

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
    size_t nIns = 5000;
    double riskAversion = 0;
    double maxWeight = 1;
    double cash = 0.01;
    double tv = 0.2;
    double sector_max = 0.2;

    int opt1;
    while ((opt1 = getopt(argc, argv, "hvr:i:t:s:m:n:")) != -1) {
        switch (opt1) {
            case 'r':
                riskAversion = std::stod(optarg);
                break;
            case 'n':
                nIns = std::stoul(optarg);
                break;
            case 'i':
                opt.set_insMaxWeight(std::stod(optarg));
                break;
            case 't':
                tv = std::stod(optarg);
                break;
            case 's':
                sector_max = std::stod(optarg);
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

    std::random_device rd;
    std::mt19937 gen(rd());
    int max_sector = 20;
    std::uniform_int_distribution<> dis(0, max_sector);
    std::vector<int> sectors(nIns, 0);
    for (size_t i = 0; i < nIns; ++i) {
        sectors[i] = dis(gen);
    }

    opt.set_verbose(false);
    opt.set_size(nIns, false);
    opt.set_cashWeight(cash);
    opt.set_LongOnly(true);
    opt.set_insMaxWeight(maxWeight);
    opt.add_sector_constrain(sectors, {}, {sector_max});
    opt.add_tv_constrain({}, tv);

    Eigen::MatrixXd pcor = Eigen::MatrixXd::Random(nIns, nIns);
    pcor.diagonal() = Eigen::VectorXd::Ones(nIns);
    Eigen::VectorXd sd_diag = Eigen::VectorXd::Random(nIns) * 0.1;
    Eigen::MatrixXd cov = FP::corr2cov(pcor, sd_diag);
    Eigen::VectorXd ret = Eigen::VectorXd::Random(nIns) * 0.01;

    opt.set_covariance(cov);
    opt.set_expected_return(ret);
    opt.set_riskAversion(riskAversion);
    steady_clock::time_point start = steady_clock::now();
    opt.solve();
    steady_clock::time_point end = steady_clock::now();
    cout << "took " << nanoseconds{end - start}.count() << " ns." << endl;
    return 0;
}

