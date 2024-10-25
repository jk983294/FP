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
    std::cout << "  -r arg (=)                          riskAversion default 2" << std::endl;
    std::cout << "  -n arg (=5000)                          n isn" << std::endl;
    std::cout << "  -i arg (=)                          max weight per instrument" << std::endl;
    std::cout << "  -t arg (=)                          tvAversion" << std::endl;
    std::cout << "  -s arg (=)                          sector" << std::endl;
    std::cout << "  -m arg (=)                          ins max weight" << std::endl;
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

    size_t test_cnt = 10;
    std::vector<int> random_pos(test_cnt, 0);
    std::vector<double> random_value(test_cnt, 0);
    std::uniform_int_distribution<> pos_dis(0, nIns);
    std::uniform_real_distribution<double> value_dist(-1., 1.);
    for (size_t i = 0; i < test_cnt; ++i) {
        random_pos[i] = pos_dis(gen);
        random_value[i] = value_dist(gen) * 0.01;
    }

    opt.set_verbose(false);
    opt.set_size(nIns, false);
    opt.set_cashWeight(cash);
    opt.set_LongOnly(true);
    opt.set_insMaxWeight(maxWeight);
    opt.add_sector_constrain(sectors, {}, {sector_max});
    // opt.add_tv_constrain({}, tv);

    Eigen::MatrixXd pcor = Eigen::MatrixXd::Random(nIns, nIns);
    pcor.diagonal() = Eigen::VectorXd::Ones(nIns);
    Eigen::VectorXd sd_diag = Eigen::VectorXd::Random(nIns) * 0.1;
    Eigen::MatrixXd cov = FP::corr2cov(pcor, sd_diag);
    Eigen::VectorXd ret = Eigen::VectorXd::Random(nIns) * 0.01;

    opt.set_covariance(cov);
    opt.set_expected_return(ret);
    opt.set_riskAversion(riskAversion);

    std::unordered_map<int, int> results;
    steady_clock::time_point start = steady_clock::now();
    for (size_t i = 0; i < test_cnt; i++) {
        opt.solve();
        results[opt.get_status()]++;

        ret(random_pos[i]) = random_value[i];
        opt.set_expected_return(ret);
        opt.set_oldWeights(opt.get_result());
        opt.set_tvAversion(tv);
    }
    steady_clock::time_point end = steady_clock::now();
    cout << "took " << nanoseconds{end - start}.count() / test_cnt << " ns." << endl;
    for (auto& item : results) {
        printf("status %d, cnt=%d\n", item.first, item.second);
    }
    return 0;
}

