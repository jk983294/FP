import numpy as np
import pyfp



if __name__ == '__main__':
    nIns = 2
    inc_cash = False
    cov = np.array([[0.03, -0.01],
                    [-0.01, 0.05]])
    ret = np.array([0.07, 0.1])
    ins_sectors = np.array([0, 1])
    sectors = np.array([0, 1])
    sector_wgts = np.array([0.8, 0.8])
    old_wgts = np.array([0.4, 0.4])

    opt = pyfp.PyOpt()
    opt.set_verbose(True)
    opt.set_size(nIns, inc_cash)
    opt.set_LongOnly(True)
    opt.set_insMaxWeight(0.6)
    opt.add_sector_constrain(ins_sectors, sectors, sector_wgts)
    opt.set_riskAversion(1.0)
    opt.set_tvAversion(0.5)
    opt.set_covariance(cov)
    opt.set_oldWeights(old_wgts)
    opt.set_expected_return_vec(ret)
    opt.solve()
    opt.tidy_info()
    print(opt.get_status())
    print(opt.get_result())
    print(opt.get_variance())
    print(opt.get_expected_return())
    print(opt.get_turnover())