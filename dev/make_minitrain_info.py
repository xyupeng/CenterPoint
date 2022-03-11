import pickle
import os


def main():
    info_path = '/data/common/waymo/infos_train_01sweeps_filter_zero_gt.pkl'
    save_path = 'data/common/waymo/infos_train1000_01sweeps_filter_zero_gt.pkl'
    info = pickle.load(open(info_path, 'rb'))
    mini_info = info[:1000]
    assert not os.path.isfile(save_path)
    pickle.dump(mini_info, open(save_path, 'w'))


main()
