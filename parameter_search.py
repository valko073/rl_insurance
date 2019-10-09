import numpy as np
import multiprocessing
import os
import sys

alphas = np.geomspace(0.00001, 1, endpoint=False, num=20)
replay_len = [1, 10, 100, 250, 500, 1000]
# replay_len = [1, 10]
target_model_updates = [.09, .5, .9, 10, 100, 500, 1000]


def worker(alph):
    print(os.path.dirname(sys.executable))
    for rep_len in replay_len:
        for tar_mod_up in target_model_updates:
            os.system("run_insurance.py --comet --num_insurances=3 --num_agents=3 --num_steps=2 "+
                      "--learning_rate={} --memory_limit={} --target_model_update={}".format(alph, rep_len, tar_mod_up))



def main():
    pool = multiprocessing.Pool()
    pool.map(worker, [alph for alph in alphas])
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
