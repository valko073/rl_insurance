import numpy as np
import multiprocessing
import os
import sys
import itertools

alphas = np.geomspace(0.00001, 0.5, endpoint=False, num=15)
replay_len = [3, 10, 100, 250, 500, 1000]
# replay_len = [1, 10]
target_model_updates = [.09, .5, .9, 10, 100, 500, 1000]


def worker(input):
    (alph, rep_len, tar_mod_up) = input
    os.system("python run_insurance.py --comet --num_insurances=3 --num_agents=3 --num_steps=100000 "+
              "--learning_rate={} --memory_limit={} --target_model_update={}".format(alph, rep_len, tar_mod_up))



def main():
    pool = multiprocessing.Pool()
    pool.map(worker, itertools.product(alphas, replay_len, target_model_updates))
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
