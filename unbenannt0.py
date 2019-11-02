import neptune
from config import EnvConfig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

comet_cfg = EnvConfig()
session = neptune.Session(api_token=comet_cfg.neptune_token)
project = session.get_project(project_qualified_name=comet_cfg.neptune_project_name)

experiments = project.get_experiments(state='succeeded')
print(experiments)

#params = []
#lyap_inter = []
lyap_intra_large = []
#neg_intra = []

for exp in experiments:
    print(exp.id)
#    properties = exp.get_properties()
#
#    model_update = properties['target_model_update']
#    mem_len = properties['memory_limit']
#    alpha = properties['learning_rate']

#    lyap_inter.append(exp.get_numeric_channels_values('lyap_exp_inter_ins_0','lyap_exp_inter_ins_1','lyap_exp_inter_ins_2').to_numpy()[0][1:])
    lyap_intra = exp.get_numeric_channels_values('lyap_exp_intra_ins_0','lyap_exp_intra_ins_1','lyap_exp_intra_ins_2').to_numpy()[:,1:]
#    neg_intra.append(sum(sum(lyap_intra<0))/600)
    lyap_intra_large.append(lyap_intra)
    # print(lyap)
#    params.append([model_update, mem_len, alpha])



#params = np.array(params).astype(np.float)
#lyap_inter = np.min(np.array(lyap_inter).astype(np.float), axis=1)
#lyap_intra = np.array(lyap_intra).astype(np.float)

with open('dumps/lyap_intra.p','wb') as f:
    pickle.dump(lyap_intra_large,f)