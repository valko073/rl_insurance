import neptune
from config import EnvConfig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

comet_cfg = EnvConfig()
session = neptune.Session(api_token=comet_cfg.neptune_token)
project = session.get_project(project_qualified_name=comet_cfg.neptune_project_name)

experiments = project.get_experiments()[171:252]
print(experiments)

params = []
lyap_inter = []
lyap_intra_large = []
neg_intra = []

for exp in experiments:
    properties = exp.get_properties()

    model_update = properties['target_model_update']
    mem_len = properties['memory_limit']
    alpha = properties['learning_rate']

    lyap_inter.append(exp.get_numeric_channels_values('lyap_exp_inter_ins_0','lyap_exp_inter_ins_1','lyap_exp_inter_ins_2').to_numpy()[0][1:])
    lyap_intra = exp.get_numeric_channels_values('lyap_exp_intra_ins_0','lyap_exp_intra_ins_1','lyap_exp_intra_ins_2').to_numpy()[:,1:]
    neg_intra.append(sum(sum(lyap_intra<0))/600)
    lyap_intra_large.append(lyap_intra)
    # print(lyap)
    params.append([model_update, mem_len, alpha])



params = np.array(params).astype(np.float)
lyap_inter = np.min(np.array(lyap_inter).astype(np.float), axis=1)
lyap_intra = np.array(lyap_intra).astype(np.float)

print(lyap_inter)


plt.figure()
plt.plot(range(len(lyap_inter)),lyap_inter)

plt.figure()
plt.subplot(1,3,1)
plt.scatter(params[:,0], lyap_inter)
plt.subplot(1,3,2)
plt.scatter(params[:,1], lyap_inter)
plt.subplot(1,3,3)
plt.scatter(params[:,2], lyap_inter)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(params[:,0],params[:,1], params[:,2], c=lyap_inter, cmap=plt.hot())
ax.set_xlabel('Model_update'); ax.set_ylabel('Memory_limit'); ax.set_zlabel('Learning_rate');
fig.colorbar(img)
plt.title('3d plot')

plt.figure()
plt.scatter(range(len(neg_intra)), neg_intra)
plt.show()
