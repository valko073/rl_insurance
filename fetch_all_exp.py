import neptune
from config import EnvConfig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

comet_cfg = EnvConfig()
session = neptune.Session(api_token=comet_cfg.neptune_token)
project = session.get_project(project_qualified_name=comet_cfg.neptune_project_name)

experiments = project.get_experiments()[171:]
print(experiments)

params = []
lyap_inter = []
lyap_intra = []
neg_intra = []

for exp in experiments:

    properties = exp.get_properties()

    model_update = properties['target_model_update']
    mem_len = properties['memory_limit']
    alpha = properties['learning_rate']

    try:
        # print(exp.id)
        lyap_intra = exp.get_numeric_channels_values('lyap_exp_intra_ins_0','lyap_exp_intra_ins_1','lyap_exp_intra_ins_2').to_numpy()[:,1:]
        neg_intra.append(sum(sum(lyap_intra<0))/600)
    # print(lyap)
        params.append([model_update, mem_len, alpha])
    except:
        print('except '+exp.id)



params = np.array(params).astype(np.float)
lyap_intra = np.array(lyap_intra).astype(np.float)





plt.figure()
plt.subplot(1,3,1)
plt.scatter(params[:,0], neg_intra)
plt.xlabel('target_model_update')
plt.ylabel('Percentage of episodes w/ negative lyap exp')
plt.subplot(1,3,2)
plt.scatter(params[:,1], neg_intra)
plt.xlabel('memory_limit')
plt.subplot(1,3,3)
plt.scatter(params[:,2], neg_intra)
plt.xlabel('alpha')



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(params[:,0],params[:,1], params[:,2], c=neg_intra, cmap=plt.hot())
ax.set_xlabel('Model_update'); ax.set_ylabel('Memory_limit'); ax.set_zlabel('Learning_rate');
fig.colorbar(img)
plt.title('3d plot of percentage of negative exponent episodes')

plt.figure()
plt.scatter(range(len(neg_intra)), neg_intra)
plt.show()


print('done')
