import neptune
from config import EnvConfig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.stattools import grangercausalitytests

comet_cfg = EnvConfig()
session = neptune.Session(api_token=comet_cfg.neptune_token)
project = session.get_project(project_qualified_name=comet_cfg.neptune_project_name)

#experiments = project.get_experiments('GRID-52')
#experiments = project.get_experiments('GRID-362')
#experiments = project.get_experiments()[30:40]
experiments = project.get_experiments(state='succeeded')
print(experiments)

params = []
lyap_inter = []
lyap_intra = []
neg_intra = []
correlations = []
granger_p_vals = []

for exp in experiments:
    print(exp.id)
    properties = exp.get_properties()

    model_update = properties['target_model_update']
    mem_len = properties['memory_limit']
    alpha = properties['learning_rate']

    lyap_inter.append(exp.get_numeric_channels_values('lyap_exp_inter_ins_0','lyap_exp_inter_ins_1','lyap_exp_inter_ins_2').to_numpy()[0][1:])
    lyap_intra = exp.get_numeric_channels_values('lyap_exp_intra_ins_0','lyap_exp_intra_ins_1','lyap_exp_intra_ins_2').to_numpy()[:,1:]
    neg_intra.append(sum(sum(lyap_intra<0))/600)
    # print(lyap)
    ins_costs = exp.get_numeric_channels_values('insurance_cost_0_log_0','insurance_cost_0_log_1','insurance_cost_1_log_0','insurance_cost_1_log_1',
                                                'insurance_cost_2_log_0','insurance_cost_2_log_1')
    ins_usages = exp.get_numeric_channels_values('step_num_insured_0_log_0','step_num_insured_0_log_1','step_num_insured_1_log_0','step_num_insured_1_log_1',
                                                 'step_num_insured_2_log_0','step_num_insured_2_log_1')
    params.append([model_update, mem_len, alpha])

    
    step_costs = []
    step_usages = []
    for i in range(3):
        costs = np.array([ins_costs['insurance_cost_'+str(i)+'_log_0'],ins_costs['insurance_cost_'+str(i)+'_log_1']]).flatten()
        costs = costs[:99983]
        
        usages = np.array([ins_usages['step_num_insured_'+str(i)+'_log_0'],ins_usages['step_num_insured_'+str(i)+'_log_1']]).flatten()
        usages = usages[:99983]
    
        step_costs.append(costs)
        step_usages.append(usages)
    
    step_costs = pd.DataFrame(step_costs).transpose().dropna().transpose()
    step_usages = pd.DataFrame(step_usages).transpose().dropna().transpose()
    
    pearsons_corr = step_costs.corrwith(step_usages, axis=1)
    correlations.append(pearsons_corr)
    
#    granger_ps = []
#    for i in range(3):
#        granger_ps.append(grangercausalitytests(np.array([step_usages.iloc[i],step_costs.iloc[i]]).transpose(),10).popitem()[1][0].popitem()[1][1])
#        
#    granger_p_vals.append(granger_ps)
#%%
params = np.array(params).astype(np.float)
lyap_inter_min = np.mean(np.abs(np.array(correlations).astype(np.float)), axis=1)
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
img = ax.scatter(params[:,0],params[:,1], params[:,2], c=lyap_inter_min, cmap=plt.hot())
ax.set_xlabel('Model_update'); ax.set_ylabel('Memory_limit'); ax.set_zlabel('Learning_rate');
fig.colorbar(img)
plt.title('3d plot')

plt.figure()
plt.scatter(range(len(neg_intra)), neg_intra)

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

plt.figure()
plt.subplot(1,3,1)
plt.scatter(params[:,0], lyap_inter_min)
plt.xlabel('target_model_update')
plt.ylabel('Percentage of episodes w/ negative lyap exp')
plt.subplot(1,3,2)
plt.scatter(params[:,1], lyap_inter_min)
plt.xlabel('memory_limit')
plt.subplot(1,3,3)
plt.scatter(params[:,2], lyap_inter_min)
plt.xlabel('alpha')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(params[:,0],params[:,1], params[:,2], c=mean_abs_corr, cmap=plt.hot())
ax.set_xlabel('Model_update'); ax.set_ylabel('Memory_limit'); ax.set_zlabel('Learning_rate');
fig.colorbar(img)
plt.title('3d plot of Mean Absolute Correlation between insurance cost and usage')
plt.show()

import plotly.express as px
import plotly.graph_objects as go
fig = go.Figure(data = [go.Scatter3d(
        x=params[:,0],
        y=params[:,1],
        z=params[:,2],
        mode='markers',
        marker=dict(
                color=neg_intra
                )
        )])
fig.show()
