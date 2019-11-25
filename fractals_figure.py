import neptune
from config import EnvConfig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
%matplotlib qt
plt.rcParams.update({'font.size': 22})

comet_cfg = EnvConfig()
session = neptune.Session(api_token=comet_cfg.neptune_token)
project = session.get_project(project_qualified_name=comet_cfg.neptune_project_name)

experiments = project.get_experiments("GRID-30")

exp = experiments[0]

vals = exp.get_numeric_channels_values('insurance_cost_1_log_0','step_num_insured_1_log_0')

vals = vals.iloc[36210:36500,:]
vals[['insurance_cost_1_log_0','step_num_insured_1_log_0']].plot()
plt.legend(['Insurance price','Insurance usage'])
plt.xlabel('Timestep t')


experiments = project.get_experiments("GRID-81")

exp = experiments[0]

vals = exp.get_numeric_channels_values('insurance_cost_2_log_0','step_num_insured_2_log_0')

vals = vals.iloc[24540:25000,:]
vals[['insurance_cost_2_log_0','step_num_insured_2_log_0']].plot(legend=False)
#plt.legend(['Insurance price','Insurance usage'])
plt.xlabel('Timestep t')


experiments = project.get_experiments("GRID-30")

exp = experiments[0]

vals = exp.get_numeric_channels_values('avg_insurance_cost_1','num_insured_1')

vals = vals.iloc[29:77,:]
ax1 = vals['avg_insurance_cost_1'].plot(label='Insurance price')
plt.ylabel('Average insurance price over episode')
plt.xlabel('Episode e')
#ax1.legend()
ax = vals['num_insured_1'].plot(secondary_y=True,label='Insurance usage')
ax.set_ylabel('Total insurance usage over episode')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax.get_legend_handles_labels()

#plt.legend(h1+h2, l1+l2)
#plt.legend(['Insurance price','Insurance usage'])


experiments = project.get_experiments("GRID-502")

exp = experiments[0]

vals = exp.get_numeric_channels_values('avg_insurance_cost_1','num_insured_1')

vals = vals.iloc[20:163,:]
ax1 = vals['avg_insurance_cost_1'].plot(label='Insurance price')
plt.ylabel('Average insurance price over episode')
plt.xlabel('Episode e')
#ax1.legend()
ax = vals['num_insured_1'].plot(secondary_y=True,label='Insurance usage')
ax.set_ylabel('Total insurance usage over episode')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax.get_legend_handles_labels()

#plt.legend(h1+h2, l1+l2)
#plt.legend(['Insurance price','Insurance usage'])



plt.figure()
ax1 = plt.subplot(3,1,1)
vals[['avg_insurance_cost_scaled_0','num_insured_0']].plot(ax=ax1,label='Insurance 1')
plt.title('Insurance 1')
ax1.legend(['Average insurance cost over episode','Total usage over episode'])
ax2 = plt.subplot(3,1,2)
vals[['avg_insurance_cost_scaled_1','num_insured_1']].plot(ax=ax2,legend=False,label='Insurance 2')
plt.title('Insurance 2')
ax3 = plt.subplot(3,1,3)
vals[['avg_insurance_cost_scaled_2','num_insured_2']].plot(ax=ax3,legend=False,label='Insurance 3')
plt.title('Insurance 3')


experiments = project.get_experiments("GRID-502")

exp = experiments[0]

vals = exp.get_numeric_channels_values('avg_insurance_cost_scaled_0','num_insured_0','avg_insurance_cost_scaled_1',
                                       'num_insured_1','avg_insurance_cost_scaled_2','num_insured_2')


plt.figure()
ax1 = plt.subplot(3,1,1)
vals[['avg_insurance_cost_scaled_0','num_insured_0']].plot(ax=ax1,label='Insurance 1')
plt.title('Insurance 1')
ax1.legend(['Average insurance cost over episode','Total usage over episode'])
ax2 = plt.subplot(3,1,2)
vals[['avg_insurance_cost_scaled_1','num_insured_1']].plot(ax=ax2,legend=False,label='Insurance 2')
plt.title('Insurance 2')
ax3 = plt.subplot(3,1,3)
vals[['avg_insurance_cost_scaled_2','num_insured_2']].plot(ax=ax3,legend=False,label='Insurance 3')
plt.title('Insurance 3')