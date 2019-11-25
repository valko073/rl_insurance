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

experiments = project.get_experiments("GRID-293")

exp = experiments[0]

vals = exp.get_numeric_channels_values('avg_insurance_cost_0','num_insured_0','avg_insurance_cost_1',
                                       'num_insured_1','avg_insurance_cost_2','num_insured_2')


plt.figure()
ax1 = plt.subplot(3,1,1)
vals['avg_insurance_cost_0'].plot(ax=ax1,label='Insurance price')
#plt.ylabel('Average insurance price over episode')
vals['num_insured_0'].plot(secondary_y=True,label='Insurance usage')
#plt.ylabel('Total insurance usage over episode')
plt.title('Insurance 1')
#ax1.legend(['Average insurance cost over episode','Total usage over episode'])
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax.get_legend_handles_labels()

plt.legend(h1+h2, l1+l2, loc=5)
ax2 = plt.subplot(3,1,2)
vals['avg_insurance_cost_1'].plot(ax=ax2,legend=False,label='Insurance 2')
plt.ylabel('Average insurance price over episode')
vals['num_insured_1'].plot(secondary_y=True,label='Insurance usage')
plt.ylabel('Total insurance usage over episode')
plt.title('Insurance 2')
ax3 = plt.subplot(3,1,3)
vals['avg_insurance_cost_2'].plot(ax=ax3,legend=False,label='Insurance 3')
#plt.ylabel('Average insurance price over episode')
vals['num_insured_2'].plot(secondary_y=True,label='Insurance usage')
#plt.ylabel('Total insurance usage over episode')
plt.title('Insurance 3')


experiments = project.get_experiments("GRID-502")

exp = experiments[0]

vals = exp.get_numeric_channels_values('avg_insurance_cost_0','num_insured_0','avg_insurance_cost_1',
                                       'num_insured_1','avg_insurance_cost_2','num_insured_2')


plt.figure()
ax1 = plt.subplot(3,1,1)
vals['avg_insurance_cost_0'].plot(ax=ax1,label='Insurance price')
#plt.ylabel('Average insurance price over episode')
vals['num_insured_0'].plot(secondary_y=True,label='Insurance usage')
#plt.ylabel('Total insurance usage over episode')
plt.title('Insurance 1')
#ax1.legend(['Average insurance cost over episode','Total usage over episode'])
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax.get_legend_handles_labels()

plt.legend(h1+h2, l1+l2, loc=5)
ax2 = plt.subplot(3,1,2)
vals['avg_insurance_cost_1'].plot(ax=ax2,legend=False,label='Insurance 2')
plt.ylabel('Average insurance price over episode')
vals['num_insured_1'].plot(secondary_y=True,label='Insurance usage')
plt.ylabel('Total insurance usage over episode')
plt.title('Insurance 2')
ax3 = plt.subplot(3,1,3)
vals['avg_insurance_cost_2'].plot(ax=ax3,legend=False,label='Insurance 3')
#plt.ylabel('Average insurance price over episode')
vals['num_insured_2'].plot(secondary_y=True,label='Insurance usage')
#plt.ylabel('Total insurance usage over episode')
plt.title('Insurance 3')