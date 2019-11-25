import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update({'font.size': 22})


with open('dumps/correlations.p', 'rb') as f:
    correlations = pickle.load(f)
with open('dumps/costs.p', 'rb') as f:
    costs = pickle.load(f)
with open('dumps/ids.p', 'rb') as f:
    ids = pickle.load(f)
with open('dumps/lyap_inter.p', 'rb') as f:
    lyap_inter = pickle.load(f)
with open('dumps/neg_intra.p', 'rb') as f:
    neg_intra = pickle.load(f)
with open('dumps/params.p', 'rb') as f:
    params = pickle.load(f)
with open('dumps/usages.p', 'rb') as f:
    usages = pickle.load(f)
with open('dumps/lyap_intra.p', 'rb') as f:
    lyap_intra = pickle.load(f)

to_remove = [20,28]
for i in to_remove:
    del correlations[i]
    del ids[i]
    del lyap_inter[i]
    del neg_intra[i]
    del params[i]
    del lyap_intra[i]
    
params = np.array(params).astype(np.float)    
ids = np.array(ids)

correlations = correlations[345:]
costs = costs[345:]
ids = ids[345:]
lyap_inter = lyap_inter[345:]
neg_intra = neg_intra[345:]
params = params[345:]
usages = usages[345:]
lyap_intra = lyap_intra[345:]


lyap_inter_min = np.mean(np.abs(np.array(lyap_inter).astype(np.float)), axis=1)
lyap_inter_min = np.min(np.abs(np.array(lyap_inter).astype(np.float)), axis=1)
lyap_inter_min = np.min(np.array(lyap_inter).astype(np.float), axis=1)
lyap_inter_min = np.min(np.abs(np.array(correlations).astype(np.float)), axis=1)

lyap_intra_min = np.mean(np.max(np.abs(np.array(lyap_intra)),axis=1),axis=1)
lyap_intra_min = np.mean(np.min(np.array(lyap_intra),axis=1),axis=1)
lyap_intra_min = np.mean(np.mean(lyap_intra,axis=1),axis=1)

lyap_intra_min = np.mean(np.max(np.abs(np.array(lyap_intra)),axis=1),axis=1)
lyap_abs_mean = np.mean(np.mean(np.abs(lyap_intra),axis=1),axis=1)

sorted_lyap = (lyap_abs_mean - lyap_intra_min).argsort()

#s = np.argsort(lyap_intra_min)[::-1].argsort() # falsch rum, groß gut
#t = np.max(np.abs(correlations),axis=1).argsort()[::-1].argsort() # falsch rum, groß gut
#u = np.min(np.array(lyap_inter).astype(np.float), axis=1).argsort().argsort()[::-1]
s = lyap_intra_min / np.max(lyap_intra_min)
t = np.max(np.abs(correlations),axis=1)
t = t / np.max(t)

tmp = np.mean(np.array([s,t]),axis=0).argsort()



to_plot = np.mean(np.array([s,t]),axis=0)

zipped = list(zip(params[:,0].astype(str),params[:,2].astype(str),to_plot))

print(ids[tmp])

plt.figure()
plt.plot(range(len(lyap_inter)),lyap_inter)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(params[:,0],params[:,1], params[:,2], c=lyap_inter_min, cmap=plt.hot())
ax.set_xlabel('Model_update'); ax.set_ylabel('Memory_limit'); ax.set_zlabel('Learning_rate');
fig.colorbar(img)
plt.title('3d plot')

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

plt.figure()
plt.subplot(1,3,1)
plt.scatter(params[:,0], lyap_inter_min)
plt.xlabel('target_model_update')
plt.ylabel('Percentage of episodes w/ negative lyap exp')
plt.subplot(1,3,2)
plt.scatter(ids, lyap_inter_min)
plt.xlabel('memory_limit')
plt.subplot(1,3,3)
plt.scatter(params[:,2], lyap_inter_min)
plt.xlabel('alpha')

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
plt.scatter(params[:,0], neg_intra)
plt.xlabel('target_model_update')
plt.ylabel('Percentage of episodes w/ negative lyap exp')
plt.subplot(1,3,2)
plt.scatter(ids, neg_intra)
plt.xlabel('memory_limit')
plt.subplot(1,3,3)
plt.scatter(params[:,2], neg_intra)
plt.xlabel('alpha')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(params[:,0],params[:,2], neg_intra, c='b')
ax.set_xlabel('Model_update'); ax.set_ylabel('Learning_rate');
plt.title('3d plot')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(params[:,0],params[:,2], lyap_inter_min, c='b')
ax.set_xlabel('Model_update'); ax.set_ylabel('Learning_rate');
plt.title('3d plot')



ax = plt.axes(projection='3d')
ax.plot_trisurf(params[:,0], params[:,2], lyap_inter_min,
                cmap='viridis', edgecolor='none');

ax = plt.axes(projection='3d')
ax.plot_trisurf(params[:,0], params[:,2], neg_intra,
                cmap='viridis', edgecolor='none');
                
                
plt.figure()
plt.subplot(1,3,1)
plt.scatter(params[:,0], lyap_intra_min)
plt.xlabel('target_model_update')
plt.ylabel('Percentage of episodes w/ negative lyap exp')
plt.subplot(1,3,2)
plt.scatter(ids, lyap_intra_min)
plt.xlabel('memory_limit')
plt.subplot(1,3,3)
plt.scatter(params[:,2], lyap_intra_min)
plt.xlabel('alpha')

plt.figure()
plt.subplot(1,3,1)
plt.scatter(params[:,0], lyap_intra_min)
plt.xlabel('target_model_update')
plt.ylabel('Percentage of episodes w/ negative lyap exp')
plt.subplot(1,3,2)
plt.scatter(params[:,1], lyap_intra_min)
plt.xlabel('memory_limit')
plt.subplot(1,3,3)
plt.scatter(params[:,2], lyap_intra_min)
plt.xlabel('alpha')


plt.figure()
plt.subplot(1,3,1)
plt.scatter(params[:,0], to_plot)
plt.xlabel('target_model_update')
plt.ylabel('Percentage of episodes w/ negative lyap exp')
plt.subplot(1,3,2)
plt.scatter(ids, to_plot)
plt.xlabel('memory_limit')
plt.subplot(1,3,3)
plt.scatter(params[:,2], to_plot)
plt.xlabel('alpha')

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
bp = ax.violinplotpy

fig = plt.figure()                
ax = plt.axes(projection='3d')
ax.plot_trisurf(params[:,0], params[:,2], tmp,
                cmap='viridis', edgecolor='none');
                
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(params[:,0],params[:,2], tmp, c='b')
ax.set_xlabel('Model_update'); ax.set_ylabel('Learning_rate');
plt.title('3d plot')


plt.figure()
ax1 = plt.subplot(2,1,1)
sorted_to_plot = np.array(sorted(zipped,key=lambda x:x[0].astype(float)))
plt.scatter(sorted_to_plot[:,0],sorted_to_plot[:,2].astype(float),c=sorted_to_plot[:,1].astype(float),s=75,cmap='viridis_r')#,vmax=0.04)
plt.colorbar().set_label('Learning rate')
plt.xlabel('Target Model Update')
plt.ylabel('Relevance of trajectory dynamics')
ax1.add_patch(matplotlib.patches.Rectangle((-.5,1.2),7,-0.705,color='grey',alpha=0.15))
ax2 =plt.subplot(2,1,2)
sorted_to_plot = np.array(sorted(zipped,key=lambda x:x[1].astype(float)))
plt.scatter(sorted_to_plot[:,1],sorted_to_plot[:,2].astype(float),c=sorted_to_plot[:,0].astype(float),s=75,cmap='viridis_r')#,vmax=400)
plt.xlabel('Learning rate')
plt.colorbar().set_label('Target model update')
plt.xticks(sorted_to_plot[:,1],[round(float(x),5) for x in sorted_to_plot[:,1]],rotation='vertical')
plt.ylabel('Relevance of trajectory dynamics')
ax2.add_patch(matplotlib.patches.Rectangle((-2,1.2),30,-0.705,color='grey',alpha=0.15))
#plt.suptitle('Ranking of experiments (low -> interesting dynamics)')


x = np.arange(0,4*np.pi,0.1)   # start,stop,step
y = np.sin(x)
z = np.cos(x)
plt.figure()
plt.plot(x,y,x,z)
plt.xticks([], [])
plt.yticks([], [])
plt.xlabel('Timestep t')
plt.legend(['Insurance price','Insurance usage'])



lyap_intra_min = np.mean(np.max(np.abs(np.array(lyap_intra)),axis=1),axis=1)
lyap_abs_mean = np.mean(np.mean(np.abs(lyap_intra),axis=1),axis=1)

sorted_lyap = (lyap_abs_mean - lyap_intra_min).argsort()

s = np.argsort(lyap_intra_min)[::-1].argsort() # falsch rum, groß gut
t = np.max(np.abs(correlations),axis=1).argsort()[::-1].argsort() # falsch rum, groß gut
#u = np.min(np.array(lyap_inter).astype(np.float), axis=1).argsort().argsort()[::-1]
tmp = np.mean(np.array([s,t]),axis=0).argsort()



to_plot = np.mean(np.array([s,t]),axis=0)

zipped = list(zip(params[np.where(to_plot>0.505),0].astype(str)[0],params[np.where(to_plot>0.505),2].astype(str)[0],to_plot[np.where(to_plot>0.505)]))

plt.figure()
plt.subplot(2,1,1)
sorted_to_plot = np.array(sorted(zipped,key=lambda x:x[0].astype(float)))
plt.scatter(sorted_to_plot[:,0],sorted_to_plot[:,2].astype(float),c=sorted_to_plot[:,1].astype(float),s=75,cmap='viridis_r')#,vmax=0.01)
plt.colorbar().set_label('Learning rate')
plt.xlabel('Target Model Update')
plt.ylabel('Relevance of trajectory dynamics')
plt.subplot(2,1,2)
sorted_to_plot = np.array(sorted(zipped,key=lambda x:x[1].astype(float)))
plt.scatter(sorted_to_plot[:,1],sorted_to_plot[:,2].astype(float),c=sorted_to_plot[:,0].astype(float),s=75,cmap='viridis_r')#,vmax=400)
plt.xlabel('Learning rate')
plt.colorbar().set_label('Target model update')
plt.xticks(sorted_to_plot[:,1],[round(float(x),5) for x in sorted_to_plot[:,1]],rotation='vertical')
plt.ylabel('Relevance of trajectory dynamics')
#plt.suptitle('Ranking of experiments (low -> interesting dynamics)')