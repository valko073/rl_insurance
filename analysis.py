import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

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
    
params = np.array(params).astype(np.float)

correlations = correlations[345:]
costs = costs[345:]
ids = ids[345:]
lyap_inter = lyap_inter[345:]
neg_intra = neg_intra[345:]
params = params[345:]
usages = usages[345:]
layp_intra = lyap_intra[345:]

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

s = np.argsort(lyap_intra_min).argsort() # falsch rum, groß gut
t = np.max(np.abs(correlations),axis=1).argsort().argsort() # falsch rum, groß gut
#u = np.min(np.array(lyap_inter).astype(np.float), axis=1).argsort().argsort()[::-1]
tmp = np.mean(np.array([s,t]),axis=0).argsort()
to_plot = np.mean(np.array([s,t]),axis=0)
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

fig = plt.figure()
plt.subplot(1,2,1)
x=np.array(sorted(zip(params[:,0].astype(str),to_plot,params[:,2]),key=lambda x:float(x[0])))
img = plt.scatter(x[:,0],x[:,1].astype(float),c=x[:,2].astype(float),cmap='jet',norm=matplotlib.colors.Normalize(0,0.3),s=100)
fig.colorbar(img)
plt.xlabel('target_model_update')
plt.ylabel('Percentage of episodes w/ negative lyap exp')
plt.subplot(1,2,2)
x=np.array(sorted(zip(params[:,2].astype(str),to_plot,params[:,0]),key=lambda x:float(x[0])))
img = plt.scatter(x[:,0],x[:,1].astype(float),c=x[:,2].astype(float),cmap='jet',s=100)
fig.colorbar(img)
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