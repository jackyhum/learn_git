import numpy as np
from tqdm import tqdm, trange
from scipy.sparse import coo_matrix, hstack,vstack
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import random
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import deepxde as dde
from timeit import default_timer
from torchviz import make_dot,make_dot_from_trace
import os



def count_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_cof_equ_K_19(i, N1, f1_m, hx, a):
    num = N1 - i - 2
    coff = f1_m[i, i+1:N1-1].T
    cofi1 = coo_matrix((2 * np.ones(num), (range(num), range(num))), shape = (num, num)).tocsr()
    cofi1 = cofi1 + coo_matrix((-1 * np.ones((num - 1)), (range(num - 1),range(1, num))),shape = (num, num)).tocsr()
    cofi3_1 = np.tril(np.ones((num, num))) - np.diag(1 / 2 * np.ones(num))
    cofi3_1 = hx * cofi3_1
    cofi3_2 = np.zeros((num,num))
    for k in range(num):
        cofi3_2[k, 0:k] = f1_m[i + 1: i + k + 1, i + k + 1].T
    cofi3 = -hx / (a) * cofi3_1 * cofi3_2
    cofi = cofi1 + coo_matrix(cofi3).tocsr() 
    return cofi, coff

def solveBetaFunction(x, gamma, amp):
    beta = np.zeros(len(x))
    for idx, val in enumerate(x):
        beta[idx] = amp*np.cos(gamma*np.arccos(val))
    return beta

def buildF(x, gamma1, amp1, gamma2, amp2):
    b1 = solveBetaFunction(x, gamma1, amp1)
    b2 = solveBetaFunction(x, gamma2, amp2)
    nx = len(x)
    f = np.zeros((nx, nx))
    for idx, val in enumerate(x):
        for idx2, val2 in enumerate(x):
            if idx2>=idx:
                f[idx][idx2] = b1[idx]*b2[idx2]
    return f

def KernelCalc2(a, hx, N1, f1_m): 
    #构造系数矩阵
    fii = np.diag(f1_m)
    cofmed = np.zeros((N1, N1))
    cofmed[:, -1] = 1/ 2 * np.ones((1, N1))
    cof_int_zeta_1 = hx * (np.triu(np.ones((N1, N1))) - cofmed - np.diag(1 / 2 * np.ones(N1)))
    K_ii = -1 / a * np.dot(cof_int_zeta_1, fii)
    K = np.diag(K_ii)

    for i in range(N1 - 3, -1, -1 ):
        [cofi, coff] = get_cof_equ_K_19(i,N1,f1_m,hx,a)
        Kibud = K[i, i] * hx * hx / (2 * a) - hx / a
        Ki = np.linalg.solve(cofi.toarray(), K[i+1,i+1:N1-1].T) + np.linalg.solve(cofi.toarray(), np.dot(Kibud, coff))
        K[i, i+1:N1-1]=Ki.T  
    Kbud = K[0, :]
    return K, Kbud 




a = 1
X = 1
dx = 1 / 20
nx = int(round(X/dx))+1
spatial = np.linspace(0, X, nx, dtype=np.float32)
N1 = len(spatial)
N = N1 - 1
#神经网络的设计                     
# Parameters
epochs =400
ntrain = 900
ntest = 100

gamma = 0.5
learning_rate = 0.01
step_size= 50
modes=12
width=32
batch_size = 40#训练批数

c1 = 20 - 20 * spatial
[eta1_m, zeta1_m]= np.meshgrid(spatial,spatial)

my_dpi=300





f = buildF(spatial, np.random.uniform(-10, 10), 8, np.random.uniform(-10, 10), 8)
kernel, _ = KernelCalc2(a, dx, N1, f)


[Time, Zeta] = np.meshgrid(spatial, spatial)
fig = plt.figure()#生成下标为1的画布
ax = fig.add_subplot(2, 1, 1, projection='3d')
ax.plot_surface(Time, Zeta, f, alpha = 0.9, cmap = "rainbow")

ax = fig.add_subplot(2, 1, 2, projection='3d')
ax.plot_surface(Time, Zeta, kernel, alpha = 0.9, cmap = "rainbow")
        

# ax.set_zlabel(r"$k(x,y)$", fontsize =10, fontproperties="Times New Roman")

plt.show()
print("ok")
