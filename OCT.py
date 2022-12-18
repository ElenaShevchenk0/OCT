import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.ticker as ticker
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)


class OptimalDecisionTrees:
    
    def __init__(self, max_depth=2, min_leaf_samples=1, alpha=0):
        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples

        # index for all nodes in a tree
        self.n_ind = [i + 1 for i in range(2 ** (self.max_depth + 1) - 1)]
        # index for branch nodes
        self.b_ind = self.n_ind[:-2**self.max_depth]
        # index for leaf nodes
        self.l_ind = self.n_ind[-2**self.max_depth:]
        # index left nodes
        self.left_nodes = self.n_ind[1::2]
        # alpha
        self.alpha = alpha
        
    def fit(self, x, y):
        """
        fit the data to the model
        """
        # class labels
        self.labels = np.unique(y)
        # number of data points
        self.n = x.shape[0]
        # number of features
        self.p = x.shape[1]
        # scale data to b\w 0 and 1
        if np.min(x) >= 0:
            if np.max(x) <= 1:
                x = x
            else:
                self.scale = np.max(x, axis=0)
                self.scale[self.scale == 0] = 1
                x = x / self.scale
        else:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            
        # solve MIP
        m, a, b, c, d, l, z, eps, K = self._constructMIO(x,y)
        # set parameters to gurobi solver
        m.setParam('Timelimit', 1800)
        m.setParam('MIPGap', 0.03)
        
        # optimize
        m.optimize()
        # write formulation of the problem to a file
        m.write('model.lp')

        # get parameters
        self.a_ = {ind:a[ind].x for ind in a}
        self.b_ = {ind:b[ind].x for ind in b}
        self.c_ = {ind:c[ind].x for ind in c}
        self.d_ = {ind:d[ind].x for ind in d}
        self.l_ = {ind:l[ind].x for ind in l}
        self.z_ = {ind:z[ind].x for ind in z}
        self.K_ = {ind:K[ind].x for ind in K}
        self.eps_ = eps
        
        
    def predict(self, x):
        """
        make prediction for each given sample
        """
        # scale data to b\w 0 and 1
        if np.min(x) >= 0:
            if np.max(x) <= 1:
                x = x
            else:
                self.scale = np.max(x, axis=0)
                self.scale[self.scale == 0] = 1
                x = x / self.scale
        else:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            
        # map leaf node index to class prediction
        node_class = {}      
        for k, v in self.c_.items():
            if int(v) == 1:
                node_class[k[1]] = k[0]
        print('node_class: ', node_class)
        
        # predict y for given x
        pred = []
        for row in x:
            t = 1
            while t in self.b_ind:
                lhs = sum([self.a_[j,t] * row[j] for j in range(self.p)])
                if lhs >= self.b_[t]:
                    t = 2 * t + 1
                else:
                    t = 2 * t
            pred.append(node_class[t])
        return pred
            
    def _constructMIO(self, x, y):
        """
        MIO formulation for ICOT
        """

        # create a model
        m = gp.Model('m')

        # create variables
        a = m.addVars(self.p, self.b_ind, vtype=GRB.BINARY, name='a') # splitting feature
        b = m.addVars(self.b_ind, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='b') # splitting threshold
        c = m.addVars(self.labels, self.l_ind, vtype=GRB.BINARY, name='c') # average distance of i from cluster t
        d = m.addVars(self.b_ind, vtype=GRB.BINARY, name='d') # splitting option
        z = m.addVars(self.n, self.n_ind, vtype=GRB.BINARY, name='z') # leaf node assignment
        l = m.addVars(self.l_ind, vtype=GRB.BINARY, name='l') # leaf node activation
        L = m.addVars(self.l_ind, vtype=GRB.CONTINUOUS, name='L') # leaf node misclassified
        N_k = m.addVars(self.labels, self.l_ind, vtype=GRB.CONTINUOUS, name='M') # leaf node samples with label
        N = m.addVars(self.l_ind, vtype=GRB.CONTINUOUS, name='N') # leaf node samples
        K = m.addVars(self.l_ind, lb=0, ub=self.n-1, vtype=GRB.INTEGER, name='K')

        # compute baseline accuracy L_hat
        L_hat = self._comp_L_hat(y)
        
        # objective function
        m.setObjective(L.sum() / L_hat + self.alpha * d.sum(), GRB.MINIMIZE)
        
        # constraints
        m.addConstrs(L[t] >= N[t] - N_k[k,t] - self.n * (1 - c[k,t]) for t in self.l_ind for k in self.labels)
        m.addConstrs(L[t] <= N[t] - N_k[k,t] + self.n * c[k,t] for t in self.l_ind for k in self.labels)
        m.addConstrs(L[t] >= 0 for t in self.l_ind)
        m.addConstrs(gp.quicksum((y[i] == k) * z[i,t] for i in range(self.n)) == N_k[k,t] for t in self.l_ind for k in self.labels)
        m.addConstrs(gp.quicksum(z[i,t] for i in range(self.n)) == N[t] for t in self.l_ind)
        m.addConstrs(gp.quicksum(c[k,t] for k in self.labels) == l[t] for t in self.l_ind)
        m.addConstrs(gp.quicksum(z[i,t] for t in self.l_ind) == 1  for i in range(self.n))
        m.addConstrs(z[i,t] <= l[t] for t in self.l_ind for i in range(self.n))
        m.addConstrs(gp.quicksum(z[i,t] for i in range(self.n)) >= self.min_leaf_samples * l[t] for t in self.l_ind)
        m.addConstrs(d[t] == gp.quicksum(a[j,t] for j in range(self.p)) for t in self.b_ind)
        m.addConstrs(b[t] <= d[t] for t in self.b_ind)
        m.addConstrs(K[t] == gp.quicksum(z[i,t] for i in range(self.n)) for t in self.l_ind)
        
        # compute epsilon
        eps = self._comp_eps(x)
        
        # find ancestors and set constraints for hierarchy 
        for t in self.n_ind:
            ancestors = []
            anc = t // 2
            if t > 1:
                ancestors.append(t)
                while anc != 0:
                    ancestors.append(anc)
                    anc = anc // 2
                for k in range(len(ancestors) - 1):
                    if ancestors[k] in self.left_nodes:
                        m.addConstrs(gp.quicksum(a[j,ancestors[k+1]] * (x[i,j] + eps[j]) for j in range(self.p))
                                     +
                                     (1 + np.max(eps)) * (1 - d[ancestors[k+1]])
                                     <=
                                     b[ancestors[k+1]] + (1 + np.max(eps)) * (1 - z[i,t]) for i in range(self.n))
                    else:
                        m.addConstrs(gp.quicksum(a[j,ancestors[k+1]] * x[i,j] for j in range(self.p))
                        >=
                        b[ancestors[k+1]] - (1 - z[i,t]) for i in range(self.n))
        return m, a, b, c, d, l, z, eps, K

    @staticmethod
    def _comp_eps(x):
        """
        compute the minimum distance among all observations within each feature
        """
        eps = []
        for j in range(x.shape[1]):
            xj = x[:,j]
            # drop duplicates
            xj = np.unique(xj)
            # sort
            xj = np.sort(xj)[::-1]
            # distance
            e = [1]
            for i in range(len(xj)-1):
                e.append(xj[i] - xj[i+1])
            # min distance
            eps.append(np.min(e) if np.min(e) else 1)
        return eps
    
    @staticmethod
    def _comp_L_hat(y):
        """
        compute baseline accuracy L_h as the most popular class in the dataset
        """
        L_hat = max(y, key=list(y).count)
        return L_hat



def tree_picture(model_fit, depth, x_train):
    """
    Plots the tree with a given depth.
    Parameters:
    model_fit: OCT model
    depth: depth of the tree
    x_train: data set
    """
    # list with the number of nodes per level
    node_per_level = [2**i for i in range(depth+1)]
    # indexes for each node in the tree
    nodes_ind = [i + 1 for i in range(2 ** (depth + 1) - 1)]
    # leaf nodes indexes
    leaf_ind = nodes_ind[-2**depth:]
    # create rectangles by coordinates
    x_rectangles = list(np.arange(1,sum(node_per_level)+1,1))
    x_curr = [1+3*i for i in range(node_per_level[-1])]
    y_rectangles = []
    x_start_edges = []
    for i in range(len(node_per_level)-1,-1,-1):
        node_indx = x_rectangles[node_per_level[i]-1:node_per_level[i]*2 - 1]
        for j in range(len(node_indx)):
            x_rectangles[node_indx[j]-1] = x_curr[j]
        x_curr = [(x_curr[::2][k] + x_curr[1::2][k] + 2)/2 - 1 for k in range(int(len(x_curr)/2))]  
    for i in range(len(node_per_level)):
        y_curr = [1+2*i] * node_per_level[::-1][i] 
        y_rectangles += y_curr
    y_rectangles = y_rectangles[::-1]
    
    # create edges connecting rectangles
    x_left_start = x_rectangles[:node_per_level[-1]-1]
    x_right_start = [x + 2 for x in x_left_start]
    y_left_start = y_rectangles[:node_per_level[-1]-1]
    y_right_start = y_rectangles[:node_per_level[-1]-1]
    x_start = sum([[x_left_start[i], x_right_start[i]] for i in range(len(x_left_start))], [])
    y_start = sum([[y_left_start[i], y_right_start[i]] for i in range(len(y_left_start))], [])
    x_end = [x + 1 for x in x_rectangles[1:]]
    y_end = sum([[2+2*i] * node_per_level[::-1][i] for i in range(len(node_per_level)-1)],[])[::-1]
    
    # draw recrangles
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    ax.set_ylim(0, 2*(depth+1) + 1)
    ax.set_xlim(0, 3*(2**depth) + 1)
    for i in range(len(x_rectangles)):
        ax.add_patch(Rectangle((x_rectangles[i], y_rectangles[i]), 2, 1, fill=None, alpha=1, lw=2))
        
    # create list for d (split)
    d = [v for k,v in model_fit.d_.items()]
    
    # create list for b (threshold)
    b = [round(v,5) for k,v in model_fit.b_.items()]

    # create list for a (feature to split on)
    a = [[0] * x_train.shape[1] for _ in range(sum(node_per_level) - node_per_level[-1])]
    for k, v in model_fit.a_.items():
        if int(v) == 1:
            a[k[1]-1][k[0]] = 1
            
    # create list for c (class prediction) 
    c = [-1] * node_per_level[-1]
    for k, v in model_fit.c_.items():
        if v == 1:
            c[k[1] - node_per_level[-1]] = k[0]
            
    # create list for K (sample size in each leaf)
    K = [np.round(v) for k,v in model_fit.K_.items()]
    
    # create list for epsilon
    epsilon = [np.round(e,5) for e in model_fit.eps_]

    # add info for brancg nodes
    for i in range(len(x_rectangles[:node_per_level[-1]-1])):
        a_ind = [int(list(np.where(a[i])[0] + 1)[0]) if int(np.max(a[i])) > 0 else -1]
        if a_ind[0] > 0:
            eps_j = epsilon[a_ind[0]-1]
            b[i] = b[i] - eps_j
        else:
            a_ind[0] = None
        ax.text([x + 0.1 for x in x_rectangles[:node_per_level[-1]-1]][i], [y + 0.6 for y in y_rectangles[:node_per_level[-1]-1]][i], f'split = {bool(int(abs(d[i])))}', fontsize=32/depth)
        ax.text([x + 0.1 for x in x_rectangles[:node_per_level[-1]-1]][i], [y + 0.2 for y in y_rectangles[:node_per_level[-1]-1]][i], f'x_{a_ind[0]} <= {np.round(b[i],3)}', fontsize=32/depth)
        
    # add info for leaf nodes
    for i in range(len(x_rectangles[node_per_level[-1]-1:])):
        ax.text([x + 0.2 for x in x_rectangles[node_per_level[-1]-1:]][i], [y + 0.4 for y in y_rectangles[node_per_level[-1]-1:]][i], f'samples={int(K[i])}', fontsize=32/depth)
        ax.text([x + 0.2 for x in x_rectangles[node_per_level[-1]-1:]][i], [y + 0.7 for y in y_rectangles[node_per_level[-1]-1:]][i], f'class={int(c[i])}', fontsize=32/depth)
    
    # plot edges
    for i in range(len(x_start)):
        plt.plot([x_end[i],x_start[i]], [y_end[i],y_start[i]], '-', color='black')

    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    plt.show()