#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from transformer import fn_transformer
import matplotlib.pyplot as plt
import time


# In[2]:


class PSO():
	def __init__(self, func, dim, pop=40, max_iter=150, lb=None, ub=None, w=0.8, c1=0.5, c2=0.5, timeplotbool=True):
		self.func = fn_transformer(func)
		self.w = w  # inertia
		self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
		self.pop = pop  # number of particles
		self.dim = dim  # dimension of particles, which is the number of variables of func
		self.max_iter = max_iter  # max iter

		self.has_constraints = not (lb is None and ub is None)
		self.lb = -np.ones(self.dim) if lb is None else np.array(lb)
		self.ub = np.ones(self.dim) if ub is None else np.array(ub)
		assert self.dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
		assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

		self.X = np.random.randint(self.lb+1, self.ub+1, size=(self.pop, self.dim) )
		self.V = np.random.randint(self.lb+1, self.ub+1, size=(self.pop, self.dim) )

		self.Y = self.cal_y()  # y = f(x) for all particles
		self.pbest_x = self.X.copy()  # personal best location of every particle in history
		self.pbest_y = self.Y.copy()  # best image of every particle in history
		self.gbest_x = np.zeros((1, self.dim))  # global best location for all particles
		self.gbest_y = 0  # global best y for all particles
		self.gbest_y_hist = []  # gbest_y of every iteration
		self.update_gbest()

		# record verbose values
		self.record_mode = False
		self.record_value = {'X': [], 'V': [], 'Y': []}

		self.xaxis = []; self.yaxis = [];
		self.timeplotbool = timeplotbool;

	def update_V(self):
		r1 = np.random.randint(1, size=(self.pop, self.dim) )
		r2 = np.random.randint(1, size=(self.pop, self.dim) )
		self.V = self.w * self.V + 		self.cp * r1 * (self.pbest_x - self.X) + 		self.cg * r2 * (self.gbest_x - self.X)

	def update_X(self):
		self.X = self.X + self.V

		if self.has_constraints:
			self.X = np.clip(self.X, self.lb, self.ub)

	def cal_y(self):
		# calculate y for every x in X
		self.Y = self.func(self.X).reshape(-1, 1)
		return self.Y

	def update_pbest(self):
		self.pbest_x = np.where(self.pbest_y < self.Y, self.X, self.pbest_x)
		self.pbest_y = np.where(self.pbest_y < self.Y, self.Y, self.pbest_y)

	def update_gbest(self):
		if self.gbest_y < self.Y.max():
			self.gbest_x = self.X[self.Y.argmax(), :].copy()
			self.gbest_y = self.Y.max()

	def recorder(self):
		if not self.record_mode:
			return
		self.record_value['X'].append(self.X)
		self.record_value['V'].append(self.V)
		self.record_value['Y'].append(self.Y)

	def timeplot(self):
		plt.figure( figsize=(25, 9))
		plt.xlabel("No of Iterations")
		plt.ylabel("Time for iteration")
		plt.plot(self.xaxis, self.yaxis,
			linewidth=4.0, color="#D5DBDB",
			marker="o", mfc="#34495E")
		plt.savefig('PSO time.png')
		plt.show()

	def run(self, max_iter=None):
		self.max_iter = max_iter or self.max_iter
		for iter_num in range(self.max_iter):
			before = time.time()
			self.update_V()
			self.recorder()
			self.update_X()
			self.cal_y()
			self.update_pbest()
			self.update_gbest()
			self.xaxis.append( iter_num )
			self.yaxis.append( time.time()-before )

		self.gbest_y_hist.append(self.gbest_y)
		if self.timeplotbool:
			self.timeplot()
		return self

	fit = run


# In[ ]:




