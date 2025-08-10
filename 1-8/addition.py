import numpy as np
import tensorcircuit as tc
from qft import QFT, IQFT
from tools import *

@block
def add(x, y):
	c = tc.Circuit(max(x + y) + 1)
	c.append(QFT(y))
	n = len(y)

	for i in range(n):
		for j in range(n):
			if i + j <= n:
				target_idx = n - 1 - j
				c.cphase(x[i], y[target_idx], theta = np.pi * (2 ** (i + j + 1)) / (2 ** n))

	c.append(IQFT(y))
	return c

@block
def cond_add(p, x, y):
	c = tc.Circuit(max([p] + x + y) + 1)
	c.append(QFT(y))
	n = len(y)

	for i in range(n):
		for j in range(n):
			if i + j <= n:
				target_idx = n - 1 - j
				c.append(ccphase(theta = np.pi * (2 ** (i + j + 1)) / (2 ** n)), indices = [p] + [x[i]] + [y[target_idx]])

	c.append(IQFT(y))
	return c