import tensorcircuit as tc
from qft import QFT, IQFT
from addition import add, cond_add
from tools import *
import numpy as np

@block
def add_const(x, value):
	c = tc.Circuit(max(x) + 1)
	n = len(x)
	c.append(QFT(x))

	for i in [i for i in range(n) if (value >> i & 1)]:
		for j in range(n):
			if i + j <= n:
				target_idx = n - 1 - j
				c.phase(x[target_idx], theta = np.pi * (2 ** (i + j + 1)) / (2 ** n))

	c.append(IQFT(x))
	return c

@block
def cond_add_const(p, x, value):
	c = tc.Circuit(max(p, max(x)) + 1)
	n = len(x)
	c.append(QFT(x))

	for i in [i for i in range(n) if (value >> i & 1)]:
		for j in range(n):
			if i + j <= n:
				target_idx = n - 1 - j
				c.cphase(p, x[target_idx], theta = np.pi * (2 ** (i + j + 1)) / (2 ** n))

	c.append(IQFT(x))
	return c

@block
def cond_cadd_const(p1, p2, x, value):
	n = len(x)
	c = tc.Circuit(max([p1, p2] + x) + 1)
	c.append(QFT(x))

	for i in [i for i in range(n) if (value >> i & 1)]:
		for j in range(n):
			if i + j <= n:
				target_idx = n - 1 - j
				c.append(ccphase(theta = np.pi * (2 ** (i + j + 1)) / (2 ** n)), indices = [p1] + [p2] + [x[target_idx]])

	c.append(IQFT(x))
	return c
