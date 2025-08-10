import tensorcircuit as tc
import numpy as np
from tools import *

def QFT(x):
    c = tc.Circuit(max(x) + 1)
    n = len(x)
    
    for i in range(n-1, -1, -1):
        c.H(x[i])
        for j in range(i):
            c.cphase(x[j], x[i], theta = np.pi / 2 ** (i - j))
    
    return c

def IQFT(x):
	c = tc.Circuit(max(x) + 1)
	n = len(x)
	
	for i in range(0, n):
		for j in range(0, i):
			c.cphase(x[j], x[i], theta = -np.pi / 2 ** (i - j))
		c.H(x[i])

	return c

def cQFT(p, x):
	c = tc.Circuit(max(p, max(x)) + 1)
	n = len(x)

	for i in range(n-1, -1, -1):
		controlled_H(c, p, x[i])
		for j in range(i):
			ccphase(c, p, x[j], x[i], theta = np.pi / 2 ** (i - j))
	
	return c

def cIQFT(p, x):
	c = tc.Circuit(max(p, max(x)) + 1)
	n = len(x)

	for i in range(0, n):
		for j in range(0, i):
			ccphase(c, p, x[j], x[i], theta = -np.pi / 2 ** (i - j))
		controlled_H(c, p, x[i])

	return c