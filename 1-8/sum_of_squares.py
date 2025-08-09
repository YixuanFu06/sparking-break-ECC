import tensorcircuit as tc
from modular_addition import * 
from tools import *

@block
def add_mod_square(x, y, z):
	c = tc.Circuit(max(x + y + z) + 1)
	n = len(x)

	for i in range(3):
		for j in range(3):
			if i != j:
				c.append(ccadd_mod_const(0, 1, [2, 3, 4, 5], (2 ** (i + j)) % 7, [6]) , indices = [x[i]] + [x[j]] + y + [z[0]])
			if i == j:
				c.append(cadd_mod_const(0, [1, 2, 3, 4], (2 ** (i + i)) % 7, [5]), indices = [x[i]] + y + [z[0]])
	
	return c