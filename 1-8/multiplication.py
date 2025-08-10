import tensorcircuit as tc
from modular_addition import * 
from constant_addition import *
from tools import *

@block
def mod_doubling():
	x = [0, 1, 2, 3]
	c = tc.Circuit(len(x))
	for i in reversed(range(len(x) - 1)):
		c.SWAP(x[i], x[i + 1])
	c.append(add_const(x, 16 - 7))
	c.append(cond_add_const(x[-1], x[0:-1], 7))
	c.X(x[0])
	c.CNOT(x[0], x[-1])
	c.X(x[0])
	return c

def mod_multiplication(x, y, o, z):
	c = tc.Circuit(max(x + y + o + z) + 1)
	for i in range(3):
		c.append(cond_mod_add(9, [0, 1, 2, 3], [4, 5, 6, 7], [8]), indices = y + o + [z[0]] + [i])
		c.append(mod_doubling(), indices = y)
	for i in range(3):
		c.append(mod_doubling().inverse(), indices = y)
	return c

@block
def mod_inverse():
	c = tc.Circuit(4)
	c.SWAP(1, 2)
	return c

def mod_square(x, y):
	c = tc.Circuit(max(x + y) + 1)
	n = 4  
	for j in range(n):
		for k in range(n):
			exp = (2 ** (j + k)) % 7
			if exp != 0:
				c.append(add_const(y, exp), indices=[x[j], x[k]] + y)
	return c
# 这里是对 x 的平方取模 7 的实现，若报错可尝试以下更简单粗暴的方法
'''def mod_square(x, y):
	c = tc.Circuit(max(x + y) + 1)
	c.cnot(x[0], y[0])
	c.cnot(x[1], y[1])
	c.cnot(x[2], y[2])

	c.swap(y[1], y[2])
	c.toffoli(x[0], x[1], x[3])
	c.toffoli(x[0], x[2], x[3])
	c.toffoli(x[1], x[2], x[3])

	c.cnot(x[3], y[0])
	c.cnot(x[3], y[1])
	c.cnot(x[3], y[2])

	c.toffoli(x[0], x[1], x[3])
	c.toffoli(x[0], x[2], x[3])
	c.toffoli(x[1], x[2], x[3])

	return c'''
