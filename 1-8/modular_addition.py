import tensorcircuit as tc
from qft import QFT, IQFT
from addition import add, cond_add
from tools import *
from constant_addition import add_const, cond_add_const, cond_cadd_const
import numpy as np

@block
def mod_add(x, y, z):
	p = 7
	n = max(x , y, z)
	c = tc.Circuit(max(x + y + z) + 1)
	c.append(add(x, y))
	c.append(add_const(y, 2 ** n - p))

	c.cnot(y[n-1], z[0])
	c.append(cond_add_const(z[0], y, p))

	c.append(add(x, y).inverse())
	c.x(y[n-1])
	c.cnot(y[n-1], z[0])
	c.x(y[n-1])
	c.append(add(x, y))
	return c

@block
def cond_mod_add(p, x, y, z):
	n = max(x, y, z)
	p = 7
	c = tc.Circuit(max([p] + x + y + z) + 1)
	c.append(cond_add(p, x, y))
	c.append(cond_add_const(p, y, 2 ** n - p))

	c.toffoli(p, y[n-1], z[0])
	c.append(cond_cadd_const(p, z[0], y, p))

	c.append(cond_add(p, x, y).inverse())
	c.x(y[n-1])
	c.toffoli(p, y[n-1], z[0])
	c.x(y[n-1])
	c.append(cond_add(p, x, y))
	return c


@block
def negation(x, z):
	n = max(x , z)
	p = 7
	c = tc.Circuit(max(x + z) + 1)
	for i in range(n-1):
		c.x(x[i])

	c.append(add_const(x, 1))
	c.cnot(x[n-1], z[0])
	c.append(cond_add_const(z[0], x, 2 ** n - p))
	c.append(add_const(x, 2 ** n - p - 1))
	c.x(x[n-1])
	c.cnot(x[n-1], z[0])
	c.x(x[n-1])
	c.append(add_const(x, 2 ** n - p))
	return c

@block
def cond_negation(p, x, z):
	n = max(p, x, z)
	p = 7
	c = tc.Circuit(max([p] + x + z) + 1)
	for i in range(n-1):
		c.cnot(p, x[i])
	
	c.append(cond_add_const(p, x, 1))
	c.toffoli(p, x[n-1], z[0])
	c.append(cond_cadd_const(p, z[0], x, 2 ** n - p))
	c.append(cond_add_const(p, x, 2 ** n - p - 1))
	c.x(x[n-1])
	c.toffoli(p, x[n-1], z[0])
	c.x(x[n-1])
	c.append(cond_add_const(p, x, 2 ** n - p))
	return c

@block
def add_mod_const(x, val, z):
	n = max(p, x, z)
	p = 7
	c = tc.Circuit(max(x + z) + 1)
	c.append(add_const(x, val))
	c.append(add_const(x, 2 ** n - p))

	c.cnot(x[n-1], z[0])
	c.append(cond_add_const(z[0], x, 2 ** n - p))

	c.append(add_const(x, val).inverse())
	c.x(x[n-1])
	c.cnot(x[n-1], z[0])
	c.x(x[n-1])
	c.append(add_const(x, val))
	return c

@block
def cadd_mod_const(ctrl, target, value, flag):
	circuit = tc.Circuit(max([ctrl] + target + flag) + 1)
	circuit.append(cond_add_const(ctrl, target, value))
	circuit.append(add_const(target, 9))

	circuit.cnot(target[3], flag[0])
	circuit.append(cond_add_const(flag[0], target, 7))
    
	circuit.append(cond_add_const(ctrl, target, value).inverse())
	circuit.x(target[3])
	circuit.cnot(target[3], flag[0])
	circuit.x(target[3])
	circuit.append(cond_add_const(ctrl, target, value))
	return circuit

@block
def ccadd_mod_const(ctrl1, ctrl2, target, value, flag):
	circuit = tc.Circuit(max([ctrl1, ctrl2] + target + flag) + 1)
	circuit.append(cadd_mod_const(0, [1, 2, 3, 4], value * 4 % 7, [5]), indices=[ctrl1] + target + [flag[0]])
	circuit.cnot(ctrl2, ctrl1)
	circuit.append(cadd_mod_const(0, [1, 2, 3, 4], value * 3 % 7, [5]), indices=[ctrl1] + target + [flag[0]])
	circuit.cnot(ctrl2, ctrl1)
	circuit.append(cadd_mod_const(0, [1, 2, 3, 4], value * 4 % 7, [5]), indices=[ctrl2] + target + [flag[0]])
	return circuit
