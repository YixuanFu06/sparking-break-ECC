import tensorcircuit as tc
from constant_addition import *
from addition import *
from tools import *
from multiplication import *
from modular_addition import *
from sum_of_squares import *

def const_inverse(c, p):
    for i in range(p):
        if(i * c) % p == 1:
            return i





def point_addition(x_1, y_1, x_2, y_2, a, b, p):
    sign = 1
    if x_1 == x_2 and y_1 == y_2:
        if y_1 == 0:
            return (0, 0)
        up = 3*(x_1 ** 2) + a
        down = 2 * y_1
    else:
        up = y_2 - y_1
        down = x_2 - x_1
        if up * down < 0:
            up = abs(up)
            down = abs(down)
            sign = -1

    down = mod_inverse(down, p)
    lam = up * down * sign % p
    x_3 = (lam ** 2 - x_1 - x_2) % p
    y_3 = (lam * (x_1 - x_3) - y_1) % p
    return x_3, y_3

def point_addition_corner(P, Q, a, b, p):
    if P == (0, 0):
        return Q
    if Q == (0, 0):
        return P

    x_1, y_1 = P
    x_2, y_2 = Q

    if x_1 == x_2 and y_1 + y_2 == 0:
        return (0, 0)
    
    point_addition_result = point_addition(x_1, y_1, x_2, y_2, a, b, p)
    return point_addition_result


def cond_ECC_add_0(p, x, x2, y2, z):
	c = tc.Circuit(max([p] + x + z) + 1) # 14 bits

	c.append(add_mod_const([0, 1, 2, 3], (7 - x2) % 7, [4]), indices = x[0:4] + [z[4]])
	c.append(add_mod_const([0, 1, 2, 3], (7 - y2) % 7, [4]), indices = x[4:8] + [z[4]])

	# now : x1 - x2 , y1 - y2 , 0

	c.append(mod_inverse(), indices = x[0:4])

	c.x(p)
	for i in range(4):
		c.cswap(p, x[i], z[i])
	c.x(p)

	c.append(mod_multiplication([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12]), indices = x + z)

	c.x(p)
	for i in range(4):
		c.cswap(p, x[i], z[i])
	c.x(p)

	c.append(mod_inverse(), indices = x[0:4])

	# now : x1 - x2 , y1 - y2 , lambda

	c.append(add_mod_const([0, 1, 2, 3], x2, [4]), indices = x[0:4] + [z[4]])
	c.append(add_mod_const([0, 1, 2, 3], y2, [4]), indices = x[4:8] + [z[4]])

	# output(c, length = 15)
	# now : x1 , y1 , lambda

	c.append(cond_negation(0, [1, 2, 3, 4], [5]), indices = [p] + x[4:8] + [z[4]])
	c.append(mod_multiplication([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12]), indices = x[0:4] + z[0:4] + x[4:8] + [z[4]])
	c.append(cond_negation(0, [1, 2, 3, 4], [5]), indices = [p] + x[0:4] + [z[4]])

	# now : -x1 , lambda * x1 - y1 , lambda

	c.append(cadd_mod_const(0, [1, 2, 3, 4], (7 - x2) % 7, [5]), indices = [p] + x[0:4] + [z[4]])
	c.append(add_mod_square([0, 1, 2, 3], [4, 5, 6, 7], [8]), indices = z[0:4] + x[0:4] + [z[4]])
	c.append(mod_multiplication([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12]), indices = x[0:4] + z[0:4] + x[4:8] + [z[4]])

	# now : x3 , y3 ( = lambda * (x2 - x3) - y2 ) , lambda

	c.append(add_mod_const([0, 1, 2, 3], y2, [4]), indices = x[4:8] + [z[4]])
	c.append(add_mod_const([0, 1, 2, 3], (7 - x2) % 7, [4]), indices = x[0:4] + [z[4]])

	# now : x3 - x2 , lambda * (x2 - x3) , lambda

	c.append(mod_inverse(), indices = x[0:4])

	c.x(p)
	for i in range(4):
		c.cswap(p, x[i], z[i])
	c.x(p)

	c.append(mod_multiplication([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12]), indices = x + z)

	c.x(p)
	for i in range(4):
		c.cswap(p, x[i], z[i])
	c.x(p)

	c.append(mod_inverse(), indices = x[0:4])

	# # now : x3 - x2 , lambda * (x2 - x3) , 0

	c.append(add_mod_const([0, 1, 2, 3], (7 - y2) % 7, [4]), indices = x[4:8] + [z[4]])
	c.append(add_mod_const([0, 1, 2, 3], x2, [4]), indices = x[0:4] + [z[4]])

	return c
