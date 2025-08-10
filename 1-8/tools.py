import tensorcircuit as tc
import numpy as np

def block(func):
	def wrapper(*stuff, **parameters):
		circuit = func(*stuff, **parameters)
		n = circuit._nqubits
		result = tc.Circuit(n)
		result.any(*range(n), unitary=circuit.matrix())
		return result
	return wrapper

def output(circuit, bit_length = -1):
	state_vector = circuit.state()
	for index in range(len(state_vector)):
		if abs(state_vector[index]) >= 0.001:
			if bit_length == -1:
				print(f"{bin(index)[2:]} : {state_vector[index]}")
			else:
				print(f"{(bin(index)[2:]).zfill(bit_length)} : {state_vector[index]}")
@block
def ccphase(circuit, control1, control2, target, theta = np.pi):
    circuit.cphase(control1, target, theta = theta / 2)
    circuit.cphase(control2, target, theta = theta / 2)
    circuit.cnot(control1, control2)
    circuit.cphase(control2, target, theta = -theta / 2)
    circuit.cnot(control1, control2)
    return circuit

@block
def controlled_H(control_bit, target_bit):
	c = tc.Circuit(max(control_bit, target_bit) + 1)
	c.any(control_bit, target_bit, unitary = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)], [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)]]))
	return c

def int_to_qubits(x, C):
	circuit = tc.Circuit(max(x) + 1)
	while len(x):
		if C & 1: circuit.x(x[0])
		C >>= 1
		x = x[1:]
	return circuit
