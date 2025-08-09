import tensorcircuit as tc
import math
import numpy
from typing import List

n = math.ceil(math.log2(p))


def mod_constant_addition(x_qubits: List[int], constant: int):
    c = tc.Circuit(max(x_qubits) + 1)
    n=len(x_qubits)
    c.append(QFT(x_qubits))
    for i in range(n // 2):
        c.swap(x_qubits[i], x_qubits[-i-1])
    for i in [i for i in range(n) if (constant >> i) & 1]:
        for j in range(n):
            if i+j <= n:
                c.phase(x_qubits[j], theta=2 * numpy.pi * 2**(i+j) / (2 ** n))
    for i in range(n // 2):
        c.swap(x_qubits[i], x_qubits[-i-1])
    c.append(QFT(x_qubits, inverse=True))
    return c

def cond_mod_constant_addition(control_qubit : int ,x_qubits: List[int], constant: int):
    c = tc.Circuit(max(control_qubit , max(x_qubits)) + 1)
    n=len(x_qubits)
    c.append(QFT(x_qubits))
    for i in range(n // 2):
        c.swap(x_qubits[i], x_qubits[-i-1])
    for i in [i for i in range(n) if (constant >> i) & 1]:
        for j in range(n):
            if i+j <= n:
                c.cphase(control_qubit , x_qubits[j], theta=2 * numpy.pi * 2**(i+j) / (2 ** n))
    for i in range(n // 2):
        c.swap(x_qubits[i], x_qubits[-i-1])
    c.append(QFT(x_qubits, inverse=True))
    return c