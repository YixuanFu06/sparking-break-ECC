import numpy as np
import tensorcircuit as tc
from typing import List

tc.set_backend("tensorflow")

def int_to_qubits(c: tc.Circuit, value: int, qubits: List[int]):
    for i, qubit in enumerate(qubits):
        if (value >> i) & 1:
            c.x(qubit)

def qft(c: tc.Circuit, qubits: List[int], inverse: bool = False):
    n = len(qubits)
    
    if not inverse:
        for i in range(n):
            c.h(qubits[i])
            for j in range(i + 1, n):
                angle = np.pi / (2 ** (j - i))
                c.crz(qubits[j], qubits[i], theta=2 * angle)
        
        for i in range(n // 2):
            c.swap(qubits[i], qubits[n - 1 - i])
    else:
        for i in range(n // 2):
            c.swap(qubits[i], qubits[n - 1 - i])
        
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, i, -1):
                angle = np.pi / (2 ** (j - i))
                c.crz(qubits[j], qubits[i], theta=-2 * angle)
            c.h(qubits[i])

def qft_add_register(c: tc.Circuit, a_qubits: List[int], b_qubits: List[int]):
    n = len(a_qubits)
    
    for i in range(n):
        for j in range(i, n):
            angle = np.pi / (2 ** (j - i))
            c.crz(a_qubits[j], b_qubits[i], theta=2 * angle)

def qft_subtract_register(c: tc.Circuit, a_qubits: List[int], b_qubits: List[int]):
    n = len(a_qubits)
    
    for i in range(n):
        for j in range(i, n):
            angle = np.pi / (2 ** (j - i))
            c.crz(a_qubits[j], b_qubits[i], theta=-2 * angle)

def modular_doubling(x_qubits: List[int], p: int) -> tc.Circuit:
    n = len(x_qubits)
    
    total_qubits = max(x_qubits) + 3 * n + 2
    c = tc.Circuit(total_qubits)
    
    aux_start = max(x_qubits) + 1
    x_copy_qubits = list(range(aux_start, aux_start + n))
    p_qubits = list(range(aux_start + n, aux_start + 2 * n))
    temp_qubits = list(range(aux_start + 2 * n, aux_start + 3 * n))
    sign_qubit = aux_start + 3 * n
    
    for i in range(n):
        c.cnot(x_qubits[i], x_copy_qubits[i])
    
    int_to_qubits(c, p, p_qubits)
    
    qft(c, x_qubits)
    qft_add_register(c, x_copy_qubits, x_qubits)
    qft(c, x_qubits, inverse=True)
    
    for i in range(n):
        c.cnot(x_qubits[i], temp_qubits[i])
    
    qft(c, temp_qubits)
    qft_subtract_register(c, p_qubits, temp_qubits)
    qft(c, temp_qubits, inverse=True)
    
    if n > 0:
        c.cnot(temp_qubits[n-1], sign_qubit)
        c.x(sign_qubit)
    
    for i in range(n):
        c.ccnot(sign_qubit, temp_qubits[i], x_qubits[i])
        c.cnot(temp_qubits[i], x_qubits[i])
        c.ccnot(sign_qubit, temp_qubits[i], x_qubits[i])
    
    if n > 0:
        c.x(sign_qubit)
        c.cnot(temp_qubits[n-1], sign_qubit)
    
    qft(c, temp_qubits)
    qft_add_register(c, p_qubits, temp_qubits)
    qft(c, temp_qubits, inverse=True)
    
    for i in range(n):
        c.cnot(x_qubits[i], temp_qubits[i])
    
    int_to_qubits(c, p, p_qubits)
    
    for i in range(n):
        c.cnot(x_qubits[i], x_copy_qubits[i])
    
    return c

def test_modular_doubling():
    n = 3
    p = 5
    
    test_cases = [
        1,  # (1 * 2) % 5 = 2
        2,  # (2 * 2) % 5 = 4  
        3,  # (3 * 2) % 5 = 1
        4,  # (4 * 2) % 5 = 3
        0,  # (0 * 2) % 5 = 0
    ]
    
    print("测试模倍增：")
    print("=" * 30)
    
    for x_val in test_cases:
        x_qubits = [0, 1, 2]
        
        c = tc.Circuit(20)
        int_to_qubits(c, x_val, x_qubits)
        
        doubling_circuit = modular_doubling(x_qubits, p)
        c.append(doubling_circuit, list(range(20)))
        
        state = c.state()
        probs = np.abs(state) ** 2
        max_idx = np.argmax(probs)
        
        result = 0
        for i, qubit_idx in enumerate(x_qubits):
            if (max_idx >> qubit_idx) & 1:
                result |= (1 << i)
        
        expected = (2 * x_val) % p
        print(f"2 * {x_val} mod {p} = {result} (期望: {expected}) {'✓' if result == expected else '✗'}")

if __name__ == "__main__":
    test_modular_doubling()