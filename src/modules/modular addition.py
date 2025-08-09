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

def modular_addition(x_qubits: List[int], y_qubits: List[int], p: int) -> tc.Circuit:
    n = len(x_qubits)
    if len(y_qubits) != n:
        raise ValueError("x_qubits和y_qubits必须有相同的长度")
    
    total_qubits = max(max(x_qubits), max(y_qubits)) + 2 * n + 2
    c = tc.Circuit(total_qubits)
    
    aux_start = max(max(x_qubits), max(y_qubits)) + 1
    p_qubits = list(range(aux_start, aux_start + n))
    temp_qubits = list(range(aux_start + n, aux_start + 2 * n))
    sign_qubit = aux_start + 2 * n
    
    int_to_qubits(c, p, p_qubits)
    
    qft(c, y_qubits)
    qft_add_register(c, x_qubits, y_qubits)
    qft(c, y_qubits, inverse=True)
    
    for i in range(n):
        c.cnot(y_qubits[i], temp_qubits[i])
    
    qft(c, temp_qubits)
    qft_subtract_register(c, p_qubits, temp_qubits)
    qft(c, temp_qubits, inverse=True)
    
    if n > 0:
        c.cnot(temp_qubits[n-1], sign_qubit)
        c.x(sign_qubit)
    
    for i in range(n):
        c.ccnot(sign_qubit, temp_qubits[i], y_qubits[i])
        c.cnot(temp_qubits[i], y_qubits[i])
        c.ccnot(sign_qubit, temp_qubits[i], y_qubits[i])
    
    if n > 0:
        c.x(sign_qubit)
        c.cnot(temp_qubits[n-1], sign_qubit)
    
    qft(c, temp_qubits)
    qft_add_register(c, p_qubits, temp_qubits)
    qft(c, temp_qubits, inverse=True)
    
    for i in range(n):
        c.cnot(y_qubits[i], temp_qubits[i])
    
    int_to_qubits(c, p, p_qubits)
    
    return c

def test_modular_addition():
    n = 3
    p = 5
    
    x_qubits = [0, 1, 2]
    y_qubits = [3, 4, 5]
    
    c = tc.Circuit(20)
    
    int_to_qubits(c, 3, x_qubits)
    
    int_to_qubits(c, 4, y_qubits)
    
    mod_add_circuit = modular_addition(x_qubits, y_qubits, p)
    c.append(mod_add_circuit, list(range(20)))
    
    state = c.state()
    probs = np.abs(state) ** 2
    
    max_idx = np.argmax(probs)
    
    result = 0
    for i, qubit_idx in enumerate(y_qubits):
        if (max_idx >> qubit_idx) & 1:
            result |= (1 << i)
    
    print(f"量子计算结果: {result}")
    print(f"期望结果: {(3 + 4) % p}")
    print(f"测试通过: {result == (3 + 4) % p}")
    
    return result == (3 + 4) % p

def test_multiple_cases():
    n = 3
    p = 5
    
    test_cases = [
        (1, 2, (1 + 2) % p),
        (2, 3, (2 + 3) % p),
        (4, 4, (4 + 4) % p),
        (0, 3, (0 + 3) % p),
    ]
    
    print("测试多个模加法案例：")
    for x_val, y_val, expected in test_cases:
        x_qubits = [0, 1, 2]
        y_qubits = [3, 4, 5]
        
        c = tc.Circuit(20)
        int_to_qubits(c, x_val, x_qubits)
        int_to_qubits(c, y_val, y_qubits)
        
        mod_add_circuit = modular_addition(x_qubits, y_qubits, p)
        c.append(mod_add_circuit, list(range(20)))
        
        state = c.state()
        probs = np.abs(state) ** 2
        max_idx = np.argmax(probs)
        
        result = 0
        for i, qubit_idx in enumerate(y_qubits):
            if (max_idx >> qubit_idx) & 1:
                result |= (1 << i)
        
        print(f"({x_val} + {y_val}) mod {p} = {result} (期望: {expected}) {'✓' if result == expected else '✗'}")

if __name__ == "__main__":
    print("测试模p加法实现")
    print("=" * 30)
    test_modular_addition()
    print()
    test_multiple_cases()
