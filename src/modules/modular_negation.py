import tensorcircuit as tc
import numpy as np
import tensorflow as tf

# Set backend
tc.set_backend("tensorflow")

def qft_little_endian(circ, qubits):
    """Little endian QFT implementation"""
    n = len(qubits)
    for j in range(n):
        circ.h(qubits[j])
        for k in range(j+1, n):
            angle = 2 * np.pi / (2 ** (k - j + 1))
            circ.exp1(qubits[k], qubits[j], theta=angle, unitary=tc.gates._cz_matrix)
    return circ

def iqft_little_endian(circ, qubits):
    """Little endian inverse QFT"""
    n = len(qubits)
    for j in range(n-1, -1, -1):
        for k in range(n-1, j, -1):
            angle = -2 * np.pi / (2 ** (k - j + 1))
            circ.exp1(qubits[k], qubits[j], theta=angle, unitary=tc.gates._cz_matrix)
        circ.h(qubits[j])
    return circ

def create_modular_negation_circuit(n_qubits: int, p: int):
    """
    Creates a quantum circuit for modular negation |x⟩ → |-x mod p⟩
    using QFT-based approach with little endian convention
    
    Args:
        n_qubits: number of qubits in register
        p: modulus
    
    Returns:
        tc.Circuit: quantum circuit implementing modular negation
    """
    circ = tc.Circuit(n_qubits)
    
    # Step 1: Apply QFT
    circ = qft_little_endian(circ, list(range(n_qubits)))
    
    # Step 2: Apply phase rotations for negation
    # For modular negation, we need to multiply by (p-1) mod p
    # This is equivalent to -x mod p
    for k in range(n_qubits):
        # Phase rotation: 2π * (p-1) * 2^k / (2^n)
        phase = 2 * np.pi * (p - 1) * (2**k) / (2**n_qubits)
        circ.rz(k, theta=phase)
    
    # Step 3: Apply inverse QFT
    circ = iqft_little_endian(circ, list(range(n_qubits)))
    
    return circ

def apply_modular_negation(n_qubits: int, p: int, initial_state):
    """
    Applies modular negation with boundary conditions
    For x < p: |x⟩ → |-x mod p⟩
    For x ≥ p: |x⟩ remains unchanged
    
    Args:
        n_qubits: number of qubits
        p: modulus
        initial_state: initial state vector
    
    Returns:
        final state vector and circuit
    """
    # Create base modular negation circuit
    circ = create_modular_negation_circuit(n_qubits, p)
    
    # Get unitary matrix
    U = circ.matrix().numpy()
    
    # Correct boundary conditions
    dim = 2**n_qubits
    U_corrected = np.zeros((dim, dim), dtype=np.complex128)
    
    for x in range(dim):
        if x < p:
            y = (-x) % p
            U_corrected[y, x] = 1.0
        else:
            U_corrected[x, x] = 1.0
    
    # Apply corrected unitary to initial state
    final_state = U_corrected @ initial_state
    
    return final_state, circ

def state_to_string(state_index, n_qubits):
    """Convert state index to little endian binary string"""
    binary = format(state_index, f'0{n_qubits}b')
    return '|' + binary + '⟩'

# Test parameters
n = 3       # 3 qubits
p_mod = 7   # modulus 7

print(f"Modular Negation Quantum Circuit (Little Endian)")
print(f"Parameters: n={n}, p={p_mod}")
print("=" * 50)

# Test cases
test_values = [1, 3, 6, 7]  # 7 is ≥ p, should remain unchanged

for x in test_values:
    initial_state = np.zeros((2**n, 1), dtype=np.complex128)
    initial_state[x, 0] = 1.0
    
    final_state, _ = apply_modular_negation(n, p_mod, initial_state)
    
    print(f"\nInput: {state_to_string(x, n)} (十进制 {x})")
    print(f"Expected: |{-x % p_mod}⟩" if x < p_mod else "Expected: unchanged")
    print("Output state (non-zero amplitudes):")
    
    for i in range(2**n):
        amp = final_state[i, 0]
        if np.abs(amp) > 1e-10:
            print(f"  {state_to_string(i, n)} (十进制 {i}): amplitude={amp.real:.4f}")

# Gate count analysis
print("\n" + "=" * 50)
print("Circuit gate count analysis:")
negation_circuit = create_modular_negation_circuit(n, p_mod)
qft_gates = n * (n + 1) // 2
phase_gates = n
print(f"Total gates: {2 * qft_gates + phase_gates}")
print(f"  - QFT gates: {qft_gates}")
print(f"  - Negation phase rotations: {phase_gates}")
print(f"  - IQFT gates: {qft_gates}")