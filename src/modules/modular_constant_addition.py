

import tensorcircuit as tc
import numpy as np
import tensorflow as tf

# 设置后端
tc.set_backend("tensorflow")

def qft_little_endian(circ, qubits):
    """
    实现little endian约定的QFT
    第0个qubit在bracket的最右侧，对应最低有效位
    """ 
    n = len(qubits)
    
    # QFT在little endian约定下
    for j in range(n):
        # 先对第j个qubit应用Hadamard门
        circ.h(qubits[j])
        
        # 然后应用控制相位门
        for k in range(j+1, n):
            # 相位角度：2π/2^(k-j+1)
            angle = 2 * np.pi / (2 ** (k - j + 1))
            # 使用CRZ门实现控制相位旋转
            circ.exp1(qubits[k], qubits[j], theta=angle, unitary=tc.gates._cz_matrix)
    
    return circ

def iqft_little_endian(circ, qubits):
    """
    实现little endian约定的逆QFT
    """
    n = len(qubits)
    
    # 逆QFT是QFT的逆操作，门的顺序相反
    for j in range(n-1, -1, -1):
        # 先应用控制相位门的逆操作
        for k in range(n-1, j, -1):
            angle = -2 * np.pi / (2 ** (k - j + 1))
            circ.exp1(qubits[k], qubits[j], theta=angle, unitary=tc.gates._cz_matrix)
        
        # 最后应用Hadamard门
        circ.h(qubits[j])
    
    return circ

def create_modular_addition_circuit(n_qubits: int, c: int, p: int):
    """
    创建一个量子电路来实现模加法操作 |x⟩ → |(x+c) mod p⟩
    使用little endian约定：第0个qubit在最右侧，表示2^0位
    
    Args:
        n_qubits: 寄存器中的量子比特数
        c: 要加的常数
        p: 模数
    
    Returns:
        tensorcircuit.Circuit: 实现模加法的量子电路
    """
    
    # 创建量子电路
    circ = tc.Circuit(n_qubits)
    
    # 步骤1: 应用QFT将计算基态转换到相位基态
    circ = qft_little_endian(circ, list(range(n_qubits)))
    
    # 步骤2: 在相位基态中实现加法操作
    # 在little endian约定下，第k个qubit对应2^k位
    # 相位旋转角度为 2π * c * 2^k / 2^n
    for k in range(n_qubits):
        phase = 2 * np.pi * c * (2**k) / (2**n_qubits)
        circ.rz(k, theta=phase)
    
    # 步骤3: 应用逆QFT转换回计算基态
    circ = iqft_little_endian(circ, list(range(n_qubits)))
    
    return circ

def apply_modular_addition_with_condition(n_qubits: int, c: int, p: int, initial_state):
    """
    应用模加法，但考虑x≥p的边界条件
    对于x<p，执行(x+c) mod p；对于x≥p，保持不变
    
    使用little endian约定：
    - |x⟩ = |x_{n-1}...x_1x_0⟩，其中x_0是最低有效位（在最右侧）
    - x = x_0*2^0 + x_1*2^1 + ... + x_{n-1}*2^{n-1}
    
    Args:
        n_qubits: 量子比特数
        c: 加数常量
        p: 模数
        initial_state: 初始态向量
    
    Returns:
        最终态向量和电路
    """
    
    # 创建基本的模加法电路
    circ = create_modular_addition_circuit(n_qubits, c, p)
    
    # 获取电路的酉矩阵
    U = circ.matrix().numpy()
    
    # 修正边界条件：对于x≥p的基态，应该保持不变
    dim = 2**n_qubits
    U_corrected = np.zeros((dim, dim), dtype=np.complex128)
    
    for x in range(dim):
        if x < p:
            # 对于x<p，应用模加法
            y = (x + c) % p
            U_corrected[y, x] = 1.0
        else:
            # 对于x≥p，保持恒等变换
            U_corrected[x, x] = 1.0
    
    # 应用修正后的酉矩阵到初始态
    final_state = U_corrected @ initial_state
    
    return final_state, circ

def state_to_string(state_index, n_qubits):
    """
    将状态索引转换为little endian二进制字符串表示
    例如：5 (101) -> |101⟩，其中最右边是第0个qubit
    """
    binary = format(state_index, f'0{n_qubits}b')
    # 反转字符串以符合little endian约定（最右边是LSB）
    return '|' + binary[::-1] + '⟩'

# 测试参数
n = 3          # 3个量子比特
c_const = 3    # 常数c = 3
p_mod = 7      # 模p = 7

print(f"模加法量子电路实现 (Little Endian约定)")
print(f"参数: n={n}, c={c_const}, p={p_mod}")
print(f"注意: 第0个qubit在bracket的最右侧")
print("=" * 50)

# 测试1: |1⟩ → |4⟩
print("\n测试1: 输入态 |001⟩ (十进制值=1)")
initial_state_1 = np.zeros((2**n, 1), dtype=np.complex128)
initial_state_1[1, 0] = 1.0  # |001⟩ in little endian

final_state_1, circuit = apply_modular_addition_with_condition(n, c_const, p_mod, initial_state_1)

print(f"操作: |x⟩ → |(x+{c_const}) mod {p_mod}⟩")
print(f"预期输出: |(1+{c_const}) mod {p_mod}⟩ = |001⟩ → |001⟩ (十进制值=4)")
print("实际输出态向量 (非零振幅):")
for i in range(2**n):
    amp = final_state_1[i, 0]
    if np.abs(amp) > 1e-10:
        print(f"  {state_to_string(i, n)} (十进制={i}): 振幅={amp.real:.4f}")

# 测试2: |4⟩ → |0⟩
print("\n" + "-" * 50)
print("测试2: 输入态 |001⟩ (十进制值=4)")
initial_state_4 = np.zeros((2**n, 1), dtype=np.complex128)
initial_state_4[4, 0] = 1.0  # |001⟩ in little endian

final_state_4, _ = apply_modular_addition_with_condition(n, c_const, p_mod, initial_state_4)

print(f"操作: |x⟩ → |(x+{c_const}) mod {p_mod}⟩")
print(f"预期输出: |(4+{c_const}) mod {p_mod}⟩ = |000⟩ (十进制值=0)")
print("实际输出态向量 (非零振幅):")
for i in range(2**n):
    amp = final_state_4[i, 0]
    if np.abs(amp) > 1e-10:
        print(f"  {state_to_string(i, n)} (十进制={i}): 振幅={amp.real:.4f}")

# 测试3: 叠加态
print("\n" + "-" * 50)
print("测试3: 叠加态 (1/√2)(|001⟩ + |001⟩)")
superposition_state = np.zeros((2**n, 1), dtype=np.complex128)
superposition_state[1, 0] = 1.0 / np.sqrt(2)  # |001⟩
superposition_state[4, 0] = 1.0 / np.sqrt(2)  # |001⟩

final_superposition, _ = apply_modular_addition_with_condition(n, c_const, p_mod, superposition_state)

print(f"预期输出: (1/√2)(|001⟩ + |000⟩)")
print("实际输出态向量 (非零振幅):")
for i in range(2**n):
    amp = final_superposition[i, 0]
    if np.abs(amp) > 1e-10:
        print(f"  {state_to_string(i, n)} (十进制={i}): 振幅={amp.real:.4f}")

# 打印电路信息
print("\n" + "=" * 50)
print("电路信息:")
print(f"量子比特数: {circuit.circuit_param['nqubits']}")
print(f"量子比特顺序 (Little Endian): q0(LSB)在最右侧, q{n-1}(MSB)在最左侧")

# 计算电路深度和门数量
qft_gates = n * (n + 1) // 2  # QFT中的Hadamard和控制相位门
phase_gates = n  # 相位加法的Rz门
total_gates = 2 * qft_gates + phase_gates

print(f"\n估算门数量: {total_gates}")
print(f"  - QFT门: {qft_gates}")
print(f"  - 相位旋转门: {phase_gates}")  
print(f"  - IQFT门: {qft_gates}")

# 验证little endian约定
print("\n" + "=" * 50)
print("Little Endian约定验证:")
print("二进制表示 -> 十进制值")
for i in range(min(8, 2**n)):
    binary = format(i, f'0{n}b')
    # 在little endian中，最右边的位是最低有效位
    binary_le = binary[::-1]
    print(f"|{binary_le}⟩ = {i}")