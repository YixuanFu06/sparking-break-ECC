import tensorcircuit as tc
import numpy as np
from typing import Sequence

def qft(c: tc.Circuit, qubits: Sequence[int]) -> tc.Circuit:
    """
    对指定的量子比特序列应用量子傅里叶变换 (QFT)。
    此版本遵循小端序 (Little-Endian) 约定：qubits[0] 是 LSB。
    """
    n = len(qubits)
    # 核心变换部分
    for i in range(n):
        # 1. 对当前量子比特应用 Hadamard 门
        c.h(qubits[i])
        # 2. 应用受控相位旋转门
        #    控制位是比当前位更重要的比特（索引更大）
        for j in range(i + 1, n):
            # 旋转角度 theta = pi / 2^(j-i)
            theta = np.pi / (2 ** (j - i))
            c.cphase(qubits[j], qubits[i], theta=theta)
    
    # 3. 在末尾反转量子比特的顺序
    for i in range(n // 2):
        c.swap(qubits[i], qubits[n - 1 - i])

    return c

def qft_dagger(c: tc.Circuit, qubits: Sequence[int]) -> tc.Circuit:
    """
    对指定的量子比特序列应用逆量子傅里叶变换 (IQFT)。
    此版本遵循小端序 (Little-Endian) 约定。
    """
    n = len(qubits)
    # 1. 首先反转量子比特的顺序
    for i in range(n // 2):
        c.swap(qubits[i], qubits[n - 1 - i])

    # 核心变换部分（以与 QFT 相反的顺序执行逆操作）
    for i in range(n - 1, -1, -1):
        # 2. 应用受控相位旋转门的逆操作
        for j in range(i + 1, n):
            theta = -np.pi / (2 ** (j - i))
            c.cphase(qubits[j], qubits[i], theta=theta)
        # 3. 应用 Hadamard 门
        c.h(qubits[i])

    return c
