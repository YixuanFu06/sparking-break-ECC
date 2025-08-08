import tensorcircuit as tc
import numpy as np
from typing import Sequence

def qft(c: tc.Circuit, qubits: Sequence[int]):
    """
    对指定的量子比特序列应用量子傅里叶变换 (QFT)。

    :param c: TensorCircuit 电路对象。
    :param qubits: 要应用 QFT 的量子比特索引序列。
    """
    n = len(qubits)
    # 核心变换部分
    for i in range(n - 1, -1, -1):
        # 1. 对当前量子比特应用 Hadamard 门
        c.h(qubits[i])
        # 2. 应用受控相位旋转门
        #    控制位是比当前位更重要的比特（索引更小）
        for j in range(i - 1, -1, -1):
            # 旋转角度 theta = pi / 2^(i-j)
            theta = np.pi / (2 ** (i - j))
            c.cphase(qubits[j], qubits[i], theta=theta)
    
    # 3. 在末尾反转量子比特的顺序
    #    这是 QFT 标准电路的一部分，用于将输出调整到正确的顺序
    for i in range(n // 2):
        c.swap(qubits[i], qubits[n - 1 - i])

def qft_dagger(c: tc.Circuit, qubits: Sequence[int]):
    """
    对指定的量子比特序列应用逆量子傅里叶变换 (IQFT)。
    它是 QFT 操作的精确逆过程。

    :param c: TensorCircuit 电路对象。
    :param qubits: 要应用 IQFT 的量子比特索引序列。
    """
    n = len(qubits)
    # 1. 首先反转量子比特的顺序，这是 QFT 最后一步的逆操作
    for i in range(n // 2):
        c.swap(qubits[i], qubits[n - 1 - i])

    # 核心变换部分（以与 QFT 相反的顺序执行逆操作）
    for i in range(n):
        # 2. 应用受控相位旋转门的逆操作（theta 取负）
        for j in range(i):
            theta = -np.pi / (2 ** (i - j))
            c.cphase(qubits[j], qubits[i], theta=theta)
        # 3. 应用 Hadamard 门（H 门是自身的逆）
        c.h(qubits[i])
