import numpy as np
import tensorcircuit as tc
from typing import Sequence
from typing import List

K = tc.set_backend("tensorflow")

def get_qubit_count(p: int) -> int:
    """根据模数p计算所需的量子比特数量:n = ceil(log2(p))"""
    return math.ceil(math.log2(p))

########################################################### qft ###########################################################

def qft(n: int) -> tc.Circuit:
    """
    对指定的量子比特序列应用量子傅里叶变换 (QFT)。
    此版本遵循小端序 (Little-Endian) 约定：qubits[0] 是 LSB。
    """
    c = tc.Circuit(n)
    qubits = list(range(n))

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
    
    return c

def qft_dagger(n: int) -> tc.Circuit:
    """
    对指定的量子比特序列应用逆量子傅里叶变换 (IQFT)。
    此版本遵循小端序 (Little-Endian) 约定。
    """
    c = tc.Circuit(n)
    qubits = list(range(n))

    # 核心变换部分（以与 QFT 相反的顺序执行逆操作）
    for i in range(n - 1, -1, -1):
        # 2. 应用受控相位旋转门的逆操作
        for j in range(i + 1, n):
            theta = -np.pi / (2 ** (j - i))
            c.cphase(qubits[j], qubits[i], theta=theta)
        # 3. 应用 Hadamard 门
        c.h(qubits[i])

    return c


########################################################### addition ###########################################################

def addition(n: int) -> tc.Circuit:
    """
    构造一个实现 |x>_n|0>_1|y>_n -> |x>_n|x+y>_{n+1} 的电路。
    """

    def biased_addition(n: int, m: int) -> tc.Circuit:
        """
        核心构建块：实现受控加法 |x>|b> -> |x>|x+b>。
        这是一个通用的受控加法器。
        """
        c = tc.Circuit(n + m)
        reg_x = list(range(n))
        reg_b = list(range(n, n + m))
        
        c.append(qft(m), reg_b)
        # 遍历控制寄存器 x 的每一位 (从 LSB 到 MSB)
        for i in range(n):
            # 遍历目标寄存器 b 的每一位 (从 LSB 到 MSB)
            # x 的第 i 位控制对 b 的傅里叶变换态施加一系列受控相位门
            for j in range(m):
                # 只有当 b 的比特位 j 不比 x 的比特位 i 更低时，才需要旋转
                # 在我们的索引约定中，这意味着 j >= i
                if j >= i:
                    # 正确的相位角
                    theta = np.pi / (2 ** (j - i))
                    c.cphase(reg_x[n - i - 1], reg_b[m - j - 1], theta=theta)
        c.append(qft_dagger(m), reg_b)

        return c

    c = tc.Circuit(2 * n + 1)
    c.append(biased_addition(n, n + 1))

    return c

def controlled_addition(n: int) -> tc.Circuit:
    """
    构造一个实现 |x>_n|0>_1|y>_n -> |x>_n|x+y>_{n+1} 的受控电路。
    """

    def controlled_biased_addition(n: int, m: int) -> tc.Circuit:
        """
        核心构建块：实现受控加法 |x>|b> -> |x>|x+b>。
        这是一个通用的受控加法器。
        """
        c = tc.Circuit(n + m + 1)
        reg_x = list(range(1, n + 1))
        reg_b = list(range(n + 1, n + m + 1))
        
        c.append(qft(m), reg_b)
        # 遍历控制寄存器 x 的每一位 (从 LSB 到 MSB)
        for i in range(n):
            # 遍历目标寄存器 b 的每一位 (从 LSB 到 MSB)
            # x 的第 i 位控制对 b 的傅里叶变换态施加一系列受控相位门
            for j in range(m):
                # 只有当 b 的比特位 j 不比 x 的比特位 i 更低时，才需要旋转
                # 在我们的索引约定中，这意味着 j >= i
                if j >= i:
                    # 正确的相位角
                    theta = np.pi / (2 ** (j - i))
                    c.multicontrol(0, reg_x[n - i - 1], reg_b[m - j - 1], ctrl=[1, 1], unitary=tc.gates.rz(theta=theta))
        c.append(qft_dagger(m), reg_b)

        return c

    c = tc.Circuit(2 * n + 2)
    c.append(controlled_biased_addition(n, n + 1), list(range(2 * n + 2)))

    return c

def subtraction(n: int) -> tc.Circuit:
    """
    构造一个实现 |x>_n|y>_{n+1} -> |x>_n|0>|y-x>_n 的电路。
    """

    def biased_subtraction(n: int, m: int) -> tc.Circuit:
        """
        核心构建块：实现 |x>|b> -> |x>|b-x>。
        """
        # 电路应包含 n (x) + m (b) 个量子比特
        c = tc.Circuit(n + m)
        reg_x = list(range(n))
        reg_b = list(range(n, n + m))
        
        c.append(qft(m), reg_b)
        # 遍历控制寄存器 x 的每一位 (从 LSB 到 MSB)
        for i in range(n):
            # 遍历目标寄存器 b 的每一位 (从 LSB 到 MSB)
            for j in range(m):
                # 只有当 b 的比特位 j 不比 x 的比特位 i 更低时，才需要旋转
                if j >= i:
                    # 将相位角取负，即可实现减法
                    theta = -np.pi / (2 ** (j - i))
                    # 使用正确的寄存器索引
                    c.cphase(reg_x[n - i - 1], reg_b[m - j - 1], theta=theta)
        c.append(qft_dagger(m), reg_b)

        return c

    c = tc.Circuit(2 * n + 1)
    c.append(biased_subtraction(n, n + 1))

    return c


############################################################## modular addition ###########################################################

def modular_addition(n: int, p: int) -> tc.Circuit:
    """
    模加法：计算 (X + Y) mod p
    输入：|X⟩|Y⟩
    输出：|X⟩|(X+Y) mod p⟩
    """

    def int_to_qubits(c: tc.Circuit, value: int, qubits: List[int]):
        """将整数编码到量子比特寄存器中（小端序）"""
        for i, qubit in enumerate(qubits):
            if (value >> i) & 1:
                c.x(qubit)

    total_qubits = 3 * n + 2
    c = tc.Circuit(total_qubits)  # 3n for x, y, and result, +2 for auxiliary bit
    
    # 第一步：将输入的|X⟩和|Y⟩相加
    # 使用addition函数：|X⟩|0⟩|Y⟩ → |X⟩|X+Y⟩
    c.append(addition(n), list(range(n)) + list([3 * n]) + list(range(n, 2 * n)))
    
    # 现在y_qubits包含了X+Y的结果
    
    # 第二步：从Y寄存器中减去p
    # 首先需要创建存储p的寄存器
    # 假设我们有足够的辅助量子比特
    p_qubits = list(range(2 * n, 3 * n))  # p寄存器
    z_qubit = 3 * n + 1  # 辅助比特Z
    
    # 将p编码到p_qubits中
    int_to_qubits(c, p, p_qubits)
    
    # 执行减法：Y = Y - P
    c.append(subtraction(n), list(range(2 * n, 3 * n)) + list([3 * n]) + list(range(n, 2 * n)))
    
    # 第三步：检查最高位判断是否为负数
    # 由于n = ceil(log2(p))，正常情况下最高位应该是0
    # 如果减去p后变为负数，最高位会变成1
    highest_bit = y_qubits[n - 1] # 最高位量子比特
    
    # 将最高位的状态复制到辅助比特Z
    c.cnot(highest_bit, z_qubit)
    
    # 第四步：条件性地加回p
    # 如果结果小于0（Z=1），我们需要把Y加上p
    # 如果结果不小于0（Z=0），什么都不做
    # 以辅助比特Z为控制比特控制对Y加p的门
    c.append(controlled_addition(n), list([3 * n + 1]) + list(range(2 * n, 3 * n)) + list([3 * n]) + list(range(n, 2 * n)))
    
    # 第五步：复原辅助比特Z
    # 对Y寄存器内的结果减去X
    # Y寄存器内要么是X+Y（-p再+回来了），要么是X+Y-p（-p后没有再操作）
    c.append(subtraction(n), list(range(n)) + list([3 * n]) + list(range(n, 2 * n)))  # Y = Y - X
    
    # 减去X后：
    # - 如果原来是X+Y，现在剩余Y，最高位是0（意味着辅助比特Z应该是1）
    # - 如果原来是X+Y-p，现在剩余Y-p，最高位是1（意味着辅助比特Z应该是0）
    
    # 需要在最高位=0时翻转Z，最高位=1时不动Z
    # 先翻转最高位
    c.x(highest_bit)
    
    # 以翻转后的最高位为控制比特来翻转Z
    c.cnot(highest_bit, z_qubit)
    
    # 再翻转最高位回来
    c.x(highest_bit)
    
    # 第六步：恢复Y寄存器
    # 将X加回到Y寄存器，恢复到最终结果
    c.append(addition(n), list(range(n)) + list([3 * n]) + list(range(n, 2 * n)))  # Y = Y + X

    int_to_qubits(c, p, p_qubits)

    return c


############################################################ multiply and square ###########################################################

def modular_multiplication(n: int, p: int) -> tc.Circuit:
    """
    模乘法 |x⟩|y⟩|0⟩ → |x⟩|y⟩|x*y mod p⟩
    参数:
        c: 量子电路
        n 是每个寄存器的量子比特数, 电路c包括3n个量子比特
        其中 x_reg, y_reg: 输入寄存器（量子比特位置索引）
        x_reg = [0, 1, ..., n-1]
        y_reg = [n, n+1, ..., 2n-1]
        结果寄存器 res_reg = [2n, 2n+1, ..., 3n-1]
        例如 n=3 时，x_reg = [0,1,2], y_reg = [3,4,5], res_reg = [6,7,8]
        p: 模数
    """

    c = tc.Circuit(3 * n)

    # 1. 通过连加计算 x*y=Σ (y*2^i) * x_i
    for i in range(n):
        # 对 x 的每一位 x_i，若为1，则|x⟩|y⟩|psi>->|x⟩|y⟩|psi+y*2^i mod p>
        # 当x的第n-i位(也即整体电路的第n-i位, 从最高位向最低位不需要辅助比特)为1时，受控模加法
        # c.append(controlled_modular_addition(y_reg, res_reg, p, control=n-i))
        c.append(controlled_modular_addition(n, p), list([n - i]) + list(range(n, 3 * n)))
        c.append(doubling(n, p), list(range(2 * n, 3 * n)))
    
    # 2. 结果已在 res_reg 中（x*y mod p）
    return c

def sqr(c: tc.Circuit, x_reg, res_reg, p):
    """
    模平方 |x⟩|0⟩ → |x⟩|x² mod p⟩
    """
    n = len(x_reg) 
    for i in range(n):
        # 对 x 的每一位 x_i，若为1，则|x⟩|y⟩|psi>->|x⟩|y⟩|psi+y*2^i mod p>
        # 当x的第n-i位(也即整体电路的第n-i位, 从最高位向最低位不需要辅助比特)为1时，受控模加法
        # c.append(controlled_modular_add(x_reg, res_reg, p, control=n-i))
        c.append(controlled_modular_addition(n, p), list([n - i]) + list(range(2 * n)))
        c.append(doubling(n, p), list(range(n, 2 * n)))
        
    return c