import numpy as np
import tensorcircuit as tc
from typing import List
import math

tc.set_backend("tensorflow")

def addition(c: tc.Circuit, x_qubits: List[int], y_qubits: List[int]):
    """
    假设的加法函数：|X⟩|Y⟩ → |X⟩|X+Y⟩
    这里我们假设这个函数已经实现，将X和Y相加，结果存储在Y寄存器中
    """
    pass  # 这个函数假设已经实现

def subtraction(c: tc.Circuit, p_qubits: List[int], y_qubits: List[int]):
    """
    假设的减法函数：|P⟩|Y⟩ → |P⟩|Y-P⟩
    通过QFT加法函数的相位翻转实现
    """
    pass  # 这个函数假设已经实现

def controlled_addition(c: tc.Circuit, control_qubit: int, p_qubits: List[int], y_qubits: List[int]):
    """
    受控加法函数：当控制位为1时，执行Y = Y + P
    """
    pass  # 这个函数假设已经实现

def modular_addition(c: tc.Circuit, x_qubits: List[int], y_qubits: List[int], p: int):
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

    n = get_qubit_count(p)
    
    # 第一步：将输入的|X⟩和|Y⟩相加
    # 使用addition函数：|X⟩|Y⟩ → |X⟩|X+Y⟩
    addition(c, x_qubits, y_qubits)
    
    # 现在y_qubits包含了X+Y的结果
    
    # 第二步：从Y寄存器中减去p
    # 首先需要创建存储p的寄存器
    # 假设我们有足够的辅助量子比特
    total_qubits = max(max(x_qubits), max(y_qubits)) + n + 2  # 需要额外的p寄存器和辅助比特
    p_qubits = list(range(max(max(x_qubits), max(y_qubits)) + 1, 
                         max(max(x_qubits), max(y_qubits)) + 1 + n))
    z_qubit = max(max(x_qubits), max(y_qubits)) + 1 + n  # 辅助比特Z
    
    # 将p编码到p_qubits中
    int_to_qubits(c, p, p_qubits)
    
    # 执行减法：Y = Y - P
    subtraction(c, p_qubits, y_qubits)
    
    # 第三步：检查最高位判断是否为负数
    # 由于n = ceil(log2(p))，正常情况下最高位应该是0
    # 如果减去p后变为负数，最高位会变成1
    highest_bit = y_qubits[n-1]  # 最高位量子比特
    
    # 将最高位的状态复制到辅助比特Z
    c.cnot(highest_bit, z_qubit)
    
    # 第四步：条件性地加回p
    # 如果结果小于0（Z=1），我们需要把Y加上p
    # 如果结果不小于0（Z=0），什么都不做
    # 以辅助比特Z为控制比特控制对Y加p的门
    controlled_addition(c, z_qubit, p_qubits, y_qubits)
    
    # 第五步：复原辅助比特Z
    # 对Y寄存器内的结果减去X
    # Y寄存器内要么是X+Y（-p再+回来了），要么是X+Y-p（-p后没有再操作）
    subtraction(c, x_qubits, y_qubits)  # Y = Y - X
    
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
    addition(c, x_qubits, y_qubits)  # Y = Y + X

    int_to_qubits(c, p, p_qubits)
