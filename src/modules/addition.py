import tensorcircuit as tc
import numpy as np
from typing import Sequence, List

# 假设 qft 和 qft_dagger 在 qft.py 文件中定义
from qft import qft, qft_dagger

K = tc.set_backend("tensorflow")

def controlled_add(c: tc.Circuit, reg_x: Sequence[int], reg_b: Sequence[int]) -> tc.Circuit:
    """
    核心构建块：实现受控加法 |x>|b> -> |x>|x+b>。
    这是一个通用的受控加法器。
    """
    n = len(reg_x)
    m = len(reg_b)
    
    qft(c, reg_b)
    # 遍历控制寄存器 x 的每一位 (从 LSB 到 MSB)
    for i in range(n):
        # 遍历目标寄存器 b 的每一位 (从 LSB 到 MSB)
        # x 的第 i 位控制对 b 的傅里叶变换态施加一系列受控相位门
        for j in range(m):
            # 只有当 b 的比特位 j 不比 x 的比特位 i 更低时，才需要旋转
            # (因为加法中，低位不影响高位)
            # 在我们的索引约定中，这意味着 j >= i
            if j >= i:
                # 正确的相位角
                theta = np.pi / (2 ** (j - i))
                c.cphase(reg_x[n - i - 1], reg_b[j], theta=theta)
    qft_dagger(c, reg_b)

    return c

def addition(c: tc.Circuit, reg_x: Sequence[int], reg_0: int, reg_y: Sequence[int]) -> tc.Circuit:
    """
    构造一个实现 |x>_n|0>_1|y>_n -> |x>_n|x+y>_{n+1} 的电路。
    此函数就地修改电路 c。

    :param c: 要修改的 TensorCircuit 电路。
    :param reg_x: 包含输入值 x 的 n 比特量子寄存器。
    :param reg_0: 1 比特的辅助寄存器，初始为 |0>，将作为和的最高位。
    :param reg_y: 包含输入值 y 的 n 比特量子寄存器。
    """
    n = len(reg_x)
    if len(reg_y) != n:
        raise ValueError("输入寄存器 x 和 y 必须有相同的长度。")

    # 1. 将 y 寄存器和辅助比特组合成一个 n+1 位的目标寄存器 `reg_sum`。
    #    reg_y 构成了低 n 位，reg_0 构成了最高有效位 (MSB)。
    #    这个组合寄存器的初始状态 |0>|y> 的整数值就是 y。
    reg_sum = [reg_0] + list(reg_y)

    # 2. 将 x 加到 y 上。
    #    调用 controlled_add(c, reg_x, reg_sum)
    #    这会执行 |x>|sum_initial> -> |x>|x + sum_initial>
    #    由于 sum_initial 的状态是 |y>，所以结果是 |x>|x+y>。
    controlled_add(c, reg_x, reg_sum)

    return c

# --- 使用示例 ---
if __name__ == '__main__':
    # =====================================================
    # Test 1: controlled_add function
    # 验证: |x>|b> -> |x>|x+b>
    # =====================================================
    print("--- Testing controlled_add function ---")
    n_ca = 4  # 控制寄存器比特数
    m_ca = 5  # 目标寄存器比特数 (需足够大以容纳和)

    reg_x_ca = list(range(n_ca))
    reg_b_ca = list(range(n_ca, n_ca + m_ca))

    val_x_ca = 10   # 1001
    val_b_ca = 7  # 01101
    expected_sum_ca = val_x_ca + val_b_ca # 22 (10110)

    c_ca = tc.Circuit(n_ca + m_ca)

    # 初始化状态 |x>|b>
    bin_x_ca = format(val_x_ca, f'0{n_ca}b')
    bin_b_ca = format(val_b_ca, f'0{m_ca}b')
    for i, bit in enumerate(bin_x_ca):
        if bit == '1': c_ca.x(reg_x_ca[i])
    for i, bit in enumerate((bin_b_ca)):
        if bit == '1': c_ca.x(reg_b_ca[i])
    measurement = c_ca.sample()
    print(f"Input (0-8): {measurement}")
    print(f"Input: x={bin_x_ca}, b={bin_b_ca}")

    controlled_add(c_ca, reg_x_ca, reg_b_ca)

    final_state_ca = c_ca.state()
    output_idx_ca = int(K.argmax(K.abs(final_state_ca) ** 2))
    output_measurement = c_ca.sample()
    print(f"Output (0-8): {output_measurement}")
    print(f"Output index: {output_idx_ca}")

    result_x_ca = (output_idx_ca >> m_ca)
    result_b_ca = output_idx_ca - (result_x_ca << m_ca)

    print(f"Input: x={val_x_ca}, b={val_b_ca}")
    print(f"Expected: x={val_x_ca}, b={expected_sum_ca}")
    print(f"Output:   x={result_x_ca}, b={result_b_ca}")
    assert result_x_ca == val_x_ca
    assert result_b_ca == expected_sum_ca
    print("controlled_add test PASSED!\n")


    # =====================================================
    # Test 2: addition function
    # 验证: |x>_n|0>_1|y>_n -> |x>_n|x+y>_{n+1}
    # =====================================================
    print("--- Testing addition function ---")
    n_add = 6  # x 和 y 的比特数

    reg_x_add = list(range(n_add))
    reg_y_add = list(range(n_add + 1, n_add + n_add + 1))
    reg_0_ancilla_add = n_add

    val_x_add = 22  # 1010
    val_y_add = 9   # 0111
    expected_sum_add = val_x_add + val_y_add # 17 (10001)

    c_add = tc.Circuit(n_add + n_add + 1)

    # 初始化状态 |x>|y>|0>
    bin_x_add = format(val_x_add, f'0{n_add}b')
    bin_y_add = format(val_y_add, f'0{n_add}b')
    for i, bit in enumerate(bin_x_add):
        if bit == '1': c_add.x(reg_x_add[i])
    for i, bit in enumerate(bin_y_add):
        if bit == '1': c_add.x(reg_y_add[i])
    measurement = c_add.sample()
    print(f"Input (0-8): {measurement}")

    addition(c_add, reg_x_add, reg_0_ancilla_add, reg_y_add)

    output_measurement = c_add.sample()
    print(f"Output (0-8): {output_measurement}")

    final_state_add = c_add.state()
    output_idx_add = int(K.argmax(K.abs(final_state_add) ** 2))
    
    result_x_add = (output_idx_add >> (n_add + 1))
    result_sum_add = output_idx_add - (result_x_add << (n_add + 1))

    print(f"Input: x={val_x_add}, y={val_y_add}")
    print(f"Expected: x={val_x_add}, x+y={expected_sum_add}")
    print(f"Output:   x={result_x_add}, x+y={result_sum_add}")
    assert result_x_add == val_x_add
    assert result_sum_add == expected_sum_add
    print("addition test PASSED!")
