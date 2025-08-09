import tensorcircuit as tc
import numpy as np
from typing import Sequence, List

# 假设 qft 和 qft_dagger 在 qft.py 文件中定义
from qft import qft, qft_dagger

K = tc.set_backend("tensorflow")

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

# --- 使用示例 ---
if __name__ == '__main__':
    # # =====================================================
    # # Test 1: controlled_add function
    # # 验证: |x>|b> -> |x>|x+b>
    # # =====================================================
    # print("--- Testing controlled_add function ---")
    # n_ca = 4  # 控制寄存器比特数
    # m_ca = 5  # 目标寄存器比特数 (需足够大以容纳和)

    # reg_x_ca = list(range(n_ca))
    # reg_b_ca = list(range(n_ca, n_ca + m_ca))

    # val_x_ca = 10   # 1001
    # val_b_ca = 7  # 01101
    # expected_sum_ca = val_x_ca + val_b_ca # 22 (10110)

    # c_ca = tc.Circuit(n_ca + m_ca)

    # # 初始化状态 |x>|b>
    # bin_x_ca = format(val_x_ca, f'0{n_ca}b')
    # bin_b_ca = format(val_b_ca, f'0{m_ca}b')
    # for i, bit in enumerate(bin_x_ca):
    #     if bit == '1': c_ca.x(reg_x_ca[i])
    # for i, bit in enumerate((bin_b_ca)):
    #     if bit == '1': c_ca.x(reg_b_ca[i])
    # measurement = c_ca.sample()
    # print(f"Input (0-8): {measurement}")
    # print(f"Input: x={bin_x_ca}, b={bin_b_ca}")

    # c_ca.append(controlled_add(n_ca, m_ca))

    # final_state_ca = c_ca.state()
    # output_idx_ca = int(K.argmax(K.abs(final_state_ca) ** 2))
    # output_measurement = c_ca.sample()
    # print(f"Output (0-8): {output_measurement}")
    # print(f"Output index: {output_idx_ca}")

    # result_x_ca = (output_idx_ca >> m_ca)
    # result_b_ca = output_idx_ca - (result_x_ca << m_ca)

    # print(f"Input: x={val_x_ca}, b={val_b_ca}")
    # print(f"Expected: x={val_x_ca}, b={expected_sum_ca}")
    # print(f"Output:   x={result_x_ca}, b={result_b_ca}")
    # assert result_x_ca == val_x_ca
    # assert result_b_ca == expected_sum_ca
    # print("controlled_add test PASSED!\n")


    # # =====================================================
    # # Test 2: addition function
    # # 验证: |x>_n|0>_1|y>_n -> |x>_n|x+y>_{n+1}
    # # =====================================================
    # print("--- Testing addition function ---")
    # n_add = 6  # x 和 y 的比特数

    # reg_x_add = list(range(n_add))
    # reg_y_add = list(range(n_add + 1, n_add + n_add + 1))
    # reg_0_ancilla_add = n_add

    # val_x_add = 22  # 1010
    # val_y_add = 9   # 0111
    # expected_sum_add = val_x_add + val_y_add # 17 (10001)

    # c_add = tc.Circuit(n_add + n_add + 1)

    # # 初始化状态 |x>|y>|0>
    # bin_x_add = format(val_x_add, f'0{n_add}b')
    # bin_y_add = format(val_y_add, f'0{n_add}b')
    # for i, bit in enumerate(bin_x_add):
    #     if bit == '1': c_add.x(reg_x_add[i])
    # for i, bit in enumerate(bin_y_add):
    #     if bit == '1': c_add.x(reg_y_add[i])
    # measurement = c_add.sample()
    # print(f"Input (0-8): {measurement}")

    # c_add.append(addition(n_add))

    # output_measurement = c_add.sample()
    # print(f"Output (0-8): {output_measurement}")

    # final_state_add = c_add.state()
    # output_idx_add = int(K.argmax(K.abs(final_state_add) ** 2))
    
    # result_x_add = (output_idx_add >> (n_add + 1))
    # result_sum_add = output_idx_add - (result_x_add << (n_add + 1))

    # print(f"Input: x={val_x_add}, y={val_y_add}")
    # print(f"Expected: x={val_x_add}, x+y={expected_sum_add}")
    # print(f"Output:   x={result_x_add}, x+y={result_sum_add}")
    # assert result_x_add == val_x_add
    # assert result_sum_add == expected_sum_add
    # print("addition test PASSED!")

    # # =====================================================
    # # Test 2: addition function
    # # 验证: |x>_n|0>_1|y>_n -> |x>_n|x+y>_{n+1}
    # # =====================================================
    # print("--- Testing subtraction function ---")
    # n_minus = 6  # x 和 y 的比特数

    # reg_x_minus = list(range(n_minus))
    # reg_y_minus = list(range(n_minus + 1, n_minus + n_minus + 1))
    # reg_0_ancilla_minus = n_minus

    # val_x_minus = 9  # 1010
    # val_y_minus = 20   # 0111
    # expected_sum_minus = val_y_minus - val_x_minus # 17 (10001)

    # c_minus = tc.Circuit(n_minus + n_minus + 1)

    # # 初始化状态 |x>|y>|0>
    # bin_x_minus = format(val_x_minus, f'0{n_minus}b')
    # bin_y_minus = format(val_y_minus, f'0{n_minus}b')
    # for i, bit in enumerate(bin_x_minus):
    #     if bit == '1': c_minus.x(reg_x_minus[i])
    # for i, bit in enumerate(bin_y_minus):
    #     if bit == '1': c_minus.x(reg_y_minus[i])
    # measurement = c_minus.sample()
    # print(f"Input (0-8): {measurement}")

    # c_minus.append(subtraction(n_minus))

    # output_measurement = c_minus.sample()
    # print(f"Output (0-8): {output_measurement}")

    # final_state_minus = c_minus.state()
    # output_idx_minus = int(K.argmax(K.abs(final_state_minus) ** 2))
    
    # result_x_minus = (output_idx_minus >> (n_minus + 1))
    # result_sum_minus = output_idx_minus - (result_x_minus << (n_minus + 1))

    # print(f"Input: x={val_x_minus}, y={val_y_minus}")
    # print(f"Expected: x={val_x_minus}, y-x={expected_sum_minus}")
    # print(f"Output:   x={result_x_minus}, y-x={result_sum_minus}")
    # assert result_x_minus == val_x_minus
    # assert result_sum_minus == expected_sum_minus
    # print("subtraction test PASSED!")

    # =====================================================
    # Test 4: controlled addition function
    # 验证: |1>|x>_n|0>_1|y>_n -> |1>|x>_n|x+y>_{n+1}
    # =====================================================
    print("--- Testing addition function ---")
    n_add = 5  # x 和 y 的比特数

    reg_x_add = list(range(1, n_add + 1))
    reg_y_add = list(range(n_add + 2, n_add + n_add + 2))
    reg_0_ancilla_add = n_add

    val_x_add = 22  # 1010
    val_y_add = 9   # 0111
    expected_sum_add = val_x_add + val_y_add # 17 (10001)

    c_add = tc.Circuit(n_add + n_add + 2)

    # set control qubit to |1>
    c_add.x(0)

    # 初始化状态 |x>|y>|0>
    bin_x_add = format(val_x_add, f'0{n_add}b')
    bin_y_add = format(val_y_add, f'0{n_add}b')
    for i, bit in enumerate(bin_x_add):
        if bit == '1': c_add.x(reg_x_add[i])
    for i, bit in enumerate(bin_y_add):
        if bit == '1': c_add.x(reg_y_add[i])
    measurement = c_add.sample()
    print(f"Input (0-8): {measurement}")

    c_add.append(controlled_addition(n_add))

    output_measurement = c_add.sample()
    print(f"Output (0-8): {output_measurement}")

    final_state_add = c_add.state()
    output_idx_add = int(K.argmax(K.abs(final_state_add) ** 2))
    
    result_x_add = (output_idx_add >> (n_add + 1)) - ((output_idx_add >> (n_add * 2 + 1)) << n_add)
    result_sum_add = output_idx_add - ((output_idx_add >> (n_add + 1)) << (n_add + 1))

    print(f"Input: x={val_x_add}, y={val_y_add}")
    print(f"Expected: x={val_x_add}, x+y={expected_sum_add}")
    print(f"Output:   x={result_x_add}, x+y={result_sum_add}")
    assert result_x_add == val_x_add
    assert result_sum_add == expected_sum_add
    print("addition test PASSED!")
