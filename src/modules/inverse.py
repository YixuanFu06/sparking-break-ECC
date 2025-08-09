import numpy as np
import tensorcircuit as tc

tc.set_backend("tensorflow")

def modinv_quantum_circuit():
    # Create a 3-qubit circuit
    n = 3
    c = tc.Circuit(n)
    
    # The modular inverse operation is simply swapping qubit 1 and qubit 2
    c.swap(0, 1)
    
    return c

def test_modinv_circuit():
    print("Input | Output (x⁻¹ mod 7)")
    print("------|-------------------")
    
    for i in range(8):
        # Create circuit for this basis state
        c = tc.Circuit(3)
        bits = format(i, '03b')
        for q in range(3):
            if bits[q] == '1':
                c.x(q)
        
        # Apply the modular inverse circuit
        inv_circuit = modinv_quantum_circuit()
        c.append(inv_circuit)
        
        # Get the resulting state
        state = c.state()
        
        # Find the basis state with maximum amplitude
        output = np.argmax(np.abs(state))
        output_state_str = format(output, '03b')
        
        # Map the input to its inverse
        inv_map = {
            0: "0 (no inverse)",
            1: "1",
            2: "4",
            3: "5",
            4: "2",
            5: "3",
            6: "6",
            7: "7 (no inverse)"
        }
        
        print(f"{bits} ({i}) | {output_state_str} ({inv_map[i]})")

if __name__ == "__main__":
    test_modinv_circuit()
