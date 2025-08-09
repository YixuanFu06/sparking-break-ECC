import math
import tensorcircuit as tc
    
    def mod_doubling():
        x=[0,1,2,3]
        c=tc.Circuit(len(x))
        for i in reversed(range(len(x)-1)):
            c.SWAP(x[i], x[i+1])
        c.append(add_const(x, 2**n -p))
        c.append(cond_add_const(x[-1],x[0:-1], p))
        c.X(x[0])
        c.CNOT(x[0], x[-1])
        c.X(x[0])
        return c