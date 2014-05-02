import time
import brian_no_units
from brian import *

# Model parameters
tau = -20 * msecond
Vt = -50 * mvolt
Vr = -60 * mvolt
el = -49 * mvolt

# Mode G2D parameters
tau =   1.0
I   =   0.0
a   = - 0.5
b   = - 10.0
c   =   0.0
d   =   10.0
e   =   3.0
f   =   1.0
alpha = 1.0
beta  = 1.0

delays = True
# model equations
eqs = '''
dV/dt = d * (tau * (alpha * W - f * V**3 + e * V**2 + I)): 1
dW/dt = d * ((a + b * V + c * V**2 - beta * W) / tau) : 1
'''

# Model equations - Dummy equations for a two state variable model 
# eqs = Equations('''
#       dV/dt  = (- V  - el - VE + 2 * V + I)/tau        : volt
#       dVE/dt = -( VE - el - V * 2 + 3 * V)/tau         : volt
#       ''')

# Connectivity weigth
psp = 0.5 * mvolt

# Nodes
G = NeuronGroup(N=74, model=eqs, threshold=Vt, reset=Vr)

# Connectivity
if delays:
	myconnection=Connection(G, G, 'V', delay=True, max_delay=10*ms, structure='dense')
	myconnection.connect_full(G, G, weight=psp, delay=(0.1*ms,10*ms))
else:
	connectivity = Connection(G, G, 'V', sparseness=0.1, weight=0.0001)

# Monitor
#M = SpikeMonitor(G, record=True)
M = MultiStateMonitor(G, record=True)

# Initial Conditions
G.V = Vr + rand(74) * (Vt - Vr)
G.W = Vr + rand(74) * (Vt - Vr)

# Run simulation and time it
tic=time.time(); run(2000 * ms); toc=time.time()

print "Elapsed time: %s seconds" % str(toc - tic)

        