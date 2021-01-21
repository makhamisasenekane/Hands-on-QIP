#hydrogen variational quantum eigensolver
#code adopted from: www.pennylane.ai

import pennylane as qml
from pennylane import numpy as np #use numpy from pennylane instead of the standard numpy
from matplotlib import pyplot as plt

geometry = 'h2.xyz'
name ='h2'
charge = 0
multiplicity=1
basis= 'sto-3g' 
h, nr_qubits = qml.qchem.generate_hamiltonian(
    name,
    geometry,
    charge,
    multiplicity,
    basis,
    mapping='jordan_wigner',
    n_active_orbitals=2,
    n_active_electrons=2,
)

print("Hamiltonian is: \n", h)
print("Number of qubits is: \n", nr_qubits)

dev = qml.device('default.qubit', wires=nr_qubits)

def ansatz(params, wires):
    qml.BasisState(np.array([1, 1, 0, 0]), wires=wires)
    for i in wires:
        qml.Rot(*params[i], wires=i)
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 1])

cost_fn = qml.VQECost(ansatz, h, dev)    

opt = qml.GradientDescentOptimizer(stepsize=0.4)
np.random.seed(42)
params = np.random.normal(0, np.pi, (nr_qubits, 3))

print(params)


max_iterations = 250
step_size = 0.05
conv_tol = 1e-06

prev_energy = cost_fn(params)
for n in range(max_iterations):
    params = opt.step(cost_fn, params)
    energy = cost_fn(params)
    conv = np.abs(energy - prev_energy)

    if n % 20 == 0:
        print('Iteration = {:},  Ground-state energy = {:.8f} Ha,  Convergence parameter = {'
              ':.8f} Ha'.format(n, energy, conv))

    if conv <= conv_tol:
        break

    prev_energy = energy

print()
print('Final convergence parameter = {:.8f} Ha'.format(conv))
print('Final value of the ground-state energy = {:.8f} Ha'.format(energy))
print('Accuracy with respect to the FCI energy: {:.8f} Ha ({:.8f} kcal/mol)'.
        format(np.abs(energy - (-1.136189454088)), np.abs(energy - (-1.136189454088))*627.503))
print()
print('Final circuit parameters = \n', params)
