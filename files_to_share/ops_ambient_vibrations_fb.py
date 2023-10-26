# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:33:58 2023

@author: lucag
"""



import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt



# analysis with imposed motion


ops.wipe()
ops.model('basic', '-ndm', 1, '-ndf', 1)

m = 1


ops.node(1, 0); ops.fix(1,1)
ops.node(2, 0); ops.mass(2, m)
ops.node(3, 0); ops.mass(3, m)
ops.node(4, 0); ops.mass(4, m)

k = 100


ops.uniaxialMaterial('Elastic',1,k) 
ops.element('zeroLength',1,1,2,'-mat',1,'-dir',1)
ops.element('zeroLength',2,2,3,'-mat',1,'-dir',1)
ops.element('zeroLength',3,3,4,'-mat',1,'-dir',1)

eigenValues = ops.eigen('-fullGenLapack',3)
ops.modalDamping(0.02, 0.02, 0.02)

dt = 0.01

nsteps = 10000


ts1 = np.random.randn(nsteps,3)

node_i = 2
ops.timeSeries('Path', node_i, '-dt', dt, '-values', *ts1[:,node_i-2])
ops.pattern('MultipleSupport', node_i)
ops.groundMotion(node_i, 'Plain', '-accel', node_i)
ops.imposedMotion(node_i, 1,node_i)
node_i = 3
ops.timeSeries('Path', node_i, '-dt', dt, '-values', *ts1[:,node_i-2])
ops.pattern('MultipleSupport', node_i)
ops.groundMotion(node_i, 'Plain', '-accel', node_i)
ops.imposedMotion(node_i, 1,node_i)
node_i = 4
ops.timeSeries('Path', node_i, '-dt', dt, '-values', *ts1[:,node_i-2])
ops.pattern('MultipleSupport', node_i)
ops.groundMotion(node_i, 'Plain', '-accel', node_i)
ops.imposedMotion(node_i, 1,node_i)
#

ops.constraints("Transformation")
ops.system("UmfPack")
ops.analysis('Transient')




accel2 = []
accel3 = []
accel4 = []

vel2 = []
vel3 = []
vel4 = []

disp2 = []
disp3 = []
disp4 = []

for _ in range(nsteps):
    ops.analyze(1,dt)
    accel2.append(ops.nodeAccel(2)[0])
    accel3.append(ops.nodeAccel(3)[0])
    accel4.append(ops.nodeAccel(4)[0])
    disp2.append(ops.nodeDisp(2)[0])
    disp3.append(ops.nodeDisp(3)[0])
    disp4.append(ops.nodeDisp(4)[0])
    vel2.append(ops.nodeVel(2)[0])
    vel3.append(ops.nodeVel(3)[0])
    vel4.append(ops.nodeVel(4)[0])
ops.wipe()



fig, axs = plt.subplots(3, 1, figsize=(15, 12))

axs[0].plot(np.arange(nsteps) * dt, disp2, label="2")
axs[0].plot(np.arange(nsteps) * dt, disp3, label="3")
axs[0].plot(np.arange(nsteps) * dt, disp4, label="4")
axs[0].set_title("Displacement")
axs[0].legend()

axs[1].plot(np.arange(nsteps) * dt, vel2, label="2")
axs[1].plot(np.arange(nsteps) * dt, vel3, label="3")
axs[1].plot(np.arange(nsteps) * dt, vel4, label="4")
axs[1].set_title("Velocity")
axs[1].legend()

axs[2].plot(np.arange(nsteps) * dt, accel2, label="2")
axs[2].plot(np.arange(nsteps) * dt, accel3, label="3")
axs[2].plot(np.arange(nsteps) * dt, accel4, label="4")
axs[2].set_title("Acceleration")
axs[2].legend()

plt.tight_layout()

plt.show()






# # Compute the FFT


fft_result = np.fft.fft(np.array(accel2))
frequencies = np.fft.fftfreq(len(accel2), dt)

# Shift the zero frequency component to the center
fft_result = np.fft.fftshift(fft_result)
frequencies = np.fft.fftshift(frequencies)

# Plot the FFT magnitude for positive frequencies
positive_frequencies = frequencies[len(frequencies)//2:]
magnitude = np.abs(fft_result[len(fft_result)//2:])
plt.figure(figsize=(10, 4))
plt.plot(positive_frequencies, magnitude)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT Magnitude (Positive Frequencies)')

plt.tight_layout()
plt.show()


#%% ambient excitation with nodal forces




ops.wipe()
ops.model('basic', '-ndm', 1, '-ndf', 1)

m = 1


ops.node(1, 0); ops.fix(1,1)
ops.node(2, 0); ops.mass(2, m)
ops.node(3, 0); ops.mass(3, m)
ops.node(4, 0); ops.mass(4, m)

k = 100


ops.uniaxialMaterial('Elastic',1,k) 
ops.element('zeroLength',1,1,2,'-mat',1,'-dir',1)
ops.element('zeroLength',2,2,3,'-mat',1,'-dir',1)
ops.element('zeroLength',3,3,4,'-mat',1,'-dir',1)

eigenValues = ops.eigen('-fullGenLapack',3)
ops.modalDamping(0.02, 0.02, 0.02)


freq = np.array(eigenValues)**0.5/2/np.pi
print("frequencies :", freq)


dt = 0.01

nsteps = 10000


ts1 = np.random.randn(nsteps,3)

node_i = 2
ops.timeSeries('Path', node_i, '-dt', dt, '-values', *ts1[:,node_i-2])
ops.pattern('Plain', node_i, node_i)
ops.load(node_i, 1)
node_i = 3
ops.timeSeries('Path', node_i, '-dt', dt, '-values', *ts1[:,node_i-2])
ops.pattern('Plain', node_i, node_i)
ops.load(node_i, 1)
node_i = 4
ops.timeSeries('Path', node_i, '-dt', dt, '-values', *ts1[:,node_i-2])
ops.pattern('Plain', node_i, node_i)
ops.load(node_i, 1)
#

ops.constraints("Transformation")
ops.system("UmfPack")
ops.analysis('Transient')




accel2 = []
accel3 = []
accel4 = []

vel2 = []
vel3 = []
vel4 = []

disp2 = []
disp3 = []
disp4 = []

for _ in range(nsteps):
    ops.analyze(1,dt)
    accel2.append(ops.nodeAccel(2)[0])
    accel3.append(ops.nodeAccel(3)[0])
    accel4.append(ops.nodeAccel(4)[0])
    disp2.append(ops.nodeDisp(2)[0])
    disp3.append(ops.nodeDisp(3)[0])
    disp4.append(ops.nodeDisp(4)[0])
    vel2.append(ops.nodeVel(2)[0])
    vel3.append(ops.nodeVel(3)[0])
    vel4.append(ops.nodeVel(4)[0])
ops.wipe()



fig, axs = plt.subplots(3, 1, figsize=(15, 12))

axs[0].plot(np.arange(nsteps) * dt, disp2, label="2")
axs[0].plot(np.arange(nsteps) * dt, disp3, label="3")
axs[0].plot(np.arange(nsteps) * dt, disp4, label="4")
axs[0].set_title("Displacement")
axs[0].legend()

axs[1].plot(np.arange(nsteps) * dt, vel2, label="2")
axs[1].plot(np.arange(nsteps) * dt, vel3, label="3")
axs[1].plot(np.arange(nsteps) * dt, vel4, label="4")
axs[1].set_title("Velocity")
axs[1].legend()

axs[2].plot(np.arange(nsteps) * dt, accel2, label="2")
axs[2].plot(np.arange(nsteps) * dt, accel3, label="3")
axs[2].plot(np.arange(nsteps) * dt, accel4, label="4")
axs[2].set_title("Acceleration")
axs[2].legend()

plt.tight_layout()

plt.show()




# # Compute the FFT


fft_result = np.fft.fft(np.array(accel2))
frequencies = np.fft.fftfreq(len(accel2), dt)

# Shift the zero frequency component to the center
fft_result = np.fft.fftshift(fft_result)
frequencies = np.fft.fftshift(frequencies)

# Plot the FFT magnitude for positive frequencies
positive_frequencies = frequencies[len(frequencies)//2:]
magnitude = np.abs(fft_result[len(fft_result)//2:])
plt.figure(figsize=(10, 4))
plt.plot(positive_frequencies, magnitude)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT Magnitude (Positive Frequencies)')
plt.xlim(0,5)
plt.tight_layout()
plt.show()

