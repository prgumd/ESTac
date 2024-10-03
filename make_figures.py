import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

import scienceplots
plt.style.use('science')

# Success data
# d_s = np.load('figure_data/pinned_0_1.9__0_0_0_trail1.npy', allow_pickle=True).item()
# pprint(d_s)

# Fail data
d_s = np.load('figure_data/pinned_2_0__0_0_0_trail3.npy', allow_pickle=True).item()

fig = plt.figure()
# for i in range(6):
#     plt.subplot(6, 2, i*2+1)

#     plt.plot(d_s['Time'], d_s['Control Input'][:, i])
#     plt.grid()

t0 = 0.0
tf = 150.0

def plot_settings():
    plt.grid()
    plt.xlim([t0, tf])
    ax = plt.gca()
    ax.get_yaxis().set_label_coords(-0.125,0.5)

rows = 7
cols = 1
plt.subplot(rows, cols, 1)
plt.plot(d_s['Time'], d_s['Total Error'] * 1000)
# plt.plot(d_s['Time'], d_s['Total Error HP'] * 1000)
plt.ylabel('$L$')
plot_settings()

plt.subplot(rows, cols, 2)
plt.plot(d_s['Time'], (d_s['Estimated Positions'][:, 0] - d_s['Estimated Positions'][0, 0])  * 1000)
# plt.ylim([-2.0, 0.0])
plt.ylabel('$X$ (mm)')
plot_settings()


plt.subplot(rows, cols, 3)
plt.plot(d_s['Time'], (d_s['Estimated Positions'][:, 1] - d_s['Estimated Positions'][0, 1])  * 1000)
plot_settings()
plt.ylabel('$Y$ (mm)')

plt.subplot(rows, cols, 4)
plt.plot(d_s['Time'], (d_s['Estimated Positions'][:, 2] - d_s['Estimated Positions'][0, 2])  * 1000)
plt.ylabel('$Z$ (mm)')
plot_settings()

plt.subplot(rows, cols, 5)
plt.plot(d_s['Time'], (d_s['Estimated Angles'][:, 0] - d_s['Estimated Angles'][0, 0]) * 180.0 / np.pi)
plt.ylabel('$\\alpha$ (deg)')
plot_settings()

plt.subplot(rows, cols, 6)
plt.plot(d_s['Time'], (d_s['Estimated Angles'][:, 1] - d_s['Estimated Angles'][0, 1]) * 180.0 / np.pi)
plt.ylabel('$\\beta$ (deg)')
plot_settings()

plt.subplot(rows, cols, 7)
plt.plot(d_s['Time'], (d_s['Estimated Angles'][:, 2] - d_s['Estimated Angles'][0, 2]) * 180.0 / np.pi)
plt.xlabel('t (seconds)')
plt.ylabel('$\\gamma$ (deg)')
plot_settings()

fig.set_size_inches(5, 6)
fig.tight_layout()
fig.savefig('key_insertion_plot.png', dpi=300)


# plt.subplot(rows, cols, 3)
# plt.plot(d_s['Time'], d_s['Estimated Angles'])
# plot_settings()

# signal_i = 0
# plt.subplot(rows, cols, 2)
# plt.plot(d_s['Time'], d_s['Demodulated Signal'][:, signal_i] * 1000)
# plot_settings()

# plt.subplot(rows, cols, 3)
# plt.plot(d_s['Time'], d_s['Demodulated Signal LP'][:, signal_i] * 1000)
# plot_settings()

# for i in range(6):
#     plt.subplot(6, 2, i*2+2)

#     plt.plot(d_f['Time'], d_f['Control Input'][:, i])
#     plt.grid()

plt.show()

