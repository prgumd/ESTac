import numpy as np
import matplotlib.pyplot as plt

AXIS_LABELS = {0:"X",1:"Y",2:"Z"}

def adjust_data(data):
    adjusted_data = {}
    
    end = np.min([value.shape[0] for value in data.values()])
    for key,value in data.items():
        adjusted_data[key] = value[:end]

    return adjusted_data

def viz_data(data,viz_dict):
    fig, axs = plt.subplots(1, 1, figsize=(8, 10))

    x = data["Time"].reshape(-1,)

    y = []
    y_legends = []
    for key,value in viz_dict.items():
        data_array = data[key][value]
        
        print(key)
        print(value)
        print(data_array.shape)
        y_legends += [key+" "+AXIS_LABELS[i] for i in value]
        # if key != "Demodulated Signal":
        #     min_vals = np.min(data_array, axis=1, keepdims=True)
        #     max_vals = np.max(data_array, axis=1, keepdims=True)
        #     data_array = (data_array - min_vals) / (max_vals - min_vals + 1e-8)
        y.append(data_array)
    y = np.vstack(y)
    
    for i in range(y.shape[0]):
        axs.plot(x,y[i],label=y_legends[i])    
    axs.legend()
    plt.show()

def high_pass_filter(value, hp_prev_value, prev_value, tau, dt):
    alpha = tau / (tau + dt)
    return alpha * (hp_prev_value + value - prev_value)

def compute_highpass(times,pressure):
    hp_prev = 0.0
    t_last = -1/6.0
    cutoff = 0.07
    tau = 1/(2*np.pi*cutoff)
    pressure_filtered = []
    p_prev = 0.0
    for t,p in zip(times,pressure):
        dt = t-t_last
        hp = high_pass_filter(p, hp_prev, p_prev, tau, dt)
        pressure_filtered.append(hp)
        t_last =t
        hp_prev = hp
        p_prev = p
    
    return np.array(pressure_filtered)

# omega_x, omega_z, omega_y= 0.9*2*np.pi, 0.83*2*np.pi, 0.7*2*np.pi  
omega_x, omega_z, omega_y= 1.2*2*np.pi, 1.5*2*np.pi, 1.7*2*np.pi
phi1_x, phi1_y, phi1_z = 0.0, 0.0, 0.0


data = np.load("./outputs/final_exp/trail_run.npy",allow_pickle=True).item()

# new_data = {}
# for key, value in data.items():
#     if key != "Tracker":
#         new_data[key] = value
# data = adjust_data(data)

# Create a figure and a 3x1 grid of subplots (3 rows, 1 column)
fig, axs = plt.subplots(10, 1, figsize=(8, 10))  # 3 rows, 1 column

# Plot data in the first subplot
axs[0].plot(data["Time"],data["Estimated Positions"][:,0]-data["Estimated Positions"][0,0], 'r')
axs[0].plot(data["Time"],data["Estimated Positions"][:,1]-data["Estimated Positions"][0,1], 'g')
axs[0].plot(data["Time"],data["Estimated Positions"][:,2]-data["Estimated Positions"][0,2], 'b')
axs[0].set_title('Estimated Positions')
axs[0].set_xticks([])

axs[1].plot(data["Time"],data["Total Error HP"], 'k')
axs[1].plot(data["Time"],0.001*np.sin(omega_x * data["Time"]), 'r')
axs[1].set_title('Total Error vs X Demodulation Signal')
axs[1].set_xticks([])

axs[2].plot(data["Time"],data["Total Error HP"], 'k')
axs[2].plot(data["Time"],0.001*np.sin(omega_y * data["Time"]), 'g')
axs[2].set_title('Total Error vs Y Demodulation Signal')
axs[2].set_xticks([])

axs[3].plot(data["Time"],data["Total Error HP"], 'k')
axs[3].plot(data["Time"],0.001*np.sin(omega_z * data["Time"]), 'b')
axs[3].set_title('Total Error vs Z Demodulation Signal')
axs[3].set_xticks([])

axs[4].plot(data["Time"],data["Position Error"], 'k')
axs[4].set_title('Position Error')
axs[4].set_xticks([])

axs[5].plot(data["Time"],data["Pressure Error"], 'k')    
axs[5].set_title('Pressure Error')
axs[5].set_xticks([])

# axs[6].plot(data["Time"],data["Demodulated Signal"][:,3], 'r')
axs[6].plot(data["Time"],data["Demodulated Signal LP"][:,4], 'g')
axs[6].plot(data["Time"],data["Demodulated Signal LP"][:,5], 'b')
axs[6].plot(data["Time"],np.zeros_like(data["Time"]), 'k--')
axs[6].set_title('Demodulated Signal')
axs[6].set_xticks([])

axs[7].plot(data["Time"],data["Tracker"][:,0,0]-np.mean(data["Tracker"][:,0,0]),label='p1_x')
axs[7].plot(data["Time"],data["Tracker"][:,1,0]-np.mean(data["Tracker"][:,1,0]),label='p1_y')
axs[7].plot(data["Time"],data["Tracker"][:,0,1]-np.mean(data["Tracker"][:,0,1]),label='p2_x')
axs[7].plot(data["Time"],data["Tracker"][:,1,1]-np.mean(data["Tracker"][:,1,1]),label='p2_y')
axs[7].plot(data["Time"],data["Tracker"][:,0,2]-np.mean(data["Tracker"][:,0,2]),label='p3_x')
axs[7].plot(data["Time"],data["Tracker"][:,1,2]-np.mean(data["Tracker"][:,1,2]),label='p3_y')
axs[7].plot(data["Time"],data["Tracker"][:,0,3]-np.mean(data["Tracker"][:,0,3]),label='p4_x')
axs[7].plot(data["Time"],data["Tracker"][:,1,3]-np.mean(data["Tracker"][:,1,3]),label='p4_y')
axs[7].set_xticks([])

axs[8].plot(data["Time"],data["Control Input"][:,0]-data["Control Input"][0,0], 'r')
axs[8].plot(data["Time"],data["Control Input"][:,1]-data["Control Input"][0,1], 'g')
axs[8].plot(data["Time"],data["Control Input"][:,2]-data["Control Input"][0,2], 'b')
axs[8].set_xticks([])
# axs[8].set_xlabel('Time')

axs[9].plot(data["Time"],data["Estimated Angles"][:,0]-data["Estimated Angles"][0,0], 'r')
axs[9].plot(data["Time"],data["Estimated Angles"][:,1]-data["Estimated Angles"][0,1], 'b')
axs[9].plot(data["Time"],data["Estimated Angles"][:,2]-data["Estimated Angles"][0,2], 'g')
axs[9].set_title('Estimated Angles')
axs[9].set_xlabel('Time')


plt.show()

