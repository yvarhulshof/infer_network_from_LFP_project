import numpy as np
import pylab as plt

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def run_simulation(t_sim=1000, stim_strength=0., stim_start=100, stim_dur=100, population=0, seed=55, output='I'):
    """
    Run a simulation to obtain data

    Parameters
    ----------
    t_sim           -- simulation time
    stim_strength   -- external stimulation strength [Hz]
    stim_start      -- external stimulation start [ms]
    stim_dur        -- external stimulation duration [ms]
    population      -- index of population for stimulation
    seed            -- random seed

    Returns
    -------
    data       -- average membrane voltage per population
    """

    print('Run spiking neural network...')
    from stimulus_params import stim_dict
    from network_params import net_dict
    from sim_params import sim_dict
    import network
    import numpy as np
    import os

    import IPython

    sim_dict['rng_seed'] = seed
    sim_dict['data_path'] = os.path.join(os.getcwd(), 'data/')

    if stim_strength != 0:
        stim_dict['thalamic_input'] = True
    stim_dict['conn_probs_th'] = np.zeros(8)
    stim_dict['conn_probs_th'][population] = 0.05
    stim_dict['th_rate'] = stim_strength
    stim_dict['th_start'] = sim_dict['t_presim'] + stim_start
    stim_dict['th_duration'] = stim_dur

    net = network.Network(sim_dict, net_dict, stim_dict)
    net.create()
    net.connect()
    net.simulate(sim_dict['t_presim'])
    net.simulate(t_sim)

    print('Collect data from file...')
    data = []
    data_neuron = []
    plt.figure()
    for i in range(8):
        print('\tPopulation {}/8'.format(i+1))
        if output == 'V_m':
            datContent = [i.strip().split() for i in open("data/voltmeter-{}-0.dat".format(7718+i)).readlines()]
            dat_ = np.array(datContent[3:]).astype(float)
            data_neuron.append(dat_)

            min_t = np.min(dat_[:, 1])
            max_t = np.max(dat_[:, 1])

            V_m = np.zeros(int(max_t - min_t) + 1)  # if resolution (dt) = 1 ms

            for i in range(np.shape(dat_)[0]):
                t_idx = int(dat_[i, 1] - min_t)
                V_m[t_idx] += dat_[i, 2]
            N = len(np.unique(dat_[:, 0]))
            V_m /= N

            # low-pass filter from 1.0 ms to 3.0 ms (LFP is the low-pass filtered signal cutoff at 300 Hz)
            V_m = running_mean(V_m, 3)

            data.append(V_m)

        if output == 'I':
            # excitatory current input
            datContent = [i.strip().split() for i in open("data/currentmeter_E-{}-0.dat".format(7726+i)).readlines()]
            dat_E = np.array(datContent[3:]).astype(float)
            # inhibitory current input
            datContent = [i.strip().split() for i in open("data/currentmeter_I-{}-0.dat".format(7734+i)).readlines()]
            dat_I = np.array(datContent[3:]).astype(float)

            min_t = np.min(dat_E[:, 1])
            max_t = np.max(dat_E[:, 1])

            I = np.zeros(int(max_t - min_t) + 1)  # if resolution (dt) = 1 ms

            for i in range(np.shape(dat_E)[0]):
                t_idx = int(dat_E[i, 1] - min_t)
                I[t_idx] += dat_E[i, 2]
                I[t_idx] += dat_I[i, 2]
            N = len(np.unique(dat_E[:, 0]))
            I /= N

            # low-pass filter from 1.0 ms to 3.0 ms (LFP is the low-pass filtered signal cutoff at 300 Hz)
            I = running_mean(I, 3)

            data.append(I)

    return np.array(data), np.array(data_neuron)

def plot_data(data):
    plt.figure(figsize=(4, 4))
    for i in range(len(data)):
        mean = np.mean(data[i])
        plt.plot(data[i] - mean + np.var(data) * i, color='black')
    plt.xlim(left=0, right=1000)
    plt.ylabel(r'$V_m$ [mV]')
    plt.xlabel(r'$Time$ [ms]')
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def evaluate(W):
    """
    Parameters
    ----------
    W:      estimate of the effective connectivity

    Returns
    -------
    Mean squared error of the difference between the ground truth and the estimated connectivity
    """
    W_gt = np.array(    # ground truth connectivity
        [[0.51,  -1.000, 0.454, -0.433, 0.037, -0.000, 0.025, -0.000],
         [0.195, -0.225, 0.046, -0.076, 0.025, -0.000, 0.004, -0.000],
         [0.039, -0.034, 0.274, -0.780, 0.008, -0.000, 0.164, -0.000],
         [0.091, -0.004, 0.111, -0.234, 0.001, -0.000, 0.099, -0.000],
         [0.119, -0.081, 0.062, -0.007, 0.023, -0.108, 0.016, -0.000],
         [0.014, -0.008, 0.007, -0.001, 0.004, -0.019, 0.001, -0.000],
         [0.052, -0.025, 0.075, -0.059, 0.046, -0.014, 0.094, -0.485],
         [0.025, -0.001, 0.002, -0.000, 0.004, -0.001, 0.032, -0.061]])

    W_est = W / np.max(abs(W))

    return np.mean(((W_gt - W_est)**2))

if __name__ == "__main__":

    data = run_simulation(t_sim=1000, stim_strength=10, stim_start=100, stim_dur=10, population=0, output='V_m')

    plot_data(data)