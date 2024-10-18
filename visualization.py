"""
Master Studienarbeit im Master Studiengang: Biomedizinische Informationstechnik
Titel: Quantitative Erklärbarkeit tiefer neuronaler Netze in der Analyse von Biosignalen
Autor: Lennart Brakelmann
FH Dortmund
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

def subplot_all_input_signals(batch_data, segment_nr, n_input_signals):
    '''
    Subplots all input signals of a segment from a given batch over the samples
    
    Args:
        - batch_data = One batch of 2048 segments
        - segment_nr = One segment
        - n_input_signals = Number of input signals
       
    '''
    def get_instance(batch_data, instance_nr, n_input_signals):
        instance = []
    
        for i in range(0,n_input_signals):
            signal =  batch_data[i][instance_nr][:]
            instance.append(signal)
        
        return instance
    
    instance = get_instance(batch_data,segment_nr, n_input_signals)
    
    fig, axs = plt.subplots(6, sharex=True)
    fig.suptitle('All Input Signals')
    fig.supxlabel('Samples')
    fig.supylabel('Amplitude')
    axs[0].plot(range(0,1000),instance[0])
    axs[1].plot(range(0,1000),instance[1])
    axs[2].plot(range(0,1000),instance[2])
    axs[3].plot(range(0,1000),instance[3])
    axs[4].plot(range(0,1000),instance[4])
    axs[5].plot(range(0,1000),instance[5])  
    plt.setp(axs, xlim=(0,1000))
    
def subplot_all_IG_attributions(IG):
    '''
    Subplot all six IG signals of a segment over the samples
    
    Args:
        - IG: IG attributions of one segment for all six input signals
       
    '''
    fig, axs = plt.subplots(6, sharex=True)
    fig.suptitle('Attributions for all Input signals')
    fig.supxlabel('Samples')
    fig.supylabel('Attributions')
    axs[0].plot(range(0,1000), np.squeeze(IG[0]))
    axs[0].grid()
    axs[1].plot(range(0,1000), np.squeeze(IG[1]))
    axs[1].grid()
    axs[2].plot(range(0,1000), np.squeeze(IG[2]))
    axs[2].grid()
    axs[3].plot(range(0,1000), np.squeeze(IG[3]))
    axs[3].grid()
    axs[4].plot(range(0,1000), np.squeeze(IG[4]))
    axs[4].grid()
    axs[5].plot(range(0,1000), np.squeeze(IG[5]))
    axs[5].grid()

    
def plot_signal_heatmap(signal, IG, signal_nr):
    '''
    Plot one time signal with the corresponding IG attributions as a heatmap over the samples
    
    Args:
        - signal: one time signal of the segment (PPG, PPG1, PPG2, TemplatePPG, TemplatePPG1,
                                                  TemplatePPG2)
        - IG: IG attributions of one segment
        - signal_nr: specifies the desired signal number for the IG attributions; must be the same
                     as signal (0 = PPG, 1 = PPG1, 2 = PPG2, 3 = TemplatePPG, 4 = TemplatePPG2,
                                5 = TemplatePPG2)
       
    '''
    IG = IG[signal_nr][0][0:1000].reshape((1,1000))
    t = range(0,1000)
    plt.figure()
    s = plt.scatter(t, signal, c=IG, cmap='seismic')#, ec='k')
    if signal_nr == 0:
        plt.title('PPG0 Signal + Attributions as heatmap and signal')
        #plt.plot(t,np.squeeze(IG), linewidth=1, color='k')
    elif signal_nr == 1:
        plt.title('PPG1 Signal + Attributions as heatmap')
        #plt.plot(t,np.squeeze(IG), linewidth=0.8, color='k')
    elif signal_nr == 2:
        plt.title('PPG2 Signal + Attributions as heatmap')
    elif signal_nr == 3:
        plt.title('Template Signal 1 + Attributions as heatmap and signal')
    elif signal_nr == 4:
        plt.title('Template Signal 2 + Attributions as heatmap and signal')
    elif signal_nr == 5:
        plt.title('Template Signal 3 + Attributions as heatmap and signal')
    plt.plot(t,signal, linewidth=0.8, color='k')
    plt.xlim([0,1000])
    plt.xlabel('Samples')
    plt.ylabel('Amplitude/Attributions')
    plt.colorbar(s)
    plt.grid()
    plt.show()
    
def plot_signal_with_IG(signal,IG, signal_nr):
    '''
    Plots a signal and the IG attributions over the samples in the same plot
    
    Args:
        - IG: IG attributions of one segment for all six input signals
        - signal: one time signal
        - signal_nr: specifies the desired signal number for the IG attributions; must be the same
                     as signal (0 = PPG, 1 = PPG1, 2 = PPG2, 3 = TemplatePPG, 4 = TemplatePPG2,
                                5 = TemplatePPG2)
       
    '''
    IG = IG[signal_nr]
    plt.figure()
    plt.xlabel('Samples')
    plt.ylabel('Amplitude/Attributions')
    if signal_nr == 0:
        plt.title('PPG0 Signal + Attributions')
    elif signal_nr == 1:
        plt.title('PPG1 Signal + Attributions')
    elif signal_nr == 2:
        plt.title('PPG2 Signal + Attributions')
    elif signal_nr == 3:
        plt.title('Template Signal 1 + Attributions')
    elif signal_nr == 4:
        plt.title('Template Signal 2 + Attributions')
    elif signal_nr == 5:
        plt.title('Template Signal 3 + Attributions')
    plt.plot(range(0,1000),signal)
    plt.plot(range(0,1000),np.squeeze(IG))
    plt.grid()
    plt.show()
    
def plot_PPG_heatmap_scatter_subplot(PPG, PPG1, PPG2, IG):
    '''
    Subplot PPG, PPG1 and PPG2 with the corresponding IG attributions as a heatmap over the samples
    
    Args:
        - PPG: PPG time signal
        - PPG1: first derivative of PPG
        - PPG2: second derivative of PPG
        - IG: IG attributions of one segment for all six input signals
       
    '''
    IG0 = IG[0][0][0:1000].reshape((1,1000))
    IG1 = IG[1][0][0:1000].reshape((1,1000))
    IG2 = IG[2][0][0:1000].reshape((1,1000))
    t = range(0,1000)
    
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('PPG, PPG1 and PPG2 + Attributions as heatmap')
    fig.supxlabel('Samples')
    fig.supylabel('Amplitude/Attributions')
    s0 = axs[0].scatter(t, PPG, c=IG0, cmap='jet')#, ec='k')
    axs[0].plot(range(0,1000),PPG, linewidth=0.5, color='k')
    axs[0].grid()
    s1 = axs[1].scatter(t, PPG1, c=IG1, cmap='jet')#, ec='k')
    axs[1].plot(range(0,1000),PPG1, linewidth=0.5, color='k')
    axs[1].grid()
    s2 = axs[2].scatter(t, PPG2, c=IG2, cmap='jet')#, ec='k')
    axs[2].plot(range(0,1000),PPG2, linewidth=0.5, color='k')
    axs[2].grid()
    #Use multiple colorbars
    fig.colorbar(s0)
    fig.colorbar(s1)
    fig.colorbar(s2)
    plt.setp(axs, xlim=(0,1000))
    #Use One Colorbar
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(s0, cax=cbar_ax)

def plot_templates_heatmap_scatter_subplot(TPPG, TPPG1, TPPG2, IG):
    '''
    Subplot Template PPG, Template PPG1 and Template PPG2 with the corresponding IG attributions
    as a heatmap over the samples
    
    Args:
        - TPPG: Template signal for PPG
        - PPG1: Template signal for first derivative of PPG
        - PPG2: Template signal for second derivative of PPG
        - IG: IG attributions of one segment for all six input signals
       
    '''
    IG0 = IG[3][0][0:1000].reshape((1,1000))
    IG1 = IG[4][0][0:1000].reshape((1,1000))
    IG2 = IG[5][0][0:1000].reshape((1,1000))
    t = range(0,1000)
    
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('Template signals + Attributions as heatmap')
    fig.supxlabel('Samples')
    fig.supylabel('Amplitude/Attributions')
    s0 = axs[0].scatter(t, TPPG, c=IG0, cmap='jet')
    axs[0].plot(range(0,1000),TPPG, linewidth=0.5, color='k')
    axs[0].grid()
    s1 = axs[1].scatter(t, TPPG1, c=IG1, cmap='jet')
    axs[1].plot(range(0,1000),TPPG1, linewidth=0.5, color='k')
    axs[1].grid()
    s2 = axs[2].scatter(t, TPPG2, c=IG2, cmap='jet')
    axs[2].plot(range(0,1000),TPPG2, linewidth=0.5, color='k')
    axs[2].grid()
    #Use multiple colorbars --> one for each subplot 
    fig.colorbar(s0)
    fig.colorbar(s1)
    fig.colorbar(s2)
    plt.setp(axs, xlim=(0,1000))
    #Use One Colorbar --> one for all subplots
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(s0, cax=cbar_ax)
    
def subplot_heatmap_and_IG(PPG, IG, signal_nr):
    '''
    Subplot one time signal and the corresponding IG attributions over the samples
    as a heatmap
    
    Args:
        - TPPG: Template signal for PPG
        - PPG1: Template signal for first derivative of PPG
        - PPG2: Template signal for second derivative of PPG
        - IG: IG attributions of one segment for all six input signals
       
    '''
    
    IG0 = IG[signal_nr][0][0:1000].reshape((1,1000))
    t = range(0,1000)
    
    fig, axs = plt.subplots(2, sharex=True)
    if signal_nr == 0:
        fig.suptitle('PPG0 Signal + Attributions as heatmap and signal')
    elif signal_nr == 1:
        fig.suptitle('PPG1 Signal + Attributions as heatmap and signal')
    elif signal_nr == 2:
        fig.suptitle('PPG2 Signal + Attributions as heatmap and signal')
    elif signal_nr == 3:
        fig.suptitle('Template Signal 1 + Attributions as heatmap and signal')
    elif signal_nr == 4:
        fig.suptitle('Template Signal 2 + Attributions as heatmap and signal')
    elif signal_nr == 5:
        fig.suptitle('Template Signal 3 + Attributions as heatmap and signal')
    fig.supxlabel('Samples')
    fig.supylabel('Amplitude/Attributions')
    s0 = axs[0].scatter(t, PPG, c=IG0, cmap='jet', ec='k')
    axs[0].plot(range(0,1000),PPG, linewidth=0.5)
    axs[1].plot(range(0,1000),np.squeeze(IG0))
    
    fig.colorbar(s0)

    
def subplot_many_templates_in_one(IG):
    '''
    Subplot IG attributions of 20 segments for TemplatePPG, TemplatePPG1 and TemplatePPG2 over
    the samples
    
    Args:
        - IG: IG attributions of 20 segments for all six input signals
       
    '''
    t = range(0,1000)
    fig, axs = plt.subplots(3)
    fig.suptitle('All Templatesignal Attributions')
    fig.supxlabel('Samples')
    fig.supylabel('Attributions')
    for i in range(0,20):
        axs[0].plot(t, np.squeeze(IG[i][3]))
        axs[1].plot(t, np.squeeze(IG[i][4]))
        axs[2].plot(t, np.squeeze(IG[i][5]))
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    
def plot_IG_for_interpolation_steps(IG25, IG75, IG175):
    '''
    Subplot 3 different IG attributions over the samples for all six time signals
    
    Args:
        - IG25: IG attributions for one segment calculated with 25 interpolation steps
        - IG75: IG attributions for one segment calculated with 75 interpolation steps
        - IG175: IG attributions for one segment calculated with 175 interpolation steps
       
    '''
    fig, axs = plt.subplots(6, sharex=True)
    fig.suptitle('All Signals + Attributions with different step size')
    fig.supxlabel('Samples')
    fig.supylabel('Attributions')
    axs[0].plot(range(0,1000), np.squeeze(IG25[0]), label='25')
    axs[0].plot(range(0,1000), np.squeeze(IG75[0]), label='75')
    axs[0].plot(range(0,1000), np.squeeze(IG175[0]), label='175')
    axs[0].legend()
    axs[0].grid()
    axs[1].plot(range(0,1000), np.squeeze(IG25[1]))
    axs[1].plot(range(0,1000), np.squeeze(IG75[1]))
    axs[1].plot(range(0,1000), np.squeeze(IG175[1]))
    axs[1].grid()
    axs[2].plot(range(0,1000), np.squeeze(IG25[2]))
    axs[2].plot(range(0,1000), np.squeeze(IG75[2]))
    axs[2].plot(range(0,1000), np.squeeze(IG175[2]))
    axs[2].grid()
    axs[3].plot(range(0,1000), np.squeeze(IG25[3]))
    axs[3].plot(range(0,1000), np.squeeze(IG75[3]))
    axs[3].plot(range(0,1000), np.squeeze(IG175[3]))
    axs[3].grid()
    axs[4].plot(range(0,1000), np.squeeze(IG25[4]))
    axs[4].plot(range(0,1000), np.squeeze(IG75[4]))
    axs[4].plot(range(0,1000), np.squeeze(IG175[4]))
    axs[4].grid()
    axs[5].plot(range(0,1000), np.squeeze(IG25[5]))
    axs[5].plot(range(0,1000), np.squeeze(IG75[5]))
    axs[5].plot(range(0,1000), np.squeeze(IG175[5]))
    axs[5].grid()
        
def plot_3PPG_heatmap_scatter_subplot(PPG, PPG1, PPG2, IG, IG1, IG2, signal):
    '''
    Subplot PPG, PPG1 and PPG2 with the corresponding IG attributions as a heatmap over the samples
    
    Args:
        - PPG: PPG time signal 1
        - PPG1: PPG time signal 2
        - PPG2: PPG time signal 3
        
        - IG: IG attributions 1 of one segment for all six input signals
        - IG: IG attributions 2 of one segment for all six input signals
        - IG: IG attributions 3 of one segment for all six input signals
       
    '''
    
    if signal == 1:
        signal_nr = 0
    elif signal == 2:
        signal_nr = 1
    elif signal == 3:
        signal_nr = 2
    elif signal == 4:
        signal_nr = 3
    elif signal == 5:
        signal_nr = 4
    elif signal == 6:
        signal_nr = 5
        
    IG_0 = IG[signal_nr][0][0:1000].reshape((1,1000))
    IG_1 = IG1[signal_nr][0][0:1000].reshape((1,1000))
    IG_2 = IG2[signal_nr][0][0:1000].reshape((1,1000))
    t = range(0,1000)
    
    fig, axs = plt.subplots(3, sharex=True)
    if signal_nr == 0:
        fig.suptitle('PPG from 3 time segments + Attributions as heatmap')
    elif signal_nr == 1:
        fig.suptitle('PPG1 from 3 time segments + Attributions as heatmap')
    elif signal_nr == 2:
        fig.suptitle('PPG2 from 3 time segments + Attributions as heatmap')
    elif signal_nr == 3:
        fig.suptitle('Template PPG from 3 time segments + Attributions as heatmap')
    elif signal_nr == 4:
        fig.suptitle('Template PPG1 from 3 time segments + Attributions as heatmap')
    elif signal_nr == 5:
        fig.suptitle('Template PPG2 from 3 time segments + Attributions as heatmap')
    fig.supxlabel('Samples')
    fig.supylabel('Amplitude/Attributions')
    s0 = axs[0].scatter(t, PPG, c=IG_0, cmap='jet')#, ec='k')
    axs[0].plot(range(0,1000),PPG, linewidth=0.5, color='k')
    axs[0].grid()
    s1 = axs[1].scatter(t, PPG1, c=IG_1, cmap='jet')#, ec='k')
    axs[1].plot(range(0,1000),PPG1, linewidth=0.5, color='k')
    axs[1].grid()
    s2 = axs[2].scatter(t, PPG2, c=IG_2, cmap='jet')#, ec='k')
    axs[2].plot(range(0,1000),PPG2, linewidth=0.5, color='k')
    axs[2].grid()
    #Use multiple colorbars
    fig.colorbar(s0)
    fig.colorbar(s1)
    fig.colorbar(s2)
    plt.setp(axs, xlim=(0,1000))
    #Use One Colorbar
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(s0, cax=cbar_ax)
    
def AOPC_curve_plot(ground_truth, all_f_x_k, mode):
    k = len(all_f_x_k)
    diff = []
    for i in range(0,k):
        if mode == 'SBP':
            diff.append(ground_truth-all_f_x_k[i][0][0])
        elif mode == 'DBP':
            diff.append(ground_truth-all_f_x_k[i][0][1])
            
    diff = np.array(diff)
    
    diff = np.insert(diff, 0, 0, axis=0)
    k = k+1
    
    plt.figure()
    plt.plot(range(0,k),diff)
    plt.xlabel('Perturbation step k')
    plt.ylabel('Diff')
    plt.title('AOPC Curve Plot')
    plt.grid()
    
    
def plot_onesignal_bwr_heatmap(IG, signal, signal_nr):
    IG_signal = IG[signal_nr][0][:]
    IG_signal = np.expand_dims(IG_signal, axis=0)
    
    t = range(0,len(signal))
    xmin = min(t)
    xmax = max(t)
    ymin = min(signal)
    ymax = max(signal)
    
    plt.figure()
    plt.plot(t,signal, c='k', linewidth=2)
    # bwr colormap
    plt.imshow(IG_signal, aspect='auto', cmap='bwr', extent=(xmin, xmax, ymin, ymax),norm=colors.CenteredNorm(vcenter=0))
    # seismic colormap
    #plt.imshow(IG_signal, aspect='auto', cmap='seismic', extent=(xmin, xmax, ymin, ymax))
    plt.colorbar()
    plt.title(f'Heatmap Plot Signal: {signal_nr}')
    plt.xlabel('samples')
    plt.ylabel('Signal')
    #plt.xlim((0,500))
    
def subplot_all_signals_bwr_heatmap(IG, signals, subject_nr, colorbar):
    IG_shape = np.shape(IG)
    t = range(0,IG_shape[2])
    
    fig, axs = plt.subplots(IG_shape[0], sharex=True)
    fig.suptitle('Attributions for all Input signals visualized as heatmap')
    fig.supxlabel('Samples')
    fig.supylabel('Attributions')
    
    if colorbar == 'multi':
        xmin = min(t)
        xmax = max(t)
        #Use one colorbar
        for i in range(0,IG_shape[0]):
            IG_signal = IG[i][0][:]
            IG_signal = np.expand_dims(IG_signal, axis=0)
            ymin = min(signals[i][subject_nr][:])
            ymax = max(signals[i][subject_nr][:])
            axs[i].plot(t, signals[i][subject_nr][:], c='k', linewidth=2)
            c = axs[i].imshow(IG_signal, aspect='auto', cmap='bwr', extent=(xmin, xmax, ymin, ymax))
            fig.colorbar(c)
    elif colorbar == 'single':
        xmin = min(t)
        xmax = max(t)
        IG_matrix = np.squeeze(IG.copy())
        matrix_shape = np.shape(IG_matrix)
        IG_vector = np.reshape(IG_matrix, (1, matrix_shape[0]*matrix_shape[1]))
        IG_min = np.min(IG_vector)
        IG_max = np.max(IG_vector)
        maximum_value = np.max([np.abs(IG_min),np.abs(IG_max)])
        IG_min = -np.abs(maximum_value)
        IG_max = np.abs(maximum_value)
        #Use multiple colorbars
        for i in range(0,IG_shape[0]):
            IG_signal = IG[i][0][:]
            IG_signal = np.expand_dims(IG_signal, axis=0)
            ymin = min(signals[i][subject_nr][:])
            ymax = max(signals[i][subject_nr][:])
            axs[i].plot(t, signals[i][subject_nr][:], c='k', linewidth=2)
            c = axs[i].imshow(IG_signal, aspect='auto', cmap='bwr', extent=(xmin, xmax, ymin, ymax), vmin=IG_min, vmax=IG_max)
        #fig.colorbar(c)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(c, cax=cbar_ax)
        
    elif colorbar == 'midpoint_norm':
        xmin = min(t)
        xmax = max(t)
        #Use one colorbar
        for i in range(0,IG_shape[0]):
            IG_signal = IG[i][0][:]
            IG_signal = np.expand_dims(IG_signal, axis=0)
            ymin = min(signals[i][subject_nr][:])
            ymax = max(signals[i][subject_nr][:])
            axs[i].plot(t, signals[i][subject_nr][:], c='k', linewidth=2)
            c = axs[i].imshow(IG_signal, aspect='auto', cmap='bwr', extent=(xmin, xmax, ymin, ymax), norm=colors.CenteredNorm(vcenter=0))
            fig.colorbar(c)
    
def subplot_3_signals_bwr_heatmap(IG, signals, subject_nr, colorbar, mode):
    IG_shape = np.shape(IG)
    t = range(0,IG_shape[2])
    
    if mode == 'PPG':
        low_lim = 0
        up_lim = 3
    elif mode == 'TPPG':
        low_lim = 3
        up_lim = 6
    
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('Attributions for all Input signals visualized as heatmap')
    fig.supxlabel('Samples')
    fig.supylabel('Attributions')
    
    if colorbar == 'multi':
        xmin = min(t)
        xmax = max(t)
        #Use multiple colorbars
        for i in range(0,IG_shape[0]):
            IG_signal = IG[i][0][:]
            IG_signal = np.expand_dims(IG_signal, axis=0)
        for i in range(low_lim,up_lim):
            ymin = min(signals[i][subject_nr][:])
            ymax = max(signals[i][subject_nr][:])
            axs[i].plot(t, signals[i][subject_nr][:], c='k', linewidth=2)
            c = axs[i].imshow(IG_signal, aspect='auto', cmap='bwr', extent=(xmin, xmax, ymin, ymax))
            fig.colorbar(c)
    elif colorbar == 'single':
        xmin = min(t)
        xmax = max(t)
        IG_matrix = np.squeeze(IG.copy())
        matrix_shape = np.shape(IG_matrix)
        IG_vector = np.reshape(IG_matrix, (1, matrix_shape[0]*matrix_shape[1]))
        IG_min = np.min(IG_vector)
        IG_max = np.max(IG_vector)
        maximum_value = np.max([np.abs(IG_min),np.abs(IG_max)])
        IG_min = -np.abs(maximum_value)
        IG_max = np.abs(maximum_value)
        #Use one colorbar
        for i in range(low_lim,up_lim):
            IG_signal = IG[i][0][:]
            IG_signal = np.expand_dims(IG_signal, axis=0)
            ymin = min(signals[i][subject_nr][:])
            ymax = max(signals[i][subject_nr][:])
            axs[i].plot(t, signals[i][subject_nr][:], c='k', linewidth=2)
            c = axs[i].imshow(IG_signal, aspect='auto', cmap='bwr', extent=(xmin, xmax, ymin, ymax), vmin=IG_min, vmax=IG_max)
        #fig.colorbar(c)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(c, cax=cbar_ax)
        
    elif colorbar == 'midpoint_norm':
        xmin = min(t)
        xmax = max(t)
        #Use one colorbar
        for i in range(low_lim,up_lim):
            IG_signal = IG[i][0][:]
            IG_signal = np.expand_dims(IG_signal, axis=0)
            ymin = min(signals[i][subject_nr][:])
            ymax = max(signals[i][subject_nr][:])
            axs[i].plot(t, signals[i][subject_nr][:], c='k', linewidth=2)
            c = axs[i].imshow(IG_signal, aspect='auto', cmap='bwr', extent=(xmin, xmax, ymin, ymax), norm=colors.CenteredNorm(vcenter=0))
            fig.colorbar(c)