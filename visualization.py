"""
Master Studienarbeit im Master Studiengang: Biomedizinische Informationstechnik
Titel: Quantitative Erklärbarkeit tiefer neuronaler Netze in der Analyse von Biosignalen
Autor: Lennart Brakelmann
FH Dortmund
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
    
def subplot_input_signals(segment, mode):
    '''
    Subplots all input signals of a segment
    
    Args:
        - segment = One segment of data
        - mode = define mode of NN
       
    '''
    
    t = range(0,1000)
    
    if mode == 'PPG':
        fig, axs = plt.subplots(6, sharex=True)
        fig.suptitle('All Input Signals')
        fig.supxlabel('Samples')
        fig.supylabel('Amplitude')
        axs[0].plot(t,segment[0,:])
        axs[1].plot(t,segment[1,:])
        axs[2].plot(t,segment[2,:])
        axs[3].plot(t,segment[6,:])
        axs[4].plot(t,segment[7,:])
        axs[5].plot(t,segment[8,:])
        plt.setp(axs, xlim=(0,1000))
    elif mode == 'ABP_multi':
        fig, axs = plt.subplots(3, sharex=True)
        fig.suptitle('All Input Signals')
        fig.supxlabel('Samples')
        fig.supylabel('Amplitude')
        axs[0].plot(t,segment[3,:])
        axs[1].plot(t,segment[4,:])
        axs[2].plot(t,segment[5,:])
        plt.setp(axs, xlim=(0,1000))

    
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
    
def subplot_3_signals_bwr_heatmap(IG, segment, colorbar, mode):
    IG_shape = np.shape(IG)
    t = range(0,IG_shape[2])
    
    subplot_nr = 3
    if mode == 'PPG':
        signals = np.zeros((3,1000))
        signals[0][:] = segment[0]
        signals[1][:] = segment[1]
        signals[2][:] = segment[2]
    elif mode == 'TPPG':
        signals = np.zeros((3,1000))
        signals[0][:] = segment[6]
        signals[1][:] = segment[7]
        signals[2][:] = segment[8]
        
        IG = IG.copy()
        IG[0][0][:] = IG[3][0][:]
        IG[1][0][:] = IG[4][0][:]
        IG[2][0][:] = IG[5][0][:]
        
    elif mode == 'ABP_multi':
        signals = np.zeros((3,1000))
        signals[0][:] = segment[3]
        signals[1][:] = segment[4]
        signals[2][:] = segment[5]
    
    elif mode == 'ABP_multi_900':
        signals = np.zeros((3,900))
        signals[0][:] = segment[3][50:950]
        signals[1][:] = segment[4][50:950]
        signals[2][:] = segment[5][50:950]
    
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('Heatmap')
    fig.supxlabel('Samples')
    fig.supylabel('Attributions/Signals')
    
    if colorbar == 'multi':
        xmin = min(t)
        xmax = max(t)
        #Use multiple colorbars
        for i in range(0,subplot_nr):
            IG_signal = IG[i][0][:]
            IG_signal = np.expand_dims(IG_signal, axis=0)
        #for i in range(low_lim,up_lim):
            ymin = min(signals[i][:])
            ymax = max(signals[i][:])
            axs[i].plot(t, signals[i][:], c='k', linewidth=1.5)
            c = axs[i].imshow(IG_signal, aspect='auto', cmap='seismic', extent=(xmin, xmax, ymin, ymax),norm=colors.CenteredNorm(vcenter=0))
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
        for i in range(0,subplot_nr):
            IG_signal = IG[i][0][:]
            IG_signal = np.expand_dims(IG_signal, axis=0)
            ymin = min(signals[i][:])
            ymax = max(signals[i][:])
            axs[i].plot(t, signals[i][:], c='k', linewidth=1.5)
            c = axs[i].imshow(IG_signal, aspect='auto', cmap='seismic', extent=(xmin, xmax, ymin, ymax), vmin=IG_min, vmax=IG_max)
        #fig.colorbar(c)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(c, cax=cbar_ax)
        
    elif colorbar == 'midpoint_norm':
        xmin = min(t)
        xmax = max(t)
        #Use one colorbar
        for i in range(0,subplot_nr):
            IG_signal = IG[i][0][:]
            IG_signal = np.expand_dims(IG_signal, axis=0)
            ymin = min(signals[i][:])
            ymax = max(signals[i][:])
            axs[i].plot(t, signals[i][:], c='k', linewidth=1.5)
            c = axs[i].imshow(IG_signal, aspect='auto', cmap='seismic', extent=(xmin, xmax, ymin, ymax), norm=colors.CenteredNorm(vcenter=0))
            fig.colorbar(c)