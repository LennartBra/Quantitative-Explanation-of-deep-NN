"""
Master Studienarbeit im Master Studiengang: Biomedizinische Informationstechnik
Titel: Quantitative Erklärbarkeit tiefer neuronaler Netze in der Analyse von Biosignalen
Autor: Lennart Brakelmann
FH Dortmund
"""
import numpy as np
import tensorflow as tf

#%% AOPC (Area over perturbation curve) Metric
def rank_attributions(IG, pattern, window_length):
    IG_matrix = np.squeeze(IG)
    shape = np.shape(IG_matrix)
    n_input_signals = shape[0]
    #Calculate sums ove the window_length
    distances = np.arange(0,shape[1],window_length)
    summed_attributions_mat = np.zeros((n_input_signals,len(distances)))
    for i in range(0,n_input_signals):
        for j in range(0,len(distances)):
            #print(f'input_signals: {i}, distances: {distances[j]+window_length}')
            Sum = np.sum(np.abs(IG_matrix[i,distances[j]:distances[j]+window_length]))
            summed_attributions_mat[i,j] = Sum
    
    mat_shape = np.shape(summed_attributions_mat)
    
    if pattern == 'morf': #most relevant first
        sum_atts_vector = np.reshape(summed_attributions_mat,(1,mat_shape[0]*mat_shape[1]))
        vec_shape = np.shape(sum_atts_vector)

        #Sort summed attributions in descending order --> morf
        atts_sorted_indices = np.flip(np.argsort(sum_atts_vector))

        atts_sorted_vector = np.zeros(vec_shape)
        #rank the sorted attribution indices
        for i in range(0,vec_shape[1]):
            atts_sorted_vector[0,atts_sorted_indices[0,i]] = i  
        #Reshape from vector form to matrix form
        ranks = np.reshape(atts_sorted_vector, mat_shape)
    
    if pattern == 'lerf': #least relevant first
        #Reshape summed attributions from matrix form to vector form --> e.g. (6,100) to (1,600)
        sum_atts_vector = np.reshape(summed_attributions_mat,(1,mat_shape[0]*mat_shape[1]))
        vec_shape = np.shape(sum_atts_vector)

        #Sort summed attributions in ascending order --> lerf
        atts_sorted_indices = np.argsort(sum_atts_vector)

        atts_sorted_vector = np.zeros(vec_shape)
        #rank the sorted attribution indices
        for i in range(0,vec_shape[1]):
            atts_sorted_vector[0,atts_sorted_indices[0,i]] = i  
        #Reshape from vector form to matrix form
        ranks = np.reshape(atts_sorted_vector, mat_shape)
    
    return summed_attributions_mat, ranks


def replace_k_features(x, ranks, k, window_length, replacement_strategy):
    x_replaced = []
    x_temp = np.squeeze(np.array(x))
    #replace features with defined replacement_strategy
    if replacement_strategy == 'local_mean':
        #replace k features
        for i in range(0,k):
            #Get index of k-th feature
            rank_index = np.where(ranks==i)
            x_index = window_length * rank_index[1][0]
            #print(f'k: {k}, x_index:{x_index}, rank_index:{rank_index[0][0]}')
            #Calculate local mean
            local_mean = np.mean(x_temp[rank_index[0][0],x_index:x_index+window_length])
            #print(f'values:{x_temp[rank_index[0][0],x_index:x_index+window_length]}, local_mean: {local_mean}')
            #Replace k-th feature with local mean
            x_temp[rank_index[0][0],x_index:x_index+window_length] = local_mean
            #Append x_temp to x_replaced
            x_replaced.append(x_temp.copy())
            
    elif replacement_strategy == 'global_mean':
        x_copy = np.squeeze(x.copy())
        #replace k features
        for i in range(0,k):
            #Get index of k-th feature
            rank_index = np.where(ranks==i)
            x_index = window_length * rank_index[1][0]
            #print(f'k: {k}, x_index:{x_index}, rank_index:{rank_index[0][0]}')
            #Calculate local mean
            global_mean = np.mean(x_copy[rank_index[0][0],:])
            #print(f'values:{x_temp[rank_index[0][0],x_index:x_index+window_length]}, global_mean: {global_mean}')
            #Replace k-th feature with local mean
            x_temp[rank_index[0][0],x_index:x_index+window_length] = global_mean
            #Append x_temp to x_replaced
            x_replaced.append(x_temp.copy())
    
    elif replacement_strategy == 'zeros':
        for i in range(0,k):
            #Get index of k-th feature
            rank_index = np.where(ranks==i)
            x_index = window_length * rank_index[1][0]
            #print(f'k: {k}, x_index:{x_index}, rank_index:{rank_index[0][0]}')
            #Replace k-th feature with zeros
            x_temp[rank_index[0][0],x_index:x_index+window_length] = 0
            #Append x_temp to x_replaced
            x_replaced.append(x_temp.copy())
    
    return x_replaced


def calculate_AOPC(x, IG, k, pattern, window_length, replacement_strategy, model):
    #Rank attributions
    IG_sum_atts, IG_ranks = rank_attributions(IG, pattern, window_length)
    #Replace k features depending on the hyperparameters
    x_replaced = replace_k_features(x, IG_ranks, k, window_length, replacement_strategy)
    #Calculate AOPC with formula
    f_x = model.predict(x, verbose=0)
    summe = 0
    all_f_x_k = []
    for i in range(0,k):
        #Make tf Tensor from numpy array
        x_k = [tf.cast(np.expand_dims(x_replaced[i][j],axis=0), tf.float32) for j in range(0,6)]
        f_x_k = model.predict(x_k, verbose=0)
        diff = f_x - f_x_k
        summe = summe + diff
        all_f_x_k.append(f_x_k)
    AOPC = 1/k * summe
    
    return AOPC, all_f_x_k





#%% APT (Ablation Percentage Threshold) Metric

def replace_feature(x, x_ground_truth, ranks, k, window_length, replacement_strategy):
    x_temp = x.copy()
    #replace features with defined replacement_strategy
    if replacement_strategy == 'local_mean':
        #replace k-th feature
        #Get index of k-th feature
        rank_index = np.where(ranks==k)
        x_index = window_length * rank_index[1][0]
        #print(f'k: {k}, x_index:{x_index}, rank_index:{rank_index[0][0]}')
        #Calculate local mean
        local_mean = np.mean(x_temp[rank_index[0][0],x_index:x_index+window_length])
        #print(f'values:{x_temp[rank_index[0][0],x_index:x_index+window_length]}, local_mean: {local_mean}')
        #Replace k-th feature with local mean
        x_temp[rank_index[0][0],x_index:x_index+window_length] = local_mean
        #Append x_temp to x_replaced
        x_replaced = x_temp
            
    elif replacement_strategy == 'global_mean':
        x_copy = x_ground_truth.copy()
        #replace k-th feature
        #Get index of k-th feature
        rank_index = np.where(ranks==k)
        x_index = window_length * rank_index[1][0]
        #print(f'k: {k}, x_index:{x_index}, rank_index:{rank_index[0][0]}')
        #Calculate local mean
        global_mean = np.mean(x_copy[rank_index[0][0],:])
        #print(f'values:{x_temp[rank_index[0][0],x_index:x_index+window_length]}, global_mean: {global_mean}')
        #Replace k-th feature with local mean
        x_temp[rank_index[0][0],x_index:x_index+window_length] = global_mean
        #Append x_temp to x_replaced
        x_replaced = x_temp.copy()
    
    elif replacement_strategy == 'zeros':
        #replace k-th
        rank_index = np.where(ranks==k)
        x_index = window_length * rank_index[1][0]
        #print(f'k: {k}, x_index:{x_index}, rank_index:{rank_index[0][0]}')
        #Replace k-th feature with zeros
        x_temp[rank_index[0][0],x_index:x_index+window_length] = 0
        #Append x_temp to x_replaced
        x_replaced = x_temp
    
    return x_replaced
    
def calculate_APT(x, IG, alpha, pattern, window_length, replacement_strategy, model, mode):
    IG_sum_atts, IG_ranks = rank_attributions(IG, pattern, window_length)
    matrix_shape = np.shape(IG_sum_atts)
    x_temp = x.copy()
    x_temp = np.squeeze(np.array(x_temp))
    x_replaced = x_temp.copy()
    J = np.size(IG_ranks)
    k = 0
    condition = False
    if mode == 'SBP':
        #Calculate Ground Truth Value
        pred = model.predict(x, verbose=0)
        SBP_true = pred[0][0]
        #Calculate specific thresholds
        SBP_Up_lim = SBP_true + (alpha*SBP_true)
        SBP_Lo_lim = SBP_true - (alpha*SBP_true)
        while condition == False:
            x_replaced = replace_feature(x_replaced, x_temp, IG_ranks, k, window_length, replacement_strategy)
            #Make tf Tensor from numpy array
            x_replaced_tensor = [tf.cast(np.expand_dims(x_replaced[j],axis=0), tf.float32) for j in range(0,matrix_shape[0])]
            pred_replaced = model.predict(x_replaced_tensor, verbose=0)
            SBP_pred = pred_replaced[0][0]
            if SBP_pred > SBP_Up_lim:
                condition = True
            elif SBP_pred < SBP_Lo_lim:
                condition = True
            else:
                k = k+1
    elif mode == 'DBP':
        pred = model.predict(x, verbose=0)
        DBP_true = pred[0][1]
        #Calculate specific thresholds
        DBP_Up_lim = DBP_true + (alpha*DBP_true)
        DBP_Lo_lim = DBP_true - (alpha*DBP_true)
        while condition == False:
            #Make tf tensor from numpy array
            x_replaced_tensor = [tf.cast(np.expand_dims(x_replaced[j],axis=0), tf.float32) for j in range(0,matrix_shape[0])]
            x_replaced = replace_feature(x_replaced, x_temp, IG_ranks, k, window_length, replacement_strategy)
            pred_replaced = model.predict(x_replaced_tensor, verbose=0)
            DBP_pred = pred_replaced[0][0]
            if DBP_pred > DBP_Up_lim:
                condition = True
            elif DBP_pred < DBP_Lo_lim:
                condition = True
            else:
                k = k+1
    elif mode == 'ABP':
        pass
    #because k starts at zero
    k = k+1
    #calculate APT
    APT = k/J
    
    return APT, k