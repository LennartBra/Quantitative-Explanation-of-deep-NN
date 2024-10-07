#---------------------------------------------------------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------------------------------------------------------
import numpy as np
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

from DataGenerator.DataGenerator_template import DataGenerator
from models.make_model_template import make_model

import tensorflow
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from keras import optimizers
from keras.models import clone_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow import keras

keras.backend.clear_session()

if tensorflow.test.gpu_device_name():
    print("Default GPU Device: {}".format(tensorflow.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

#---------------------------------------------------------------------------------------------------------------------
# Initialize paths
#---------------------------------------------------------------------------------------------------------------------

# Main path of final preprocessed data
path_main = "PulseDB/pulsedb0/"
# IDs
files = os.listdir(path_main+"dev0/")
### Necessary for using subset ###
#files = files[:10]
##################################


#---------------------------------------------------------------------------------------------------------------------
# Initiliaze trainingsparameter
#---------------------------------------------------------------------------------------------------------------------
# loo = LeaveOneOut()
# loo.get_n_splits(files)

n_splits = 3
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
kfold.get_n_splits(files)

batch_size = 2048

if __name__=="__main__":
    print("Train TemplateNet with PulseDB")
    all_mae_sbp, all_mae_dbp, subject_result, all_pred, all_r_sbp, all_r_dbp = [], [], [], [], [], []
    for nr_fold, (train_index, test_index) in enumerate(kfold.split(files)):
        if nr_fold in [1,2]:continue

        # Separate training, validation and test ids
        train_index, val_index = train_test_split(train_index, test_size=0.1)
        train_id = [files[x] for x in train_index]
        val_id = [files[x] for x in val_index]
        test_id = [files[x] for x in test_index]

        # Generators
        print("Loading Datagenerator")
        generator_train = DataGenerator(path_main, train_id, batch_size=batch_size)
        generator_val = DataGenerator(path_main, val_id, batch_size=batch_size)
        generator_test = DataGenerator(path_main, test_id, batch_size=batch_size, shuffle=False)


        model = make_model()

        # Make training
        optimizer = optimizers.Adam(learning_rate=0.0001)

        es = EarlyStopping(monitor="mae", patience=10)
        mcp = ModelCheckpoint('best_model_template'+str(nr_fold)+'.h5', monitor='val_mae', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-8)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        model.fit(generator_train,
                        validation_data=generator_val,
                        epochs=50,
                        verbose=1,
                        callbacks=[es, mcp, reduce_lr])

        # Make prediction
        nr_data = generator_test.__len__()
        all_mae = np.zeros((nr_data,2))

        print('Prediction')
        for batch_index in range(nr_data):
            batch_data, temp_true = generator_test.__getitem__(batch_index)

            temp_pred = model.predict(batch_data, verbose=0, batch_size=batch_size)
            if batch_index==0:
                data_pred = temp_pred
                data_true = temp_true
                continue

            if batch_index==nr_data-1:
                for i in range(len(temp_pred)):
                    if temp_pred[i,0]==0:
                        temp_pred = temp_pred[i-1] 
                        temp_true = temp_true[i-1]
                        
            data_pred = np.concatenate((data_pred, temp_pred), axis=0)
            data_true = np.concatenate((data_true, temp_true), axis=0)
        
            mae_sbp_batch = mean_absolute_error(temp_pred[...,0], temp_true[:,0])
            mae_dbp_batch = mean_absolute_error(temp_pred[...,1], temp_true[:,1])

            all_mae[batch_index] = np.array([mae_sbp_batch, mae_dbp_batch])
        r_sbp, _ = pearsonr(data_pred[:,0], data_true[:,0]) 
        r_dbp, _ = pearsonr(data_pred[:,1], data_true[:,1]) 
        
        mae_sbp = np.mean(all_mae[:,0])
        mae_dbp = np.mean(all_mae[:,1])
        all_pred.append(np.array(data_pred))
        
        print(mae_sbp)
        print(mae_dbp)
        print(r_sbp, r_dbp)

        all_mae_sbp.append(mae_sbp)
        all_mae_dbp.append(mae_dbp)
        all_r_sbp.append(r_sbp)
        all_r_dbp.append(r_dbp)
        
    mae_sbp_mean = np.mean(all_mae_sbp)
    mae_dbp_mean = np.mean(all_mae_dbp)
    r_mean_sbp = np.mean(all_r_sbp)
    r_mean_dbp = np.mean(all_r_dbp)
    
    print("Mean of SBP: ", mae_sbp_mean)
    print("Mean of DBP: ", mae_dbp_mean)
    print("Mean of r: ", r_mean_sbp, r_mean_dbp)
    













