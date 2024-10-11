#Import the classes for processing of PulseDB
from PreprocessingPulseDB import PreprocessingPulseDB
from Preprocess_Pressure import PreprocessingPulseDB_Pressure

#Define Paths for Preprocessing
#target_path = "C:/Users/vogelsto/Desktop/Studienarbeit/pulse_db_norm/"
target_path = "C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/Data/"
#data_path = "Q:/2023_09_07_PulseDB/measurements/PulseDB_Vital/"
data_path = "C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Pulse DB Database/PulseDB_Vital/"


#%% Preprocess PulseDB data for VitalDB
'''
preprocessor = PreprocessingPulseDB(data_path, target_path, db='v', replace=True)
preprocessor.process()
'''
#%% Preprocess PulseDB data for MIMIC3
'''
# Preprocess MIMIC3
data_path = "Q:/2023_09_07_PulseDB/measurements/PulseDB_MIMIC/"
preprocessor = PreprocessingPulseDB(data_path, target_path, db='m', replace=True)
preprocessor.process()
'''

#%% Preprocess Pressure Data of VitalDB
preprocessor_pressure = PreprocessingPulseDB_Pressure(data_path, target_path, db='v', replace=True)
preprocessor_pressure.process()