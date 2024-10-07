from PreprocessingPulseDB import PreprocessingPulseDB


#target_path = "C:/Users/vogelsto/Desktop/Studienarbeit/pulse_db_norm/"
target_path = "C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Code/Data/"
# Preprocess VitalDB
#data_path = "Q:/2023_09_07_PulseDB/measurements/PulseDB_Vital/"
data_path = "C:/Biomedizinische Informationstechnik/2. Semester/Projektarbeit/Pulse DB Database/PulseDB_Vital/"
preprocessor = PreprocessingPulseDB(data_path, target_path, db='v', replace=True)
preprocessor.process()





#%%
'''
# Preprocess MIMIC3
data_path = "Q:/2023_09_07_PulseDB/measurements/PulseDB_MIMIC/"
preprocessor = PreprocessingPulseDB(data_path, target_path, db='m', replace=True)
preprocessor.process()
'''


