import h5py, numpy as np
from thresholds import thresholds
from setSplits import setSplits

#Sample on how to get indexes for I_t, I_v, O_t, O_v using setSplits function
ff = h5py.File('data/trainReconstructionErrors_N.hdf5','r')
trainScores = np.array(ff['loss'])
ff.close()
index_inliers_train, index_inliers_val, index_outliers_train, index_outliers_val = setSplits(trainScores)

#Sample on how to determine threshold using different thresholding approaches
ff = h5py.File('data/trainReconstructionErrors_N.hdf5','r')
trainScores = np.array(ff['loss'])
ff.close()
print(thresholds(trainError=trainScores, mode='MaxRE'))
print(thresholds(trainError=trainScores, mode='StdRE'))
print(thresholds(trainError=trainScores, mode='RedMaxRE'))

f = h5py.File('data/trainReconstructionErrors_I_t.hdf5','r')
trainloss = np.array(f['loss'])
f.close()
f = h5py.File('data/valReconstructionErrors_I_v-O_v.hdf5','r')
valloss = np.array(f['loss'])
valLabels = np.array(f['labels'])
f.close()
print(thresholds(trainError=trainloss, valError=valloss, valLabels=valLabels, mode='OptRE'))
print(thresholds(trainError=trainloss, valError=valloss, valLabels=valLabels, mode='OptROC'))
print(thresholds(trainError=trainloss, valError=valloss, valLabels=valLabels, mode='OptPR'))
