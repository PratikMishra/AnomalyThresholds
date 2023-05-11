# Returns indexes for I_t, I_v, O_t, O_v sets
import math, numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

def setSplits(trainScores, whisker=1.7239):
    # trainScores: numpy array of reconstruction errors calculated on initial or full training set (N)
    # whisker: whisker value for IQR analysis used to split initial or full training set (N) into inliers (I) and outliers (O) sets
    # Returns:
    #   index_inliers_train: Indexes corresponding to set N for samples belonging to inliers training set I_t
    #   index_inliers_val: Indexes corresponding to set N for samples belonging to inliers validation set I_v
    #   index_outliers_train: Indexes corresponding to set N for samples belonging to outliers training set O_t
    #   index_outliers_val: Indexes corresponding to set N for samples belonging to outliers validation set O_v
    print('Using IQR analysis with whisker: ', whisker)
    fig1,ax1 = plt.subplots()
    bp = ax1.boxplot(trainScores ,whis = whisker)
    # plt.show()
    num_outliers = len(bp['fliers'][0].get_ydata())
    q1 = bp['boxes'][0].get_ydata()[0]
    q3 = bp['boxes'][0].get_ydata()[2]
    q2 = bp['medians'][0].get_ydata()[0]

    #Dividing into inliers and outliers using IQR analysis
    cutoff = (q3-q1)*whisker
    tmp = np.reshape(np.asarray(list(range(trainScores.shape[0]))),(-1,1))
    inliers = np.reshape(tmp[trainScores<=(q3+cutoff)],(-1,1))
    outliers = np.reshape(tmp[trainScores>(q3+cutoff)],(-1,1))
    print('No. of outlier samples: ', num_outliers)

    #Randomizing the index sets
    np.random.shuffle(inliers); np.random.shuffle(outliers)

    #Doing 90-10 train-validation split for inliers and outliers set
    num_inliers_val = math.ceil(inliers.shape[0]*0.1); num_inliers_train = inliers.shape[0]-num_inliers_val
    index_inliers_train = inliers[0:num_inliers_train,:]
    index_inliers_val = inliers[num_inliers_train:,:]

    num_outliers_val = math.ceil(outliers.shape[0]*0.1); num_outliers_train = outliers.shape[0]-num_outliers_val
    index_outliers_train = outliers[0:num_outliers_train,:]
    index_outliers_val = outliers[num_outliers_train:,:]

    return index_inliers_train, index_inliers_val, index_outliers_train, index_outliers_val