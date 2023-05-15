# AnomalyThresholds
This is the code for the paper ["Empirical Thresholding on Spatio-temporal Autoencoders Trained on Surveillance Videos in a Dementia Care Unit"](https://www.researchgate.net/profile/Shehroz-Khan-3/publication/370068564_Empirical_Thresholding_on_Spatio-temporal_Autoencoders_Trained_on_Surveillance_Videos_in_a_Dementia_Care_Unit/links/643dcf04e881690c4bdec548/Empirical-Thresholding-on-Spatio-temporal-Autoencoders-Trained-on-Surveillance-Videos-in-a-Dementia-Care-Unit.pdf). The paper has been accepted at 20th Conference on Robots and Vision. The code gives different thresholds to choose from in anomaly detection problems in the absence of a validation set.

# Data
Due to ethical considerations, the data used in the paper cannot be made publicly available. Here, we have provided a dummy reconstruction error files in the data folder for the purpose of running the code.

# Creating Validation Set from Normal Data
![Creating Validation Set from Normal Data](https://github.com/PratikMishra/AnomalyThresholds/blob/main/cross-validation.jpg)
We first create two sets from the reconstruction error of full training samples (N) – Inliers (I) and Outliers (O), using IQR analysis with Ω = 1.5. In the absence of samples from anomalous class, O can serve as a proxy for unseen anomalous events. Then, I and O are further divided with 90%-10% split into: training (I<sub>t</sub>, O<sub>t</sub>) and validation (I<sub>v</sub>, O<sub>v</sub>) sets (see above figure).

# Process flow
We train an anomaly detection model (a 3D Convolutional Autoencoder in paper) on the full normal data (N), and obtain reconstruction error on all the training samples. Then, we can apply the following thresholding methods:
- MaxRE: maximum value of reconstruction error on all the training samples is considered as a threshold.
- StdRE: The threshold is set as 3 standard deviations (σ) away from the mean of the reconstruction error.
- RedMaxRE: To tighten the above thresholds, we perform IQR analysis on the reconstruction error of the training samples, and extreme reconstruction values are removed based on a rejection rate, ω. The maximum of the remaining reconstruction
error values is considered as a threshold. Please refer to the paper for more details.

The above threshold mechanisms don't need a proxy validation set. However, the below threshold approaches need a proxy validation set and were observed to perform better than the previous approaches. For the below approaches, first we train an anomaly detection model on I<sub>t</sub>, and obtain reconstruction error on all the samples in I<sub>t</sub>.
- OptRE: We choose each reconstruction error as a threshold and find the corresponding gmean using I<sub>v</sub> and O<sub>v</sub>. The reconstruction error with the highest gmean is chosen as the threshold.
- OptROC: We make a receiver operating characteristic curve using I<sub>v</sub> and O<sub>v</sub>. This will give various true positive rates and false positive rates along with corresponding thresholds on the proxy validation set (by using ["roc curve"](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) function). The corresponding threshold value that gives the highest gmean is chosen as the threshold.
- OptPR: We make a precision-recall curve using I<sub>v</sub> and O<sub>v</sub>. This will give various precision and recall values along with corresponding thresholds on the proxy validation set (by using ["precision recall curve"](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) function). The corresponding threshold value that gives the highest gmean is chosen as the threshold.

During testing, any sample with a reconstruction error greater than the threshold will be considered an anomalous event.

# Usage of scripts
Below is the description of different scripts:
- setSplits.py: Returns indexes for I<sub>t</sub>, I<sub>v</sub>, O<sub>t</sub>, O<sub>v</sub> sets. <br />
- thresholds.py: Calculates threshold using one of many different thresholding approaches of choice. <br />
- sample.py: Contains the sample code on how to use setSplits.py and thresholds.py scripts. <br />
