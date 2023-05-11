# AnomalyThresholds
This is the code for the paper ["Empirical Thresholding on Spatio-temporal Autoencoders Trained on Surveillance Videos in a Dementia Care Unit".](https://www.researchgate.net/profile/Shehroz-Khan-3/publication/370068564_Empirical_Thresholding_on_Spatio-temporal_Autoencoders_Trained_on_Surveillance_Videos_in_a_Dementia_Care_Unit/links/643dcf04e881690c4bdec548/Empirical-Thresholding-on-Spatio-temporal-Autoencoders-Trained-on-Surveillance-Videos-in-a-Dementia-Care-Unit.pdf). The paper has been accepted at 20th Conference on Robots and Vision.

# Data
Due to ethical considerations, the data used in the paper cannot be made publicly available. Here, we have provided a dummy data (sample_data.zip) for the purpose of running the code.

# Usage of scripts
setSplits.py: Returns indexes for I_t, I_v, O_t, O_v sets. <br />
thresholds.py: Calculates threshold using one of many different thresholding approaches of choice. <br />
sample.py: Contains the sample code on how to use setSplits.py and thresholds.py scripts. <br />

The ["paper".](https://www.researchgate.net/profile/Shehroz-Khan-3/publication/370068564_Empirical_Thresholding_on_Spatio-temporal_Autoencoders_Trained_on_Surveillance_Videos_in_a_Dementia_Care_Unit/links/643dcf04e881690c4bdec548/Empirical-Thresholding-on-Spatio-temporal-Autoencoders-Trained-on-Surveillance-Videos-in-a-Dementia-Care-Unit.pdf) describes the details about the different thresholding approaches for an anomaly detection approach.
