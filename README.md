# AnomalyThresholds
This is the code for the paper ["Empirical Thresholding on Spatio-temporal Autoencoders Trained on Surveillance Videos in a Dementia Care Unit".](https://www.researchgate.net/profile/Shehroz-Khan-3/publication/370068564_Empirical_Thresholding_on_Spatio-temporal_Autoencoders_Trained_on_Surveillance_Videos_in_a_Dementia_Care_Unit/links/643dcf04e881690c4bdec548/Empirical-Thresholding-on-Spatio-temporal-Autoencoders-Trained-on-Surveillance-Videos-in-a-Dementia-Care-Unit.pdf)

# Environment Setup
First please create an appropriate environment using conda: 

> conda env create -f conda_torch.yml

> conda activate torch

# Data
Due to ethical considerations, the data used in the paper cannot be made publicly available. Here, we have provided a dummy data (sample_data.zip) for the purpose of running the code.

# Running the models
To run the model CAE_2DConv or CAE_3DConv, follow the below template:

> python <model_python_script> --num_workers <number_of_num_workers> --gpus <GPU_option> --max_epochs <number_of_epochs> --train_batch_size <batch_size_for_training> --train_file_path <Path_to_folder_containing_training_video_frames> --test_file_path <Path_to_folder_containing_test_video_frames> --label_file_path <Path_to_HDF5_file_containing_test_labels>

GPU_option: set 0 to use CPU and -1 to use all available GPUs

Example:

> python CAE_3DConv.py --num_workers 5 --gpus -1 --max_epochs 51 --train_batch_size 5 --train_file_path sample_data --test_file_path sample_data --label_file_path sample_labels.hdf5

# Train Model
To train a model, set settng='train' in the CAE_2DConv.py or CAE_3DConv.py script and run the above command.

# Evaluate Model
To evaluate/test a model, set settng='test' in the CAE_2DConv.py or CAE_3DConv.py script and run the above command.
