# 2023-HGR5-CNN_LSTM
Code of the paper
***Assessing the influence of LSTM and post-processing in CNN based Hand Gesture Recognition using EMG***
developed in Matlab. 


# Instructions
All these instructions are identical for the CNN or the CNN-LSTM model. 
To choose the desired model run the corresponding script from either the folder CNN/ or CNN-LSTM/. 
To change between doing or not doing post-processing, change the corresponding flag in the **Shared.m** script.

## Configurations
1. Most of the models hyper-parameter can be configured in the **Shared.m** script.
1. Neural network architecture and trianing parameters are defined in the **modelCreation.m** script.

## Requirements
1. Download the [EMG-EPN-612 dataset](https://laboratorio-ia.epn.edu.ec/es/recursos/dataset/2020_emg_dataset_612) and paste it in **EMG-EPN-612 dataset/**. 


## Training
1. Create the datastores running the script: **spectrogramDatasetGeneration.m**. 
1. Train the model running the script: **modelCreation.m**.
* Trained models are saved by date in the folder **Models/** or **ModelsLSTM/**.
1. Evaluate training and validation recognition accuracy running the script: **modelEvaluation.m**.

## Testing
1. Evaluate on the testing subset of the EMG-EPN-612 dataset by running the script: **testDataEvaluation.m**.
* To change the model to be evaluated, change the variable *modelFileName* in the corresponding script.
* The script will generate a **responses.json** file with the predictions in the folder "*model*/Test-Data/".
1. If desired, submit the **responses.json** file to the public online evaluator. 



<!-- Execution -->

# Abstract

In the field of Hand Gesture Recognition (HGR) using EMG signals, creating a model that can be widely applied to new subjects without the need for additional training has proven to be a significant challenge. 
This is due to the variability in the EMG signals across individuals, which makes it difficult to develop a model that generalizes well to new subjects. 
This study aims to address this challenge by examining the effect of incorporating a post-processing algorithm on the performance of a HGR model based on spectrograms and CNNs. 
The study also compares the combination of CNNs and CNN-LSTM to assess the influence of the memory cells on the model. 
The public EMG-EPN-612 dataset was used for training and testing. 
When using post-processing, the CNN model achieved a recognition accuracy of 87.26% ± 11.14%, while the CNN-LSTM achieved 90.55% ± 9.45%. 
The inclusion of the memory cells increased accuracy by 3.29%, but at the cost of 53 times more learnables. 
For its part, the post-processing algorithm increased accuracy by 41.86% for the CNN model (from 45.40% ± 15.51%) and 24.77% in the case of the CNN-LSTM model (from 65.78% ± 15.51%). 
These experimental results showed that the post-processing algorithm had a greater impact in recognition than the usage of memory cells. 
These findings suggest new paths for research in HGR architectures beyond the traditional focus on the classification and feature extraction stages. 
For reproducibility purposes, we made publicly available the source code in Github. 




# Authors
Lorena Isabel Barona López, Francis M. Ferri, Jonathan Zea, Ángel Leonardo Valdivieso
Caraguay and Marco E. Benalcázar


# Reference
Code of the paper: 

"Assessing the influence of LSTM and post-processing in CNN based Hand Gesture Recognition using EMG"


