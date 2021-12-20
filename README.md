# Estimation-of-Daily-Bicycle-Traffic-Using-Machine-and-Deep-Learning-Techniques

## Authors
* MD MINTU MIAH, MS<sup>1</sup>
* KATE KYUNG HYUN, PhD<sup>2</sup>
* STEPHEN P MATTINGLY, PhD<sup>3</sup>
* HANNAN KHAN, BS<sup>4</sup>

> <sup>1</sup>Doctoral Research Assistant, Department of Civil Engineering, the University of Texas at Arlington, TX 76019, USA, Corresponding Email: [mdmintu.miah@mavs.uta.edu](mdmintu.miah@mavs.uta.edu)
>
> <sup>2</sup>Assistant Professor, Department of Civil Engineering, the University of Texas at Arlington, TX 76019, USA, Email: [kate.hyun@uta.edu](kate.hyun@uta.edu)
>
> <sup>3</sup>Professor, Department of Civil Engineering, the University of Texas at Arlington, TX 76019, USA, Email: [mattingly@uta.edu](mattingly@uta.edu)
>
> <sup>4</sup>Graduate Student, Department of Computer Science Engineering, the University of Texas at Arlington, TX 76019,USA, Email: [hannan.khan@mavs.uta.edu](hannan.khan@mavs.uta.edu)

## Abstract
Machine learning (ML) architecture has successfully characterized complex motorized
volumes and travel patterns; however, non-motorized traffic has given less attention to
ML techniques and relied on simple econometric models due to lack of data for
complex modeling. Recent advancement of smartphone-based location data that
collect and process large amounts of daily bicycle activities makes the use of machine
learning techniques for bicycle volume estimations possible and promising. This study
develops seven modeling techniques ranging from advanced techniques such as Deep
Neural Network (DNN), Shallow Neural Network (SNN), Random Forest (RF),
XGBoost, to conventional and simpler approaches such as Decision Tree (DT),
Negative Binomial (NB), and Multiple Linear Regression to estimate the Daily Bicycle
Traffic (DBT). This study uses 6,746 daily bicycle volumes collected from 178
permanent and short-term count locations from 2017 to 2019 in Portland, Oregon. A
total of 45 independent variables capturing anonymous bicycle user activities (Strava
count, bike share), built environments, motorized traffic, and sociodemographic
characteristics create comprehensive variable sets for predictive modeling. Two
variable dimension reduction techniques using principal component analysis and
random forest variable importance analysis ensure that the models are not overgeneralized
or over-fitted with a large variable set. The comparative analysis between
models shows that machine learning techniques of SNN and DNN produce higher
accuracies in estimating daily bicycle volumes. Results show that the DNN models
predict the DBT with a maximum mean absolute percentage error (APE) of 22% while
the conventional model (liner regression) shows 45% of APE.

![Graphical Abstract](/images/graphical_abstract.png)

## Requirements

To install requirements:
It is best to create an Anaconda environment and execute the following command:

```setup
pip install -r requirements.txt
```

The data is available in 'data/'  
_**NOTE:**_ The data does not include the feature 'strava_count' as that is proprietary data. In order to run this code, that column is necessary.

## Project Structure

This project is split into multiple .ipynb files.  
Each file corresponds with one or more models that we tested on our data. (For example: the 'DNN_Route.ipynb' file contains the deep neural network models we created/evaluated with dimension-reduced data, normal data, scaled data, ...)  
Within each hyperparameter search, we have included the 'Wall Time' which is the time it took for our computer to run that particular piece of code.



## Pre-trained Models

The pretrained models are available in the appropriate folder.  
(For example: the CNN models are available in the 'models/' folder under the specifc model name I want. In the following code snippet I chose the CNN model based on 45 scaled variables and one convolutional layer.)

I can load this model in two different ways:
1. To obtain the model architecture, we must load the approprate '......study.pkl' file using:
```
study = joblib.load("cnn_trials/cnn_45_scaled_one_conv_lyr_study.pkl")
print("Best trial until now:")
print(" Value: ", study.best_trial.value)
print(" Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")
======================================================================
Best trial until now:
 Value:  158.00619506835938
 Params: 
    batch_size: 16
    n_hdn_layers: 4
    neurons_HL1: 714
    out_channel: 128
    kernel_size: 5
    conv_activation: linear
    dropout_prob: 0.1
    mx_pl_size: 2
    mx_pl_strides: 3
    HL0_ac_fn: relu
    HL1_ac_fn: linear
    HL2_ac_fn: relu
    HL3_ac_fn: linear
```
The model can then be created using either Keras or PyTorch:  
_**Note:**_ In each model, the output layer has 1 node with a linear activation function.
```
cnn_model = Sequential([
    layers.Conv1D(128, 5, activation='linear', input_shape=(X_scaled.shape[1], 1)),
    layers.Dropout(0.1),
    layers.MaxPooling1D(pool_size=2, strides=3, padding='valid'),
    layers.Flatten(),
    layers.Dense(714, activation='relu'),
    layers.Dense(357, activation='linear'),
    layers.Dense(178, activation='relu'),
    layers.Dense(89, activation='linear'),
    layers.Dense(1, activation='linear')
])
```
2. The PyTorch model can be directly loaded by selecting the '......trial#.pickle' file:
```
with open('models/cnn_45_scaled_one_conv_lyr_trial3271.pickle', 'rb') as f:
    torch.load(f)

 # Or you can use:

 cnn_model = torch.load('cnn_45_scaled_one_conv_lyr_trial3271.pickle')
```

_**NOTE:**_ The CNN and DNN saved models were made using GPU, loading and use of these models will likely require a GPU as well.
