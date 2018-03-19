# Determining reliability of IT (Inferior Temporal) neural outputs predicted using a given covariate matrix.

This is a walkthrough of the code implementations which can be used to determine reliability of predicted IT(Inferior Temporal) neural outputs. Description of the terminologies are available in Readme.md file located in the previous directory.

Reliability is measured in terms of explained variance(%) between predicted IT matrix vectors and actual IT matrix vectors. The closer the explained variance is to 100%, the more similar/reliable the predicted IT matrix vectors are. It is calculated using this formula, where the predicted vector is **f** and the actual vector is **y** :
<p align="center">
  <img src="https://github.com/Tapojit/Predicting-IT-cortex-output-using-DNN-and-analysis-MIT-/blob/master/img/eqn5.PNG">
</p>
where:
<p align="center">
  <img src="https://github.com/Tapojit/Predicting-IT-cortex-output-using-DNN-and-analysis-MIT-/blob/master/img/eqn6.PNG">
</p>
This metric of determining reliability is used in Cadieu et al.(2)


The implementations are in a modular form, that is, multiple functions used for loss calculation and crossvalidation are in object form, so that anyone can create their own loss/crossvalidation implementation and pass them as arguments. Loss function objects here end with *_LM* filename, whereas regression crossvalidation object ends with *_RCV* filename. Only one regression crossvalidation object is here, whose usage will be shown for the purpose of this experiment.

For this example, V4 multiunit matrix will be used as covariate matrix.

```
%Getting and unzipping data file
file = websave('data','https://s3.amazonaws.com/cadieu-etal-ploscb2014/PLoSCB2014_data_20141216.zip');
unzip('data.zip');

%Loading V4 data matrix
covariate_matrix = load('PLoSCB2014_data_20141216/NeuralData_V4_multiunits.mat');
covariate_matrix = covariate_matrix.features;

%Loading IT data matrix
response_matrix = load('PLoSCB2014_data_20141216/NeuralData_IT_multiunits.mat');
response_matrix = response_matrix.features;
```
The V4 and IT matrices are passed as arguments to the Cadieu_RCV object.

```
%The final argument here is the filename of .mat file where summary
%results will be stored
results = Cadieu_RCV(covariate_matrix, response_matrix, 'IT & V4');
```
There are more arguments for Cadieu_RCV, but the first three are compulsory. Either three or all arguments need to be entered. If only the first three are entered, default values for the the other arguments are used.

In this case, summary results such as mean explained variance for the entire predicted IT matrix, explained variance for individual channels/columns of predicted IT matrix will be stored in a .mat file with filename 'IT & V4'.

It is recommended to look at the comments in the scripts, which can be used as guides to create other objects to obtain other forms of statistical analyses from this kind of data.

##  References
1.  “1.5 - The Coefficient of Determination, r-Squared | STAT 501,” https://onlinecourses.science.psu.edu/stat501/node/255.

2.  Cadieu CF, Hong H, Yamins DLK, Pinto N, Ardila D, Solomon EA, et al. (2014) Deep Neural Networks Rival the Representation of Primate IT Cortex for Core Visual Object Recognition. PLoS Comput Biol 10(12): e1003963. https://doi.org/10.1371/journal.pcbi.1003963
