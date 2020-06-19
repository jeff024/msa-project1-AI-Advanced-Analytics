# Email Spam Classifier

## Table of contents
* [Abstract](#abstarct)
* [Dependencies](#Dependencies)
* [DataSet](#DataSet)
* [HOW TO RUN](#HOWTORUN)
* [Step By Step Walkthrough](#walkthrough)
* [Result](#result)

### Abstract
The main focus of this project revolves around the constant issue of modern spam emails and the countermeasure study of spam mail identification. In this project, I built a spam detector using 6 Machine Learning models and evaluate them with test data. The dataset is downloaded from UCI repository. As per my analysis, SVM model using linear kernel and Neural network worked well for spam classification, whereas others performed not as good.

### Dependencies
- Python == '3.6'
- Pandas == '1.0.3'
- Numpy == '1.18.1'
- sklearn == '0.23.1'
- keras == '2.3.1'
- tensorflow == '1.14.0'
- matplotlib == '3.1.3'

#### How to set up
- Install Python 3.6: https://www.python.org/downloads/release/python-360/
- All other dependencies can be downloaded by `pip3 install something`

### DataSet
This project used the Dataset downloaded from "https://archive.ics.uci.edu/ml/datasets/Spambase" which refers to sample Project 3 in MSA project AI & Advanced Analysis (A ML model to identify emails are spam or not)

### How To Run
- Please make sure all the required dependencies are installed on your machine before running
- Just simply type `python3 SpamEmail.py` in terminal and you can get this model up and running !

### Step By Step Walkthrough
1. Load dataset from "dataset/spambase.data"
2. Ckecking for any missing values and fix the dataset if any. (No missing value found in this dataset)
3. Preprocessing data befor modelling
4. Start trainging and testing following models:
    * SVM-Linear
    * SVM-polynomial
    * SVM-sigmoid
    * Multinomial Naive-Bayes model
    * Gaussian Naive-Bayes model
5. Preprocess the data again for the usage of Nerual Network
6. Constructing Nerual Network and fitting the model
7. Evaluate this model against test set
8. Put all the test result in a bar chart for comparising performance


### Result
<img src = "https://github.com/jeff024/msa-project1-AI-Advanced-Analytics/blob/master/image/accuracy.jpg" width = "500">
It is clear from the comparison that SVM model using linear kernel and Neural network both work pretty well. Since Neural network gets the highest accuracy, I would choose this model as my final choice.