# Email Spam Classifier
 
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

### DataSet
This project used the Dataset downloaded from "https://archive.ics.uci.edu/ml/datasets/Spambase" which refers to sample Project 3 in MSA project AI & Advanced Analysis (A ML model to identify emails are spam or not)

### How To Run
- Please make sure all the required dependencies are installed on your machine before running
- Just simply type `python3 SpamEmail.py` in terminal and you can get this model up and running !

### Result
<img src = "https://github.com/jeff024/msa-project1-AI-Advanced-Analytics/blob/master/image/accuracy.jpg" width = "500">
It is clear from the comparison that SVM model using linear kernel and Neural network both work pretty well. Since Neural network gets the highest accuracy, I would choose this model as my final choice.