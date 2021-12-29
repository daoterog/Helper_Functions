# Anomaly Detection

This folder contains modules designed to work with anomaly detection problems. The functions
are designed in order to evaluate the models in a supervised way, this makes necessary to count
with the real labels of the data. Despite this, the implemented frameworks were built to improve 
unsupervised models performance, since this is the most used type of learning to tackle this 
problems. 

## Contents

- 'scikit_feature_master': library developed by the Data Mining and Machine Learning Lab 
at Arizona State University. Contains over 40 popular feature selection algorithms, the SPEC
algorithm is specifically useful for this use cases. 
- 'analysis.py': functions that allow to visualize the observations categorized as outliers
in a 2D plane (through PCA), and allow to see the percentage of frozen transactions in a 
fraud detection use case.
- 'feature_engineering.py': most of the functions contained in this folder were built to 
work with a fraud detection problem. The used dataset had only 1 numerical variable and 8
categorical ones. This suposed a problem since solely one-hot encoding variables may not 
provide models with the most relevant information. The function developed in here try to 
uniquely characterize transactions by assigning a deviation number to each specified subgroup.
- 'frameworks.py': contains a set of functions that allow to implement several types of 
frameworks. 'cross_validation_framework' is the main function and it allows to configure certain
hyperparameters of the build framework, the variety of frameworks provided are an implementation
of several papers found in literature, in addition, to a framework developed by me, by adding a
multiple layer filtering process to a mix of frameworks.  'cross_validation_bagging' corresponds 
to the bagging routine build for the problem. The thought process and testing put behind these 
functions is exhibited in the article contained in the 'FinancialFraud_AnomalyDetection' repository
pinned in my profile.