# predicting_death_from_heart_disease_R

## Overview 

In this project, I will use machine learning (ML) to predict patient's survival following a heart failure, using patients' available electronic medical records. In order to try to reproduce the results reported in a paper called **"Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone**" (https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5), I will also rank the features corresponding to the most important risk factors, as determined by the ML algorithms. 

I will build  and tune 8 different models:

1. k-Nearest Neighbors (KNN)
2. Random Forest (RF)
3. Adaptive Boosting (AdaBoost)
4. Linear Discriminant Analysis (LDA)
5. Support Vector Machine (SVM) - Linear 
6. Support Vector Machine (SVM) - Radial
7. Generalized linear model via penalized maximum likelihood (GLMNET)
8. Multivariate adaptive regression spline (MARS)


## Dataset

The dataset here is [Heart failure clinical records Data Set]("https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records") taken from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php"). The dataset can also be found on Kaggle (https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/notebooks). 

## File structure
There are three files: 

1. Report in .Rmd format
2. Report in .pdf format knit from the .Rmd file
3. R script in .R format that generates predicted outcomes and their evaluations


This project, [HarvardX: PH125.9x, Data Science: Capstone](https://www.edx.org/course/data-science-capstone), is a part of the [Professional Certificate in Data Science](https://www.edx.org/professional-certificate/harvardx-data-science) course led by HarvardX. This program was supported in part by NIH grant R25GM114818.
