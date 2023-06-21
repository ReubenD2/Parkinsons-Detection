
# Hybrid Model for Detecting Parkinson’s Disease using speech data

A Hybrid Model for Detection of Parkinson’s Disease using speech data. This model makes use of stacking classifiers made from Machine Learning Algorithms such as Support 
Vector Machine, K-nearest neighbor and XGBoost. This also makes use of Synthetic 
oversampling of the minority class and use of a feature selection algorithm while retaining 
the most important ones that contribute to the accuracy of the model. This system is 
deployed through a web application using Streamlit.




## Results

| Full Features K-fold scores   | Recall      | Accuracy | f1-score| precision | roc_auc |
| -------------                 | ----------  | -------- | ------  | -----     | -----   |
| SVM                           | 96.66| 96.63    | 96.62   | 96.97  | 99.49   |          
| KNN                           | 89.47| 89.37    | 89.11   | 91.59  | 98.39   |
| XGBoost                       | 93.75| 93.64    | 93.61   | 94.31  | 98.28       |
| Proposed Method (Stacking classifier)| 97.08| 97.05   | 97.04  | 97.3   | 99.42    |



| Selected Features K-fold scores | Recall    | Accuracy | f1-score| precision | roc_auc |
| -------------                 | ----------  | -------- | ------  | -----     | -----   |
| SVM                           | 93.3 | 93.22    | 93.2  | 93.77  | 98.14   |          
| KNN                           | 92.88| 92.83    | 92.8  | 93.4   | 95.84   |
| XGBoost                       | 93.71| 93.66    | 93.6  | 94.28  | 98.43    |
| Proposed Method (Stacking classifier)| 94.55| 94.5   | 94.46  | 95.08   | 98.36 |



## Team

 
 | Members       |
 | ------------- |
 | Malay Thakkar (Team Leader) |
 | Viraj Landhe  |
 | Neelanjaan De |
 | Bharat Dedhia |


## License

[MIT](https://choosealicense.com/licenses/mit/)

