Результаты обучения:
Boosting, Decision Tree и Random Forest имеют лучшие показатели, однако есть риск переобучения.


Naive_Bayes

alpha             0.000010
Train Accuracy    0.978980
Test Accuracy     0.950668
Test Recall       0.930314
Test Precision    0.964486


Boosting

		precision    recall  f1-score   support

           0       1.00      0.99      0.99      4718
           1       0.99      1.00      0.99      4262

    accuracy                           0.99      8980
   macro avg       0.99      0.99      0.99      8980
weighted avg       0.99      0.99      0.99      8980


Decision Tree

		precision    recall  f1-score   support

           0       0.99      1.00      1.00      4718
           1       1.00      0.99      0.99      4262

    accuracy                           1.00      8980
   macro avg       1.00      1.00      1.00      8980
weighted avg       1.00      1.00      1.00      8980


Random Forest

		precision    recall  f1-score   support

           0       0.99      0.99      0.99      4718
           1       0.99      0.99      0.99      4262

    accuracy                           0.99      8980
   macro avg       0.99      0.99      0.99      8980
weighted avg       0.99      0.99      0.99      8980