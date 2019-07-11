# Model Performance Evaluation  

This section looks at techniques for evaluating model performance.  

In order to measure and quantify performance of a model, we need to decide on how to compare predicted versus actual values.   
The answer for this is not straight forward.  

The reason is:  
Regression algorithm produce continuous numeric output  
Binary Classification Algorithm produce Yes/No, Pass/Fail type binary answers  
Multi-class classification algorithms produce one of many possible known outcomes.  

So, we need to use appropriate methods based on the type of data predicted: continuous, binary and categorical  

Provided Sample Notebooks and Data sets compare performance of four different models and rank them.  

For Regression, we use Plots, Residual Histograms and RMSE Metric   
For Binary Classification, we use Plots, Confusion Matrix, and Metrics like Precision, Recall, Accuracy, F1 Score, AUC Score and so forth  
For Multiclass Classification, we use Plots, Confusion Matrix, Different ways to average class level metrics into model level metrics  

