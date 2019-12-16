import numpy as np
import pandas as pd
from dcml import DataClusterBasedMachineLearning

##########################################
# Test code
##########################################
# Step 1. Machine Learning
x_data = np.array([[667., 7], [693.3, 7], [732.9, 6], [658.9, 1], [702.8, 7], [667., 7], [693.3, 7], [732.9, 6], [658.9, 1], [702.8, 7], [667. , 7], [693.3, 7], [732.9, 6], [658.9, 1], [702.8, 7], [697.2, 1], [658.7, 2], [723.1, 1], [719.5, 3], [687.4, 1], [704.1, 1], [658.8, 4], [667.8, 3], [703.4, 3]])
y_data = np.array([[667.], [693.3], [732.9], [658.9], [702.8], [667.], [693.3], [732.9], [658.9], [702.8], [667.], [693.3], [732.9], [658.9], [702.8],[697.2], [658.7], [723.1], [719.5], [687.4], [704.1], [658.8], [667.8], [703.4]])

x_df = pd.DataFrame({'X1': x_data[:, 0], 'X2': x_data[:, 1]})
y_df = pd.DataFrame({'Y': y_data[:, 0]})

clustering_algs = ['GaussianMixture', 'DBSCAN']
prediction_algs = ['RandomForest', 'GradientBoosting']

max_clusters = 3

dc_ml = DataClusterBasedMachineLearning(x_df, y_df, clustering_algs, prediction_algs, max_clusters)
ml_family = dc_ml.run()

##########################################
# Step 2. Prediction
x_test_data = np.array([[661., 7], [696.3, 8]])
y_test_data = np.array([[662.], [699.3]])
x_test_df = pd.DataFrame({'X1': x_test_data[:, 0], 'X2': x_test_data[:, 1]})
y_test_df = pd.DataFrame({'Y': y_test_data[:, 0]})

yPredicted, r2 = dc_ml.perform_prediction(ml_family, x_test_df, y_test_df)

print(y_test_df)
print(f'R2 = {r2}')
