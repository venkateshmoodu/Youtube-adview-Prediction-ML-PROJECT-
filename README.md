# Youtube-adview-Prediction-ML-PROJECT-

This project aims to predict the adview count of YouTube videos using machine learning. We use a dataset (train.csv) containing roughly 15,000 videos with features such as views, likes, dislikes, comments, published date, duration, category, and the target adview count
github.com
. The objective is to build regression models that estimate a video’s adview count from these metrics
github.com
. We preprocess the data to clean invalid entries, encode categorical features, and scale numeric fields before training various models. All modeling and analysis are done in Python using libraries like scikit-learn, pandas, matplotlib, and Keras.

Data Exploration & Preprocessing

We begin by loading the CSV data into pandas and examining its shape and head. Initial EDA includes a histogram of the category feature and a line plot of adview counts to inspect data distribution (not shown here). We then clean the data:

Outlier Removal: Videos with extremely high adview counts (≥2,000,000) are removed to reduce skew in the target.

Invalid Entries: Any rows where views, likes, dislikes, or comments contain non-numeric placeholders (e.g. 'F') are dropped.

Category Encoding: The category feature (originally letters A–H) is mapped to integers 1–8 for modeling.

Numeric Conversion: Text fields for views, likes, dislikes, and comments are converted to numeric types.

Label Encoding: Categorical columns like duration, vidid, and published are label-encoded so that each unique text is mapped to a number
scikit-learn.org
. This transforms non-numeric labels into numeric codes.

Next, we convert the video duration to seconds. The original duration is an ISO 8601 string (e.g. "PT15M30S"). We parse hours, minutes, and seconds from this string and compute the total seconds, storing this numeric value back into the dataset. After encoding and conversion, the features are all numeric. Finally, we drop the original adview (target) and vidid columns from the feature set. We set aside the adview column as the target variable.

To preprocess systematically, we apply the following steps:

LabelEncoder: Convert categorical text features to numeric labels
scikit-learn.org
.

Drop Unused Columns: Remove adview from features (it becomes the target) and drop unique ID columns.

Create Target Variable: Store adview counts as Y_train.

Each step cleans and transforms the raw data into a form suitable for machine learning.

Feature Scaling

Before training models, we split the data into training and test sets (80/20 split, random state 42). We then apply MinMax scaling to the features
scikit-learn.org
. This scales each feature individually to the 0–1 range, which can improve model training and convergence. Specifically, we fit a MinMaxScaler on the training features and transform both training and test sets so that the minimum value maps to 0 and the maximum to 1
scikit-learn.org
. This preserves the relative relationships between values while normalizing the scale of each feature.

Model Training & Evaluation

We train and evaluate several regression models on the processed data. Model performance is reported using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) on the test set. These metrics quantify the average and squared differences between predicted and actual adview counts. We compare the following models:

Linear Regression: A basic regression model that fits a linear equation to estimate the target
en.wikipedia.org
. It assumes a linear relationship between features and the target, serving as a simple baseline.

Support Vector Regression (SVR): An SVM-based regressor that finds a function approximating the data with a margin. SVR predicts continuous outputs (unlike SVM for classification) by fitting a regression line that can handle non-linear relationships through kernels
analyticsvidhya.com
. Its ability to use kernels (linear, RBF, etc.) allows capturing complex patterns.
The SVR model was trained using default kernels (e.g. RBF) to predict adviews, and its error metrics were calculated for comparison
analyticsvidhya.com
.

Random Forest Regressor: An ensemble of decision trees. A random forest builds many trees on random subsets of the data, then averages their predictions
epa.gov
. This typically reduces variance and improves accuracy over a single tree
epa.gov
. It can capture complex feature interactions without heavy parameter tuning.
We trained a Random Forest (200 trees, max depth 25, etc.) on the data. Each tree in the forest votes on the predicted adview count, and the final output is averaged across trees
epa.gov
. The forest approach helps generalize better than one tree alone.

Artificial Neural Network (ANN): A feedforward neural network (a Multi-Layer Perceptron) with fully connected layers. We used a Keras sequential model with two hidden dense layers (6 neurons each, ReLU activation) and an output layer of size 1. ANNs model complex non-linear relationships by learning weighted sums and activations
geeksforgeeks.org
. The network was trained for 100 epochs with the Adam optimizer and mean-squared-error loss.
The ANN maps input features through hidden layers to the predicted adview. Its architecture of dense layers is capable of capturing intricate patterns in the data
geeksforgeeks.org
.

Decision Tree Regressor: A single decision tree that predicts the target by splitting on feature values
geeksforgeeks.org
. We trained a DecisionTreeRegressor as well. (Although its performance often underwhelms compared to the ensemble, we save it for potential use.)

After training each model, we evaluate on the test set using MAE, MSE, and RMSE to compare performance. (Lower values indicate better fit.)

Model Saving and Output

Finally, the trained models are saved for future use. The decision tree is serialized to a file using joblib.dump, and the ANN model is saved in Keras’s HDF5 format (.h5). This allows loading the trained models later for inference without retraining.
## Model Output
Here’s a sample prediction from the trained model:

<p align="center">
  <a href="https://raw.githubusercontent.com/venkateshmoodu/Youtube-adview-Prediction-ML-PROJECT-/refs/heads/main/Screenshot%202025-10-22%20123338.png">
    <img src="https://raw.githubusercontent.com/venkateshmoodu/Youtube-adview-Prediction-ML-PROJECT-/refs/heads/main/Screenshot%202025-10-22%20123338.png" alt="Logo">
  </a>

</p>

Overall, this README details the end-to-end process: data loading, cleaning, preprocessing (encoding and scaling), model training with several algorithms, evaluation, and model persistence. Each modeling approach is documented with references to the underlying methods and ensures reproducibility of the pipeline

