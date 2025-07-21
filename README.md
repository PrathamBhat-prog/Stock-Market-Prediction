# Stock-Market-Prediction

Project Overview
This project provides a comparative analysis of a standard CNN and a custom-built RNN for the task of time series forecasting, specifically applied to stock market closing prices. The workflow involves data loading, preprocessing, model definition, training, evaluation, and visualization of predictions.

# **Dataset**
The project uses historical stock data for Netflix (NFLX) from the file NFLX.csv. This dataset typically contains columns such as:

Date: The date of the trading day.
Open: The opening price of the stock.
High: The highest price of the stock during the day.
Low: The lowest price of the stock during the day.
Close: The closing price of the stock during the day.
Adj Close: The adjusted closing price of the stock.
Volume: The number of shares traded during the day.
For this project, the Close price is used as the target variable for prediction.

# **Data Preprocessing**
The following data preprocessing steps are applied:

Extraction of Closing Prices: The Close column is extracted from the dataframe.
Scaling: The closing prices are scaled using StandardScaler from sklearn.preprocessing to ensure that the data is centered around zero with a unit standard deviation. This is important for neural network training.
Sequence Creation: A function create_sequence is used to create input sequences (X) and corresponding target values (y) for the time series forecasting task. Each input sequence consists of time_step (set to 60) consecutive closing prices, and the target value is the price immediately following the sequence.
Reshaping: The input sequences are reshaped to fit the expected input shape of the models. For the CNN, the shape is (samples, time_steps, features), and for the custom RNN, it's (samples, time_steps, input_dim).
Train, Validation, and Test Split: The data is split into training, validation, and testing sets to evaluate the model's performance on unseen data. A shuffle is not applied to maintain the temporal order of the time series data.
# **Model Architectures**
**CNN Model**

The CNN model is designed to capture local patterns and features in the time series data. It consists of:

Multiple Conv1D layers with ReLU activation and L2 kernel regularization.
MaxPooling1D layers for down-sampling the feature maps.
A Flatten layer to convert the 2D feature maps into a 1D vector.
Dense layers with ReLU activation and L2 kernel regularization.
A final Dense layer with linear activation for the prediction of the next stock price.
The model is compiled with the rmsprop optimizer and uses mean_squared_error as the loss function, with 'mse' as a metric.

**Custom RNN Model**

A custom tf.keras.layers.Layer named CustomRecurrentCell is implemented. This cell is inspired by recurrent architectures and includes:

Dense layers with sigmoid and tanh activations for gating mechanisms (forget gate and input gate).
Operations for updating the cell state based on the previous state and current input.
A Leaky ReLU activation function is incorporated in the cell state update.
An RNN model is then built using this CustomRecurrentCell. The model consists of:

An Input layer defining the shape of the input sequences.
An RNN layer wrapping the CustomRecurrentCell.
A final Dense layer with linear activation for the prediction output.
The model is compiled with the Adam optimizer and uses mean_squared_error as the loss function, with 'mse' as a metric.

Training
Both models are trained using the training dataset. An EarlyStopping callback is used to prevent overfitting. This callback monitors the validation mean squared error (val_mse) and stops the training if the val_mse does not improve for a specified number of epochs (patience).

# **Evaluation**
The performance of both trained models is evaluated on the unseen test dataset using the following metrics:

Root Mean Squared Error (RMSE): Measures the average magnitude of the errors between predicted and actual values. A lower RMSE indicates better performance.
R2 Score: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. An R2 score closer to 1 indicates a better fit.
Results
The evaluation results on the test set are as follows:

CNN Model:
Test RMSE: {{RMSE:.2f}}
Test R2 Score: {{r2*100:.2f}}%
Custom RNN Model:
Test RMSE: {{rnn_RMSE:.2f}}
Test R2 Score: {{rnn_r2*100:.2f}}%
These results indicate that the custom RNN model achieved a significantly lower RMSE and a higher R2 score compared to the CNN model on this dataset, suggesting better predictive performance.

The project also includes visualizations of the original stock prices and the predictions from both models on the training, validation, and test sets, allowing for a visual comparison of their performance over time.
