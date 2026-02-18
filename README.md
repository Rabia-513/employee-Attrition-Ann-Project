 **ANN-Based Employee Attrition Prediction**:


# ANN-Based Employee Attrition Prediction

This project implements an Artificial Neural Network (ANN)-based model to predict employee attrition using various deep learning techniques. The model is optimized through multiple phases, and the application is built with **Streamlit** to provide an interactive dashboard for model evaluation and comparison. The app helps businesses predict which employees are likely to leave, thereby improving retention strategies.

## Features

* **Dataset Preview**: Allows users to preview the dataset and see its shape, as well as a preview of the first few rows.
* **Phase 1: Exploratory Data Analysis (EDA)**:

  * Interactive data exploration features, including missing value heatmaps, correlation matrices, histograms, box plots, and outlier handling.
* **Phase 2: Baseline ANN**: Build a simple baseline ANN model and visualize its architecture.
* **Phase 3: Optimizer Comparison**: Compare the performance of different optimizers (Adam, SGD, RMSprop, Adagrad) and analyze training curves.
* **Phase 4: Optimized ANN**: Implement advanced techniques like batch normalization, dropout regularization, and optimized ANN architecture to improve model performance.

## Installation

To run this project locally, you will need Python 3.7 or higher and several required libraries. You can install them by following these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/ann-employee-attrition-prediction.git
   cd ann-employee-attrition-prediction
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. Upload the dataset (`Modified_HR_Employee_Attrition.csv`) through the interface to start the application.

## Technologies Used

* **Streamlit**: For building the interactive user interface and dashboard.
* **TensorFlow/Keras**: For building and training the ANN model.
* **Scikit-learn**: For preprocessing, splitting data, and implementing various machine learning techniques.
* **Pandas & NumPy**: For data manipulation and handling.
* **Matplotlib & Seaborn**: For data visualization.

## Project Structure

```
ann-employee-attrition-prediction/
│
├── app.py                  # Main Streamlit app file
├── Modified_HR_Employee_Attrition.csv  # Dataset
├── README.md               # Project documentation
```

## How to Use the App

1. **Upload Dataset**: When you open the app, upload the **Modified_HR_Employee_Attrition.csv** dataset to begin the analysis.
2. **Explore the Data**: In the **Dataset Preview** section, preview the dataset to understand its structure and columns.
3. **Phase 1 - EDA & Preprocessing**:

   * Visualize missing values, correlations, and handle outliers in the data interactively.
   * Normalize numerical features to prepare for model training.
4. **Phase 2 - Baseline ANN**:

   * Configure and build a baseline Artificial Neural Network (ANN) model.
   * View the architecture summary of the baseline model.
5. **Phase 3 - Optimizer Comparison**:

   * Train the baseline ANN with different optimizers and compare their performance.
6. **Phase 4 - Optimized ANN**:

   * Build and train an optimized ANN with advanced techniques like batch normalization and dropout regularization.
   * Compare the results from different optimizers and evaluate the final model.

## Future Enhancements

* **Advanced Hyperparameter Tuning**: Implement more advanced techniques like **GridSearchCV** and **RandomizedSearchCV** for better hyperparameter optimization.
* **Feature Engineering**: Improve feature selection and extraction methods to enhance model performance.
* **Deployment**: Deploy the app to the web using **Heroku** or **Streamlit Cloud** for wider access.


