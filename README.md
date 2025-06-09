# Multiple Linear Regression Visualizer

This is an interactive **Streamlit** web app that allows you to perform **multiple linear regression** on your own dataset with ease.

---

## Features

- **Upload your own CSV dataset**
- **Select input features and a target variable** for regression
- **Train a linear regression model** on the selected data
- **Display the regression equation** with coefficients and intercept
- **Show key regression metrics:** MAE, MSE, RMSE, R² Score
- **Visualize the regression results:**
  - For 1 input feature: 2D scatter plot with regression line
  - For 2 input features: 3D scatter plot with regression plane
  - For more than 2 features: Pairplot colored by residual errors

---

## Installation

Make sure you have Python installed (Python 3.7+ recommended).

1. Clone this repo or download the app file `app.py`
2. Install required packages:

```bash
pip install streamlit scikit-learn pandas matplotlib seaborn numpy
