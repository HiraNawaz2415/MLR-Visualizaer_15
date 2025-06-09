import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("🚗 Multiple Linear Regression: Fuel Consumption Predictor")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Sample of your data:")
    st.dataframe(df.head())

    # Check for required columns
    st.write("### Columns detected:", list(df.columns))
    feature_cols = st.multiselect("Select feature columns", df.columns.tolist())
    target_col = st.selectbox("Select target column", df.columns.tolist())

    if len(feature_cols) > 0 and target_col:
        X = df[feature_cols]
        y = df[target_col]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Show regression equation
        coefs = model.coef_
        intercept = model.intercept_
        st.write("### Regression Equation:")
        equation = f"{target_col} = {intercept:.2f}"
        for feat, coef in zip(feature_cols, coefs):
            sign = "+" if coef >= 0 else "-"
            equation += f" {sign} {abs(coef):.2f} × {feat}"
        st.latex(equation)

        # Predict user input
        st.sidebar.header("Input your own data to predict")
        user_input = []
        for feat in feature_cols:
            val = st.sidebar.number_input(f"Input {feat}", value=float(X[feat].mean()))
            user_input.append(val)

        user_pred = model.predict([user_input])[0]
        st.subheader("🔮 Prediction for your input:")
        st.write(f"{target_col} = {user_pred:.2f}")

        # Show metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        st.subheader("📊 Model Performance Metrics:")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        st.write(f"R² Score: {r2:.2f}")

        # Visualizations
        st.subheader("📈 Visualizations")

        # Scatter plot of actual vs predicted
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color='blue', edgecolors='k')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        # Residual plot
        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_pred, residuals)
        ax2.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residual Plot")
        st.pyplot(fig2)

        # 3D plot if exactly 2 features
        if len(feature_cols) == 2:
            fig3 = plt.figure()
            ax3 = fig3.add_subplot(111, projection='3d')
            ax3.scatter(X_test[feature_cols[0]], X_test[feature_cols[1]], y_test, color='blue', label='Actual')
            ax3.scatter(X_test[feature_cols[0]], X_test[feature_cols[1]], y_pred, color='red', label='Predicted')
            ax3.set_xlabel(feature_cols[0])
            ax3.set_ylabel(feature_cols[1])
            ax3.set_zlabel(target_col)
            ax3.set_title("3D Plot: Actual vs Predicted")
            ax3.legend()
            st.pyplot(fig3)

    else:
        st.warning("Please select at least one feature column and one target column.")
else:
    st.info("Upload a CSV file to get started.")
