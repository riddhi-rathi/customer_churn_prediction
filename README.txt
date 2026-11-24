CUSTOMER CHURN PREDICTION - LOCAL HOST

1. Extract this ZIP into a folder.
2. Copy WA_Fn-UseC_-Telco-Customer-Churn.csv into the same folder as app.py.
3. Open the folder in VS Code.
4. Open Terminal > New Terminal.
5. Install dependencies:
   python -m pip install -r requirements.txt
6. Start the website:
   python -m streamlit run app.py
7. Open http://localhost:8501 if it does not open automatically.
8. Stop the server with Ctrl+C in the terminal.

The app trains Logistic Regression (with PCA), Random Forest, and XGBoost if installed.
It selects the model with the highest test F1-score and provides a customer prediction form.
