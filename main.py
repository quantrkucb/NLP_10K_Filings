import pandas as pd


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image
st.markdown(
    """
    <style>
    .main {
        max-width: 65%;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def main():
    data_ = pd.read_csv('data/data_2cleaned.csv')
    data_ = data_.drop(columns = ['Unnamed: 0'])
    data_['Outlook'] = np.where(data_['Future Return']>0.04, 1, 0)
    # Data split
    y = data_['Outlook']
    X = data_[['litigous_score', 'superfluous_score', 'interesting_score', 'modal_score', 'harvard_1', 'polarity_score', 'avg_sen_length', 'fog_index']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)

    # Streamlit Dashboard
    st.title("Classification Approach")

    st.text("""
    First, we can treat this as a classification problem. 
    I define a threshold of annual returns (4%) above which I define as having a
    strong outlook (1) and anything below 4% is considered to have a weak outlook (0).
    
    Then, we can apply some standard classification algorithms:
        -Logistic Regression
        -Random Forest
        -Support Vector Classifier""")


    st.header("---- Annual Outlook ---- ")
    # Logistic Regression
    st.header("Logistic Regression")
    model = LogisticRegression()
    model.fit(X_train, y_train)
    log_reg_acc = model.score(X_test, y_test)
    st.write(f"Logistic Regression Accuracy: {log_reg_acc:.4f}")

    # Random Forest with adjustable n_estimators
    st.header("Random Forest Classifier")
    rf_estimators = st.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=50)
    rf_model = RandomForestClassifier(n_estimators=rf_estimators, random_state=25)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    st.write(f"Random Forest Accuracy: {rf_accuracy:.4f}")

    # Feature Importance for Random Forest
    st.subheader("Feature Importance (Random Forest)")
    rf_feature_importance = rf_model.feature_importances_
    fig, ax = plt.subplots(figsize=(5, 2))
    sns.barplot(x=rf_feature_importance, y=X.columns, ax=ax)
    st.pyplot(fig)


    col1, col2 = st.columns(2)
    with col1:
    # Confusion Matrix for Random Forest
        st.subheader("Confusion Matrix (Random Forest)")
        rf_cm = confusion_matrix(y_test, rf_pred)
        fig, ax = plt.subplots()
        sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Predictions vs Actual:")
        data_.loc[X_test.index, 'Predictions'] = rf_model.predict(X_test)
        data_.loc[X_test.index][['year', 'company', 'Outlook', 'Predictions']]
    rf_model.predict(X_test)
    weighting = 1/len(rf_model.predict(X_test))
    return_ = []
    # model.predict(X_test)
    index_ = 0
    weights_ = []
    for i in rf_model.predict(X_test):
        if i == 0:
            weights_.append(-weighting)
            return_.append(-weighting*data_.loc[X_test.index[index_], 'Future Return'])
        else:
            weights_.append(weighting)
            return_.append(weighting*data_.loc[X_test.index[index_], 'Future Return'])
        index_ +=1

    st.write("Net portfolio return for an equal-weighted portfolio:")
    st.write(f"{(sum(return_)*100):.2f} %")

    from sklearn.svm import SVC

    svc_model = SVC(kernel='rbf', random_state=25)
    svc_model.fit(X_train, y_train)
    svc_pred = svc_model.predict(X_test)
    svc_accuracy = svc_model.score(X_test, y_test)
    print(f"SVC Accuracy: {svc_accuracy:.4f}")

    # st.subheader("Feature Importance (SVC)")
    # svc_feature_importance = svc_model.feature_importances_
    # fig, ax = plt.subplots()
    # sns.barplot(x=svc_feature_importance, y=X.columns, ax=ax)
    # st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
    # Confusion Matrix for Random Forest
        st.subheader("Confusion Matrix (SVC)")
        rf_cm = confusion_matrix(y_test, svc_pred)
        fig, ax = plt.subplots()
        sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Predictions vs Actual (SVC):")
        data_.loc[X_test.index, 'Predictions'] = svc_model.predict(X_test)
        data_.loc[X_test.index][['year', 'company', 'Outlook', 'Predictions']]
    weighting = 1/len(svc_model.predict(X_test))
    return_ = []
    # model.predict(X_test)
    index_ = 0
    weights_ = []
    for i in svc_model.predict(X_test):
        if i == 0:
            weights_.append(-weighting)
            return_.append(-weighting*data_.loc[X_test.index[index_], 'Future Return'])
        else:
            weights_.append(weighting)
            return_.append(weighting*data_.loc[X_test.index[index_], 'Future Return'])
        index_ +=1

    st.write("Net portfolio return for an equal-weighted portfolio:")
    st.write(f"{(sum(return_)*100):.2f} %")



    # # Gradient Boosting with adjustable n_estimators
    # st.header("Gradient Boosting Classifier")
    # gb_estimators = st.slider("Number of Estimators (Gradient Boosting)", min_value=50, max_value=500, value=100, step=50)
    # gb_model = GradientBoostingClassifier(n_estimators=gb_estimators, random_state=0)
    # gb_model.fit(X_train, y_train)
    # gb_pred = gb_model.predict(X_test)
    # gb_accuracy = accuracy_score(y_test, gb_pred)
    # st.write(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")

    # # Feature Importance for Gradient Boosting
    # st.subheader("Feature Importance (Gradient Boosting)")
    # gb_feature_importance = gb_model.feature_importances_
    # fig, ax = plt.subplots()
    # sns.barplot(x=gb_feature_importance, y=X.columns, ax=ax)
    # st.pyplot(fig)

    # # Confusion Matrix for Gradient Boosting
    # st.subheader("Confusion Matrix (Gradient Boosting)")
    # gb_cm = confusion_matrix(y_test, gb_pred)
    # fig, ax = plt.subplots()
    # sns.heatmap(gb_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    # st.pyplot(fig)

    # Run Streamlit
    # In terminal, run: streamlit run dashboard.py

def page_one():
    # Select company for detailed analysis
    data_ = pd.read_csv('data/data_2cleaned.csv')
    data_ = data_.drop(columns = ['Unnamed: 0'])
    data_['Outlook'] = np.where(data_['Future Return']>0.04, 1, 0)

    st.text("""
    In this project, I scraped 10K filing data from EDGAR for 5 years 2014:2019 for 4 companies. 
    On each of these, I calculate scores using the nltk library and the Mcdonald 
    word dictonary for:
            - Litigous Score: Measures the likelihood of legal language
            - Superfluous Score: Checks for unnecessary language
            - Interesting Score: Highlights engaging content
            - Modal Score: Tracks the frequency of modal verbs
            - Polarity Score: The higher this is the more positive the sentiment. 
            - Harvard_1: Measures sentiment based on Harvard's dictionary
            - Avg Sentence Length: Analyzes readability
            - Fog Index: A readability metric based on sentence length and complex words
            
    Data Sources: Kaggle, EDGAR Database, Mcdonald website, NLTK library, Yahoo Finance
    
    The goal is then to use these calculated scores as features in a regression to predict
    the return at t+1 based on 10-K filings at time t.
    
    I look at annual returns and the next month's returns to see if predictions are better
    at a shorter horizon or longer horizon.
     
    I also try to approach the prediction in 2 approaches: a binary classification problem (strong or weak return outlook),
    and a continuous linear regression approach with exact returns being predicted """)
    company = st.selectbox("Select Company", data_['company'].unique())

    # Filter data by selected company
    company_data = data_[data_['company'] == company]

    col1, col2 = st.columns(2)

    with col1:
        # Line plot for NLP-based scores over time
        st.subheader(f"NLP-based Score Trends for {company}")
        fig, ax = plt.subplots()
        company_data.plot(x='year', y=['litigous_score', 'superfluous_score', 'interesting_score', 'modal_score', 'harvard_1', 'polarity_score'], ax=ax)
        plt.ylabel('Scores')
        plt.title('NLP-based Scores Over Time')
        st.pyplot(fig)


    with col2:
        st.subheader(f"Annual Return Trends for {company}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='year', y='Future Return', data=company_data, ax=ax, palette='viridis')
        plt.ylabel('Future Return')
        plt.title('Annual Return Trends Over Time')
        st.pyplot(fig)

    st.subheader(f"Correlation Heatmap for {company}")
    correlation_matrix = company_data[['litigous_score', 'superfluous_score', 'interesting_score', 'modal_score', 'harvard_1', 'polarity_score', 'avg_sen_length', 'fog_index', 'Outlook']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Correlation Between Scores and Future Return')
    st.pyplot(fig)
    plt.legend()
    # st.pyplot(fig)
        # Line plot for NLP-based scores over time


def page_two():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score



    st.write("""
    It's clear that linear regression isn't the right approach for predicting returns especially in a small dataset such as this.
    Hence the classification approach (which predicts the outlook ie the relative size/direction is better).
    However, the best approach here in continuous regression is the gradient boosting method which has a small positive R-squared. 
    """)
    # Sample DataFrame creation
    # Assuming you have a DataFrame `data_` already defined
    # Uncomment and adjust the following line based on your data
    # data_ = pd.read_csv('your_data_file.csv')

    results_df = pd.read_csv('results/optimized_model_comparison.csv')

    st.header("---- Annual Return ----")
    # Load and display saved plots for each model
    model_names = results_df['Model'].unique()
    for model_name in model_names:
        # Load and display the corresponding plot
        image_path = f"/Users/rishikumra/Downloads/Projects/sentiment-analysis-sec-master/results/{model_name.replace(' ', '_')}_actual_vs_predicted.png"
        try:
            image = Image.open(image_path)
            st.image(image, caption=f"{model_name}: Actual vs Predicted")
        except FileNotFoundError:
            st.write(f"No plot found for {model_name}")
        
        predictions_df = pd.read_csv(f"results/{model_name.replace(' ', '_')}_predictions.csv")
        st.subheader(f"{model_name} Predictions")
        st.write(predictions_df.head(10)) 

    # Display the MSE comparison table for optimized models
    st.subheader("Optimized Model Comparison Table")
    st.write(results_df)

    # Load and display predictions for each model
    #   # Display the first 10 predictions
    # data_ = pd.read_csv('/Users/rishikumra/Downloads/Projects/sentiment-analysis-sec-master/data/data_2cleaned.csv')
    # data_ = data_.drop(columns = ['Unnamed: 0'])
    # # Prepare the data
    # y = data_['Future Return']
    # X = data_[['litigous_score','superfluous_score','interesting_score', 
    #             'modal_score', 'harvard_1', 'polarity_score', 
    #             'avg_sen_length','fog_index']]

    # # Split the data into training and testing sets
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=25)

    # # Initialize a DataFrame to store the results
    # results_df = pd.DataFrame(columns=['Model', 'MSE', 'R² Score'])

    # # Function to evaluate and display results for a model
    # def evaluate_model(model_name, model, x_train, y_train, x_test, y_test, results_df):
    #     # global results_df  # Reference the global DataFrame

    #     st.subheader(f"{model_name} Model")
        
    #     # Fit the model
    #     model.fit(x_train, y_train)
    #     predictions = model.predict(x_test)

    #     # Calculate metrics
    #     mse = mean_squared_error(y_test, predictions)
    #     r2 = r2_score(y_test, predictions)

    #     # Add results to the DataFrame
    #     results_df = results_df.append({'Model': model_name, 'MSE': mse, 'R² Score': r2}, ignore_index=True)

    #     # Display metrics
    #     st.write(f"Mean Squared Error: {mse:.2f}")
    #     st.write(f"R² Score: {r2:.2f}")

    #     # Plot actual vs predicted values
    #     fig, ax = plt.subplots(figsize=(8, 4))
    #     sns.scatterplot(x=y_test, y=predictions, ax=ax)
    #     ax.plot(y_test, y_test, color='red', linestyle='--')  # 45-degree line
    #     ax.set_xlabel("Actual Future Return")
    #     ax.set_ylabel("Predicted Future Return")
    #     ax.set_title(f"{model_name}: Actual vs Predicted")
    #     st.pyplot(fig)

    #     # Show predictions in a table
    #     predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    #     st.write(predictions_df.head(10))

    #     return results_df

    # # Function to optimize and evaluate models
    # def optimize_and_evaluate_models(results_df):
    #     # Linear Regression with Standardization
    #     scaler = StandardScaler()
    #     x_train_scaled = scaler.fit_transform(x_train)
    #     x_test_scaled = scaler.transform(x_test)

    #     linear_model = LinearRegression()
    #     results_df = evaluate_model("Optimized Linear Regression", linear_model, x_train_scaled, y_train, x_test_scaled, y_test, results_df)

    #     # Random Forest Regression with Hyperparameter Tuning
    #     rf_model = RandomForestRegressor(random_state=25)
    #     param_grid_rf = {
    #         'n_estimators': [50, 100, 200],
    #         'max_depth': [None, 10, 20, 30],
    #         'min_samples_split': [2, 5, 10]
    #     }
    #     rf_grid_search = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='neg_mean_squared_error', verbose=0)
    #     rf_grid_search.fit(x_train, y_train)
    #     best_rf_model = rf_grid_search.best_estimator_
    #     results_df = evaluate_model("Optimized Random Forest Regression", best_rf_model, x_train, y_train, x_test, y_test, results_df)

    #     # Gradient Boosting Regression with Hyperparameter Tuning
    #     gb_model = GradientBoostingRegressor(random_state=25)
    #     param_grid_gb = {
    #         'n_estimators': [100, 200],
    #         'learning_rate': [0.01, 0.1, 0.2],
    #         'max_depth': [3, 5, 7]
    #     }
    #     gb_grid_search = GridSearchCV(gb_model, param_grid_gb, cv=5, scoring='neg_mean_squared_error', verbose=0)
    #     gb_grid_search.fit(x_train, y_train)
    #     best_gb_model = gb_grid_search.best_estimator_
    #     results_df = evaluate_model("Optimized Gradient Boosting Regression", best_gb_model, x_train, y_train, x_test, y_test, results_df)

    #     return results_df
    # # Streamlit app layout
    # st.title("Optimized Continuous Prediction Models for Future Returns")

    # # Optimize and evaluate models
    # results_df = optimize_and_evaluate_models(results_df)

    # # Display the MSE comparison table for optimized models
    # st.subheader("Optimized Model Comparison Table")
    # st.write(results_df)

    # Additional visualizations if desired
    # st.subheader("Feature Importance (Random Forest)")
    # rf_importance = results_df.loc[results_df['Model'] == "Optimized Random Forest Regression", 'MSE']
    # fig, ax = plt.subplots(figsize=(6, 4))
    # sns.barplot(x=rf_importance, y=X.columns, ax=ax)
    # ax.set_title("Feature Importance")
    # st.pyplot(fig)


st.sidebar.title("10-K Filings Sentiment Analysis")
page = st.sidebar.radio("Select a page:", ["Exploratory Data Analysis", "Classification Results", "Regression Results"])

# Render the selected page
if page == "Exploratory Data Analysis":
    page_one()
elif page == "Classification Results":
    main()
elif page == "Regression Results":
    page_two()
