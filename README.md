
Please visit https://nlp10kfilingsrk.streamlit.app/ for the Interactive Dashboard.


---NLP on 10K Filings Project (Rishi Kumra) ---
  
\## Data  
The dataset consists of 10K filings for four selected stocks (names can
be specified) from 2014 to 2019. The raw data is sourced from EDGAR, and sentiment analysis is performed using NLP techniques
to derive sentiment scores for each filing. 
  
\## Sentiment Analysis  
Sentiment analysis is conducted using NLTK, SpacY and Mcdonald's word dictionary. The sentiment scores derived include:  
- \*\*Litigious Score\*\*  
- \*\*Superfluous Score\*\*  
- \*\*Interesting Score\*\*  
- \*\*Modal Score\*\*  
- \*\*Polarity Score\*\*  
- \*\*Average Sentence Length\*\*  
- \*\*Fog Index\*\*  
  
These scores help determine the sentiment outlook of each filing, which
is a crucial input for predicting future stock performance.  
  
\## Predictive Modeling  
The project utilizes both classification and regression techniques:  
1. \*\*Classification Algorithms\*\*: To classify the sentiment outlook
(positive or negative).  
2. \*\*Regression Algorithms\*\*: To predict the actual continuous
return for the following year based on the sentiment scores.  
  
Algorithms used include:  
- Logistic Regression  
- Random Forest Classifier
- Support Vector Classifier
- Linear Regression
- Gradient Boosting Regressor  
  
\## Results  
The Streamlit Dashboard is made for a potential client/non-Python user
to navigate the results.  
The results are presented in the form of visualizations and tables,
showcasing the models\' performance in predicting future returns. The
saved plots and tables can also be found in the \`results\` directory.  
  
\## Technologies Used  
- Python  
- Pandas  
- Scikit-learn  
- NLTK/SpaCy  
- Matplotlib/Seaborn  
- Jupyter Notebook  
  
\## Installation  
To set up the project, clone this repository:
\`\`\`bash  
git clone https://github.com/quantrkucb/NLP_10K_Filings.git
