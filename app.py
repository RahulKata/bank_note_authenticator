from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import streamlit as st
import pandas as pd
import pickle
import time

df = pd.read_csv("BankNote_Authentication.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest')
)
with open("classifier.pkl", "rb") as file:
    classifier = pickle.load(file)


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name == 'Random Forest':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 10, 100)
        params['n_estimators'] = n_estimators
    return params


params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC()
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                           max_depth=params['max_depth'], random_state=1234)
    return clf


def highlight_survived(s):
    return ['background-color: #96ff94']*len(s) if s.Class else ['background-color: #ffcccb']*len(s)


def predict_note_authentication(variance, skewness, curtosis, entropy):
    clf, score = train()
    prediction = clf.predict([[variance, skewness, curtosis, entropy]])
    return prediction, score


def train():
    clf = get_classifier(classifier_name, params)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, accuracy_score(y_test, y_pred)


def main():
    with st.spinner('Wait for it.....'):
        time.sleep(5)
    html_temp = """
    <div style="background-color:black;padding:20px;border: 5px ridge #66fcf1">
    <h1 style="color:white;text-align:center;color:#66fcf1">Bank Note Authenticator</h1>
    </div>
    <br>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    variance = st.text_input("Variance", "Enter the value....")
    skewness = st.text_input("Skewness", "Enter the value....")
    curtosis = st.text_input("Kurtosis", "Enter the value....")
    entropy = st.text_input("Entropy", "Enter the value....")
    result = ""
    score = ""
    Class = ""
    if st.button("Predict"):
        result, score = predict_note_authentication(
            variance, skewness, curtosis, entropy)
        if result:
            st.success('The Note is Genuine')
            st.balloons()
        else:
            st.error('This Note is Forged')
        st.write(f"Accuracy: {score*100} %")

    st.sidebar.info(
        'Data were extracted from images that were taken from genuine and forged banknote-like specimens. Red[0] denotes forged and green denotes genuine[1].')
    if st.sidebar.checkbox("Show Dataset"):
        st.dataframe(df.style.apply(highlight_survived, axis=1))


if __name__ == '__main__':
    main()
