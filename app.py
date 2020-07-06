import time
import pickle
import pandas as pd
import streamlit as st

df = pd.read_csv("BankNote_Authentication.csv")


with open("classifier.pkl", "rb") as file:
    classifier = pickle.load(file)


def highlight_survived(s):
    return ['background-color: #96ff94']*len(s) if s.Class else ['background-color: #ffcccb']*len(s)


def predict_note_authentication(variance, skewness, curtosis, entropy):
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)
    return prediction


def main():
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
    Class = ""
    if st.button("Predict"):
        result = predict_note_authentication(
            variance, skewness, curtosis, entropy)
        if result:
            st.success('The Note is Genuine')
            st.balloons()
        else:
            st.error('This Note is Forged')

    if st.checkbox("Dataset"):
        with st.spinner('Wait for it.....'):
            time.sleep(5)
        st.dataframe(df.style.apply(highlight_survived, axis=1))
        st.info(
            'Data were extracted from images that were taken from genuine and forged banknote-like specimens. Red[0] denotes forged and green denotes genuine[1].')


if __name__ == '__main__':
    main()
