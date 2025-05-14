import pickle
from joblib import load
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

X_train = np.load('x_train.npy', allow_pickle=True)
X_test = np.load("x_test.npy", allow_pickle=True)
Y_train = np.load("y_train.npy", allow_pickle=True)
Y_test = np.load("y_test.npy", allow_pickle=True)

print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)

mail_data = ["I hope all is well. I had fun meeting you and designing our Career Focused Project together on Tuesday. I know we agreed that I would make a PowerPoint presentation, but I don't have any way to save my work after each session. Could you please bring a jump drive to next session? If you can't, I will ask Ms. Johnson for one tomorrow.I am also interested in learning the importance of teamwork at your job. What do you think makes someone an effective team player in the workforce? What type of team assignments do you normally work on? Do you prefer working in teams or on your own? Thank you for taking the time to answer my questions and I look forward to hearing from you.Sincerely,Derrick Smith"]

# svm_model = load('dump_dt_model.joblib')
# svm_model = svm_model.fit(X_train, Y_train)
dt_model = load(open('dump_svm_model.joblib', 'rb'))

# load the vectorizer
loaded_vectorizer = load(open('vectorizer.joblib', 'rb'))

data_transformed = loaded_vectorizer.transform(mail_data).toarray()

print(data_transformed)

predictions = dt_model.predict(data_transformed)

if predictions==0:
    print("Message is Ham")
else:
    print("Message is Spam")