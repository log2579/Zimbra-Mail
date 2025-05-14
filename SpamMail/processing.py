from joblib import load
import sys 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string
import nltk
from nltk.corpus import stopwords

#nltk.download('stopwords')

arg = sys.argv[1]

def clean_mess(text):
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Convert to lowercase
    text = text.lower()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)

with open(arg, 'r') as f:
    email = f.read()

from_address = re.search(r'From: (.+)', email).group(1)
to_address = re.search(r'To: (.+)', email).group(1)
date_get = re.search(r'Date: (.+)', email).group(1)
subject = re.search(r'Subject: (.+)', email).group(1)
message_id = re.search(r'Message-ID: (.+)', email).group(1)

print("From:", from_address)
print("To:", to_address)
print("Date:", date_get)
print("Subject:", subject)
print("Message-ID:", message_id)

# doan message
# --=_7919a7cb-4852-47ca-9734-03cf0b12eb9c
# Content-Type: text/plain; charset=utf-8
# Content-Transfer-Encoding: 7bit
#
# khong con gi de mat.
# dong 2 mesage
# dong message
#
# --=_7919a7cb-4852-47ca-9734-03cf0b12eb9c
# Content-Type: text/html; charset=utf-8
# Content-Transfer-Encoding: 7bit

pattern = r'Content-Type: text/plain; charset=utf-8\nContent-Transfer-Encoding: 7bit\n\n(.*?)\n\n--='
# "." khop voi bat ki ki tu nao ngoai tru (/n)
# "*" co the xuat hien 0 hoac nhieu lan
# "?" co khong hoac 1 ki tu dang sau
# ".*" greedy quantifier khop voi chuoi dau den chuoi cuoi (neu chi co 1 chuoi duy nhat giong voi ".*?")
# ".*?" reluctant or "non-greedy" quantifier khop voi chuoi dau

match = re.search(pattern, email, re.DOTALL)
# re.DOTALL khop voi bat ki ki tu nao ke ca dau (.) va dau xuong dong

if match:
    email_msg = match.group(1).strip()
else:
    email_msg = ""

print("Message:\n", email_msg)

email_msg = subject + " " + email_msg

#ms=['XJS*C4JDBQADN1.NSBN3*2IDNEN*GTUBE-STANDARD-ANTI-UBE-TEST-EMAIL*C.34X']
ms = [""]

#ms.append(clean_mess(email_msg))
ms = [clean_mess(email_msg)]
#print(type(ms))

# load the vectorizer
loaded_vectorizer = load(open('vectorizer.joblib', 'rb'))

# load the model
svm_model = load(open('dump_svm_model.joblib', 'rb'))


data_transformed = loaded_vectorizer.transform(ms).toarray()

predictions = svm_model.predict(data_transformed)

if predictions==0:
    print("Status Predict: :  Ham")
else:
    print("Status Predict: :  Spam")