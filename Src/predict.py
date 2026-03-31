import pickle

model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

msg = input("Enter message: ")

msg_vec = vectorizer.transform([msg])
prediction = model.predict(msg_vec)

print("Result:", prediction[0])
