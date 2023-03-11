# Importing the Libraries
import tensorflow as tf
import numpy as np
import pickle

# Load the model and tokenizer
model = tf.keras.models.load_model('model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

def Predict_Next_Words(model, tokenizer, text):
    """
        In this function we are using the tokenizer and models trained
        and we are creating the sequence of the text entered and then
        using our model to predict and return the the predicted word.
    
    """
    for i in range(3):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = np.array(sequence)
        preds = model.predict(sequence)
        predicted_class_index = np.argmax(preds)
        
        for key, value in tokenizer.word_index.items():
            if value == predicted_class_index:
                predicted_word = key
                break
        
        print(predicted_word)
        return predicted_word
    
# Predicting the next word
while(True):
    text = input("Enter your line: ")
    
    if text == "q":
        print("Ending The Program.....")
        break
    
    else:
        try:
            text = text.split(" ")
            text = text[-3]
            text = ''.join(text)
            
            Predict_Next_Words(model, tokenizer, text)
            
        except:
            continue
