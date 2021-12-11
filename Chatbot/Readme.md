# Chatbot interface trained using NLTK and Tensorflow(Keras)  
Model is created by taking a dictionary with various sample input questions from the users and the corresponding tag to what category does the question like:
(greeting, goodbye, hospital_search, thanks, options, blood_pressure, adverse_drug etc.) the same dictionary also has sample responses to the questions asked. 
First the model is trained  on the input question form the users based on bag of words representation of the input sentence converted to a lemmatized version, 
it is then fed to a Neural network of 2 dense layers and an output softmax layer. 
The model is then saved and for the chatbot first we identify to what class the input question belongs and based on the best prediction above a threshold value 
we choose a random response from the response section of the dictionary for a particular class. 
The GUI for the chatbot is created using tkinter.  

<p align="center">
<img align="center" src="https://github.com/mayanksinghkgp/NLP_Resources/blob/main/Chatbot/chatbot.jpg" width="250">
</p>
