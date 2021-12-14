# NLP_Resources
### Scripts and short projects related to Natural Language processing  

**1. Chatbot**  
A chatbot userinterface trained on a simple ANN using NLTK and tensorflow libraries for training and Tkinter for the GUI.  
[Details of the project here](Chatbot/Readme.md)

**2. Consumer complaint classification**  
Data downloaded form the US consumer complaints database with over 2.3 Million data points. Used NLTK, Keras and sklearn to perform classification of the customer narratives
in to different product classes with the help of Embedding and LSTM layers.  
[Consumer_Complaint_Clasification](https://github.com/mayanksinghkgp/NLP_Resources/tree/main/Consumer_Complaint_Clasification)  

**3. Corona Fake news identification**  
Dataset contains articles form different sources and also the label 'fake' or 'true'. Data is preprocessed using regex and vectorized with tfidf using PorterStemmer. This post
processed data and the labels are fed to a Logistic Regression model and tuned for the predictions. Model gives a F1 score of .92 for both classes.  
[Fake_News_identification](https://github.com/mayanksinghkgp/NLP_Resources/tree/main/Fake_News_identification)  

**4. Movie review sentiment analysis**  
Dataset is taken form the IMBD movie review data from Standford sentiment Analysis page. Data is tokenized in DistilBERT tokenized and padded to the longest sentence length. Another mask vector is created for the model to take care of the padding, and fed to the DistilBERT transformer. The [CLS] token embedding is taken from the output of the tranfoemer and is fed to a logistic regression model whose parameters are optimized in gridsearch.  
[DistilBERT_for_movie_review_sentiment](https://github.com/mayanksinghkgp/NLP_Resources/tree/main/DistilBERT_for_movie_review_sentiment)  

**n. Test_models**  
Trail models for different concepts  
