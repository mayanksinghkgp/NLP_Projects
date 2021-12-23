# NLP_Resources
### Scripts and short projects related to Natural Language processing  

**1. Consumer complaint classification**  
- Data downloaded form the US consumer complaints database with over 2.3 Million data points. Used NLTK, Keras and sklearn to perform classification of the customer narratives
in to different product classes with the help of Embedding and LSTM layers.  
[Consumer_Complaint_Clasification](https://github.com/mayanksinghkgp/NLP_Resources/tree/main/Consumer_Complaint_Clasification)  

**2. Chatbot**  
- A chatbot userinterface trained on a simple ANN using NLTK and tensorflow libraries for training and Tkinter for the GUI.  
[Details of the project here](Chatbot/Readme.md)


**3. Corona Fake news identification**  
- Dataset contains articles form different sources and also the label 'fake' or 'true'. Data is preprocessed using regex and vectorized with tfidf using PorterStemmer. This post
processed data and the labels are fed to a Logistic Regression model and tuned for the predictions. Model gives a F1 score of .92 for both classes.  
[Fake_News_identification](https://github.com/mayanksinghkgp/NLP_Resources/tree/main/Fake_News_identification)  

**4. Movie review sentiment analysis**  
- Dataset is taken form the IMBD movie review data from Standford sentiment Analysis page. Data is tokenized in DistilBERT tokenized and padded to the longest sentence length. Another mask vector is created for the model to take care of the padding, and fed to the DistilBERT transformer. The [CLS] token embedding is taken from the output of the tranfoemer and is fed to a logistic regression model whose parameters are optimized in gridsearch.  
[DistilBERT_for_movie_review_sentiment](https://github.com/mayanksinghkgp/NLP_Resources/tree/main/DistilBERT_for_movie_review_sentiment)  

**5. FastText Multi-Label classification**  
- Multi-Label classification is performed on  A [kaggle dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) for the study of negative online behaviors, like toxic comments (i.e. comments that are rude, disrespectful or otherwise likely to make someone leave a discussion). The task involved classification into different tags like toxic, obscene, threat etc. FastText model is used for the classification with the help of ktrain library.  
[FastText_MultiLabel](https://github.com/mayanksinghkgp/NLP_Resources/tree/main/FastText_MultiLabel)  

**6. Named Entity Recognition using LSTM**  
- Named Entity Recognition model is trained from an existing dataset with NER tags (IOB). A few data preprocessing steps were performed on the dataset with close to 50k sentences and 17 different NE tags. Pandas dataframe is used for the initial cleaning and further preprocessing (tokenization, padding etc.) has been performed using Keras. The model is built with an embedding layer and a bidirectional LSTM and Spatial dropout has been used for the regularization. Finally the model output is through a softmax layer which gives the class with max probability.  
[NER_using_LSTMs](https://github.com/mayanksinghkgp/NLP_Resources/tree/main/NER_using_LSTMs)  

**n. Test_models**  
Trail models for different concepts  
