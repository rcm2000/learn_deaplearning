from sklearn.feature_extraction.text import TfidfVectorizer

doc1 = 'she likes python'
doc2 = 'she hates python'

tfv = TfidfVectorizer();
result = tfv.fit_transform([doc1,doc2]).toarray();

