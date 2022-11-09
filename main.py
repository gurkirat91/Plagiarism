import nltk
import re
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import string
from string import punctuation
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
punctuation = punctuation + '\n' + '—' + '“' + ',' + '”' + '‘' + '-' + '’'

ps=PorterStemmer()
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    # print(token_words)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(ps.stem((lemmatizer.lemmatize(word,get_wordnet_pos(word)))))  ##for lem and Stem
        stem_sentence.append(" ")
    return "".join(stem_sentence)

# text="i am gurkirat singh. i study in thapar institute . COE is my stream"
# sentences = nltk.sent_tokenize(text) #sentence toS. entence segmentation (Seg)
# sen=[]

#function to do segmentation , Stemming ,Lemmatisation............
def SEG_Stem_LEM(text):
  sentences = nltk.sent_tokenize(text)
  sen=[]
  for i in sentences:
    x=stemSentence(i)
    sen.append(x)

  return "".join(sen)

#.................................................................
import pandas as pd
import numpy as np

df=pd.read_csv("articles1.csv")
# print(df.head())

df.drop(columns = ['Unnamed: 0'], inplace = True)

# Function to clean the html from the article
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


def preprocessing(article):
    global article_sent

    # Converting to lowercase
    article = article.str.lower()

    # Removing the HTML
    article = article.apply(lambda x: cleanhtml(x))

    # Removing the email ids
    article = article.apply(lambda x: re.sub('\S+@\S+', '', x))

    # Removing The URLS
    article = article.apply(lambda x: re.sub(
        "((http\://|https\://|ftp\://)|(www.))+(([a-zA-Z0-9\.-]+\.[a-zA-Z]{2,4})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}))(/[a-zA-Z0-9%:/-_\?\.'~]*)?",
        '', x))

    # Removing the '\xa0'
    article = article.apply(lambda x: x.replace("\xa0", " "))

    # Removing the contractions
    # article = article.apply(lambda x: expand_contractions(x))
    ##################### SEG_STEM_LEM funciton  ###########################################
    #VVVVVIMP......
    article=article.apply(lambda x : SEG_Stem_LEM(x))
    ########################################################################################
    # Stripping the possessives
    article = article.apply(lambda x: x.replace("'s", ''))
    article = article.apply(lambda x: x.replace('’s', ''))
    article = article.apply(lambda x: x.replace("\'s", ''))
    article = article.apply(lambda x: x.replace("\’s", ''))

    # Removing the Trailing and leading whitespace and double spaces
    article = article.apply(lambda x: re.sub(' +', ' ', x))

    # Copying the article for the sentence tokenization
    article_sent = article.copy()

    # Removing punctuations from the article
    article = article.apply(lambda x: ''.join(word for word in x if word not in punctuation))

    # Removing the Trailing and leading whitespace and double spaces again as removing punctuation might
    # Lead to a white space
    article = article.apply(lambda x: re.sub(' +', ' ', x))

    # Removing the Stopwords
    article = article.apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

    return article

df=pd.read_csv('articles1.csv')

df=df.head(3)
auth=df['author']
df=df['content']
df=preprocessing(df)
content=[]
aut=[]
for i in df:
    content.append(i)
for i in auth:
    aut.append(i)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import manhattan_distances,cosine_similarity
def vectorize(text):
    return TfidfVectorizer().fit_transform(text).toarray()
def similarity(doc1,doc2):
    return cosine_similarity([doc1,doc2])

vectors=vectorize(content)
# print(vectors.shape)
s_vector=list(zip(aut,vectors))
#
# print(s_vector[1])
#
plag_res=set()
# def plag_check():
#     global s_vector
#     for author, text_a in s_vector:
#         new_vector=s_vector.copy()
#         cur_idx=new_vector.index((author,text_a))
#         # print(cur_idx)
#         for author2, text_b in new_vector:
#             sim_score=similarity(text_a,text_b)[1][0]
#             print(f"{author} and {author2} ={sim_score}")
#             # author_pair=sorted((author,author2))
#             # print(similarity(text_a,text_b))
#             # print(author_pair)
#             # score=(author_pair[0],author_pair[1],sim_score)
#             # plag_res.add(score)
#     # return plag_res
#
#working.......
# def plag_check():
#     global s_vector
#     for author, text_a in s_vector:
#         new_vector=s_vector.copy()
#         cur_idx=new_vector.index((author,text_a))
#         # print(cur_idx)
#         for author2, text_b in new_vector:
#             if author!=author2:
#                 sim_score=similarity(text_a,text_b)[1][0]
#                 print(f"{author} comparison with  {author2} ={sim_score}")

# def plag_check():
#     flag=0
#     global s_vector
#     for author, text_a in s_vector:
#         new_vector=s_vector.copy()
#         cur_idx=new_vector.index((author,text_a))
#         # print(cur_idx)
#         for author2, text_b in new_vector:
#             if author!=author2 and flag<2:
#                 sim_score=similarity(text_a,text_b)[1][0]
#                 print(f"{author} comparison with  {author2} ={sim_score}")
#                 flag+=1
def plag_check(s_vector):
    flag=0
    scores=[]
    sim_auth=[]

    # global s_vector
    for author, text_a in s_vector:
        new_vector=s_vector.copy()
        # cur_idx=new_vector.index((author,text_a))
        # print(cur_idx)
        for author2, text_b in new_vector:
            if author!=author2 and flag<len(s_vector)-1:
                sim_score=similarity(text_a,text_b)[1][0]
                # print(f"{author} comparison with  {author2} ={sim_score}")
                scores.append(sim_score)
                comp_str=f"{author} comparison with  {author2}"
                sim_auth.append(comp_str)
                flag+=1

    return scores,sim_auth
# plag_check()
# for data in plag_check():
#     print(data)






