
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import string


#storing the vaccine data set in df

df=pd.read_csv(r"C:\Users\Ajit\Desktop\covid.csv")
print('\nOriginal data set is -: ')
print(df)
#importing nltk and wordcloud module


from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud,STOPWORDS
SIA=SentimentIntensityAnalyzer()


#creating function to clean the data(removing unwanted things from tweets)


def clean(text):
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('<.*?>+', '', text)
    return text

#applying the clean function to the column containing tweets
df['text'] = df['text'].apply(lambda x:clean(x))



# Classifing tweets as pos,neg,neu and storing it to the new column 'sentiment' 


scores=[]
for i in range(len(df['text'])):
    
    score = SIA.polarity_scores(df['text'][i])
    score=score['compound']
    scores.append(score)
sentiment=[]
for i in scores:
    if i>=0.05:
        sentiment.append('Positive')
    elif i<=(-0.05):
        sentiment.append('Negative')
    else:
        sentiment.append('Neutral')
df['sentiment']=pd.Series(np.array(sentiment))
df[['text','sentiment']]


# strip punctuation from tweets(We are doing this later because it may change sentiment)

def clean_text(text):
    text = str(text).lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    return text
df['text'] = df['text'].apply(lambda x:clean_text(x))
df[['text','sentiment']]


#creating new data set which just contains tweets and corresponding sentiment value
#removing duplicate tweets

tweets=df[['text','sentiment']].drop_duplicates()

print('\nData set after cleaning and applying sentiment analysis(Only tweets and their sentiment values -: \n)')
print(tweets)
#Creating a new columns,which contain tokenization of corresponding tweets

from nltk import tokenize
def tokenization(a):
    return tokenize.word_tokenize(a)

tweets['tokenized'] = tweets['text'].apply(lambda x: tokenization(x.lower()))
tweets


#importing stopword and adding some stopword to it from our side


stopword = nltk.corpus.stopwords.words('english')
stopword.extend(['covid','vaccine','vaccines','dose','doses','vaccinated','vaccination','pfizer','covidvaccine','amp','first','today','corona','coronavirus','got','day','covidvaccination','second','take','took','jab','jabs','pfizerbiontech'])

#removing stopwords from tweets and storing rest of word to new column 'No_stopword'
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
tweets['No_stopwords'] = tweets['tokenized'].apply(lambda x: remove_stopwords(x))

print('\nTweets after tokenization and removing stopwords(we add tokenization and No_stopword columns to it) -: \n ')
print(tweets)
#Pie_Chart
tags=tweets['sentiment'].value_counts().sort_values(ascending=False)
tags=dict(tags)
 
pos_=tags['Positive']
neg_=tags['Negative']
neu_=tags['Neutral']

#total useful tweets
sum_=pos_+neg_+neu_
print('\nTotal no. of tweets after cleaning -: ',end='')
print(sum_,'\n\n')
print(tweets['sentiment'].value_counts())

def percentage(k):
    return (k/sum_)*100
slices_tweets = [percentage(pos_), percentage(neg_),percentage(neu_)]
    
analysis = ['Positive', 'Negative', 'Neutral']
colors = ['g', 'r', 'y']

plt.pie(slices_tweets, labels=analysis,explode = (0.03, 0.03, 0.03) ,shadow=False, autopct='%1.1f%%') #to generate the pie chart

plt.savefig(r'C:\Users\Ajit\Desktop\pie_chart.png',format='png',dpi=600)
plt.show() #to disply the generated chart


#grouping tweets based on positive, negative, neutral value
temp=tweets.groupby('sentiment')
def merge(x):
    return ' '.join(i for i in x)


temp1=temp.get_group('Positive').No_stopwords.apply(lambda x: merge(x))


text1 = ",".join(review for review in temp1)
wordcloud = WordCloud(width=1600, height=800,max_words=100, colormap='Set2',background_color="black").generate(text1)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.title('WordCloud for Positive Tweets',fontsize=19)
plt.savefig(r'C:\Users\Ajit\Desktop\positive_wordcloud.png',format='png',dpi=600)
plt.show()


#Wordcloud of Negative Tweets
temp2=temp.get_group('Negative').No_stopwords.apply(lambda x: merge(x))



text2 = ",".join(review for review in temp2)
wordcloud = WordCloud(width=1600, height=800, max_words=100, colormap='Set2',background_color="black").generate(text2)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.title('WordCloud for Negative Tweets',fontsize=19)
plt.savefig(r'C:\Users\Ajit\Desktop\negative_wordcloud.png',format='png',dpi=600)
plt.show()

#Wordcloud of Neutral Tweets
temp3=temp.get_group('Neutral').No_stopwords.apply(lambda x: merge(x))

text3 = ",".join(review for review in temp3)
wordcloud = WordCloud(width=1600, height=800, max_words=100, colormap='Set2',background_color="black").generate(text3)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.title('WordCloud for Neutral Tweets',fontsize=19)
plt.savefig(r'C:\Users\Ajit\Desktop\neutral_wordcloud.png',format='png',dpi=300)
plt.show()


#Wordcloud of all Tweets
text4 = text1+' '+text2+' '+text3
wordcloud = WordCloud(width=1600, height=800, max_words=100, colormap='Set2',background_color="black").generate(text4)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.title('WordCloud for all Tweets',fontsize=19)
plt.savefig(r'C:\Users\Ajit\Desktop\all_tweets_wordcloud.png',format='png',dpi=300)
plt.show()



