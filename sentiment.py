from keras import Sequential
from keras.layers import Dense, Input
from sklearn.utils import shuffle
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import pandas as pd

DATASET_ENCODING = "ISO-8859-1"

df = pd.read_csv("data/dataset.zip", encoding = DATASET_ENCODING)
df= df.iloc[:,[0,-1]]
df.columns = ['sentiment','tweet']
df = pd.concat([df.query("sentiment==0").sample(20000),df.query("sentiment==4").sample(20000)])
df.sentiment = df.sentiment.map({0:0,4:1})
df = shuffle(df).reset_index(drop=True)

df,df_test = train_test_split(df,test_size=0.2)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def vectorize(df):
    embeded_tweets = embed(df['tweet'].values.tolist()).numpy()
    targets = df.sentiment.values
    return embeded_tweets,targets

embeded_tweets,targets = vectorize(df)

model = Sequential()
model.add(Input(shape = (512,), dtype = 'float32'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['acc'])

num_epochs = 10
batch_size = 32   ## 2^x

history = model.fit(embeded_tweets, 
                    targets, 
                    epochs=num_epochs, 
                    validation_split=0.1, 
                    shuffle=True,
                    batch_size=batch_size)

from sklearn.metrics import accuracy_score

embed_test,targets = vectorize(df_test)
predictions = model.predict(embed_test).astype(int)

print("Accuracy Score: ", accuracy_score(predictions,targets)*100)

model.save('./checkpoints/checkpoint')
