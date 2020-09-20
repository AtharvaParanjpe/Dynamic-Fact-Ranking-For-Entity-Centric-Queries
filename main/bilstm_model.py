import pandas as pd
import numpy as np
import re
import numpy as np
import tensorflow as tf
import math

class MyModel2(tf.keras.Model):
    def __init__(self, vocab_size):
        super(MyModel2, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=(512),embeddings_initializer="glorot_normal",mask_zero=True)
        self.bidirectional_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True))
        self.dense_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def __call__(self, q, p, o):

        ## embedding layer
        q_token_embeddings = self.embedding_layer(q)
        p_token_embeddings = self.embedding_layer(p)
        o_token_embeddings = self.embedding_layer(o)        

        ## reshaping before feeding to Bi-LSTM
        q_token_embeddings = tf.reshape(q_token_embeddings, (1, q_token_embeddings.shape[0], q_token_embeddings.shape[1]))
        p_token_embeddings = tf.reshape(p_token_embeddings, (1, p_token_embeddings.shape[0], p_token_embeddings.shape[1]))
        o_token_embeddings = tf.reshape(o_token_embeddings, (1, o_token_embeddings.shape[0], o_token_embeddings.shape[1]))
        
        ## Bi-LSTM
        queryEmbedding = self.bidirectional_lstm(q_token_embeddings)
        predEmbedding = self.bidirectional_lstm(p_token_embeddings)
        objEmbedding = self.bidirectional_lstm(o_token_embeddings)
        
        ## taking the first and last outputs of lstm layer 
        ## first -> right context
        ## last -> left context
        query_emb = tf.concat([queryEmbedding[:,0,:512],queryEmbedding[:,49,512:]],axis=1)
        pred_emb = tf.concat([predEmbedding[:,0,:512],predEmbedding[:,49,512:]],axis=1)
        obj_emb = tf.concat([objEmbedding[:,0,:512],objEmbedding[:,49,512:]],axis=1)

        ## q.pT  
        scores = tf.matmul(query_emb, tf.transpose(pred_emb))
        # print(scores.shape)
        return scores

## homogeneous encoding for all the queries,predicates, objects
maxLengthPadding = 50


## split the words in predicates and objects
def splitCamelCasing(camelCasedWord):
    camelCaseSplit = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', camelCasedWord)).split()
    return camelCaseSplit

## different datasets

facts = pd.read_csv('../data/fact_ranking_coll.tsv', delimiter='\t', encoding='utf-8')

facts = facts.drop(["id", "qid", "rel", "en_id"], axis=1)

## To give a higher importance to a higher utility value

def invertRanks(rankArray):
  maximum = 2 
  for i in range(len(rankArray)):
    rankArray[i]=rankArray[i]/maximum
  return rankArray


ENCODING_DIM = 50
maxRank = 0
y = invertRanks(list(facts['imp'].values))
print(y[10:100])

embeddings_dictionary = {}

## Creates an encoding for the entire wordList
def encode_sequence(wordList):
    encodedValue = []  ## dimension of embedding
    for word in wordList:
        encodedValue += [embeddings_dictionary[word]]
    ## padding the remaining values to create a homogeneous encoding
    encodedValue += [0] * (maxLengthPadding - len(encodedValue))
    return encodedValue


################################################# Preparing the Embedding Space / Dictionary of Words ############################################


############# 1. Preparing the embedding layer ###############

x = []
queryWords = []
adjustedPredicates = []
objWords = []

for i in range(len(facts["pred"])):
    ## splitting the query
    tempName = re.sub('[^a-zA-Z0-9 \n\.]', '', facts["query"].values[i].strip())
    queryWords += tempName.lower().strip().split(" ")

    ## splitting the object
    oW = facts["obj"].values[i].lower().strip()
    if ('<' in oW and '>' in oW and 'dbp' in oW):
        oW = oW.split(":")
        oW = oW[1].split(">")
        oW = oW[0].split('_')
    elif ('www' in oW):
        oW = oW.split(".")
    else:
        oW = oW.split(" ")

    ## for removing special characters
    oW = [re.sub('[^a-zA-Z0-9 \n\.]', '', x) for x in oW]
    objWords += oW
    
    #splitting the predicates
    pred = facts["pred"].values[i].lower().split(":")
    pred = pred[1].split(">")
    pred = splitCamelCasing(pred[0])
    pred = [re.sub('[^a-zA-Z0-9 \n\.]', '', x) for x in pred]
    adjustedPredicates += pred

totalWords = queryWords + adjustedPredicates + objWords
totalWords = set(totalWords)
vocab_size = len(totalWords)

## creating the dictionary
for i, x in enumerate(totalWords):
    embeddings_dictionary[x] = i

### embedding the tokens

queryWordEncodings = []
predWordEncodings = []
objWordEncodings = []


for i in range(len(facts["pred"].values)):

    name = re.sub('[^a-zA-Z0-9 \n\.]', '', facts["query"].values[i])
    queryWords = name.lower().strip().split(" ")
    queryWordEncodings += [encode_sequence(queryWords)]
    
    ## similar to creating the dictionary, but this time it encodes the words using the values from dictionary
    oW = facts["obj"].values[i].lower().strip()
    if ('<' in oW and '>' in oW and 'dbp' in oW):
        oW = oW.split(":")
        oW = oW[1].split(">")
        oW = oW[0].split('_')
    elif ('www' in oW):
        oW = oW.split(".")
    else:
        oW = oW.split(" ")

    ## for removing special characters
    oW = [re.sub('[^a-zA-Z0-9 \n\.]', '', x) for x in oW]
    oW = encode_sequence(oW)
    objWordEncodings += [oW]

    ## encoding the words of predicates
    pred = facts["pred"].values[i].lower().split(":")
    pred = pred[1].split(">")
    pred = splitCamelCasing(pred[0])
    pred = [re.sub('[^a-zA-Z0-9 \n\.]', '', x) for x in pred]
    predWordEncodings += [encode_sequence(pred)]

# print(len(queryWordEncodings[0]),len(objWordEncodings[0]),len(predWordEncodings[0]))

## defining custom model parameters as per tensorflow2 guidelines

model_2 = MyModel2(vocab_size)
loss_obj = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam( learning_rate=0.0001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')

## x_tr : training x -> contains the query data, predicate data, object data
## y_tr : training y -> contains the normalized targets
def train_step(x_tr,y_tr):
    qE = x_tr[0]
    pE = x_tr[1]
    oE = x_tr[2]
    batch_loss = 0
    with tf.GradientTape() as tape:
        for i in range(len(qE)):
            q= tf.convert_to_tensor(qE[i], dtype=tf.int32)
            p = tf.convert_to_tensor(pE[i], dtype=tf.int32)
            o = tf.convert_to_tensor(oE[i], dtype=tf.int32)
            predictions = model_2(q, p, o)
            loss = loss_obj(y_tr[i],predictions)
            batch_loss+=loss
        gradients = tape.gradient(batch_loss/len(qE), model_2.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model_2.trainable_variables))
    return batch_loss

## train test split step

from sklearn.utils import shuffle

EPOCHS = 10

data_len = int(0.8*len(queryWordEncodings))
# data_len = len(queryWordEncodings)
queryWordEncodingsTrain,predWordEncodingsTrain,objWordEncodingsTrain,y_train = shuffle(queryWordEncodings[:data_len],objWordEncodings[:data_len],predWordEncodings[:data_len],y[:data_len])
queryWordEncodingsTest,predWordEncodingsTest,objWordEncodingsTest,y_test = shuffle(queryWordEncodings[data_len:],objWordEncodings[data_len:],predWordEncodings[data_len:],y[data_len:])



batch_size = 16

num_batches_train = math.floor(data_len/batch_size)

test_size = len(queryWordEncodings)
num_batches_test = math.floor(test_size/batch_size)


print("Num Batches Train:",num_batches_train)
print(data_len/batch_size)
print(len(y_train))

# ###### To check the number of predicates not appearing in the train set
# count = 0

# for p in predWordEncodingsTest:
#   if(p in pWTrainSet):
#     count+=1
#     pWTrainSet.append(p)
# print(count)

## main epoch analysis

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  loss_per_epoch = 0

  for i in range(num_batches_train):
    x_train = [queryWordEncodingsTrain[i*batch_size:(i+1)*batch_size],predWordEncodingsTrain[i*batch_size:(i+1)*batch_size],objWordEncodingsTrain[i*batch_size:(i+1)*batch_size]]
    loss1 = train_step(x_train,y_train[i*batch_size:(i+1)*batch_size])
    loss_per_epoch += loss1
  x_train = [queryWordEncodingsTrain[num_batches_train*batch_size:],predWordEncodingsTrain[num_batches_train*batch_size:],objWordEncodingsTrain[num_batches_train*batch_size:]]
  loss1 = train_step(x_train,y_train[num_batches_train*batch_size:])
  loss_per_epoch += loss1
  print("Epoch Loss for epoch "+str(epoch)+": ",loss_per_epoch/(num_batches_train+1) )

## grouping the dataset by query to get the ndcg scores

def modifiedInvertRanks(rankArray):
  for rank in rankArray:
    maximum = 2 
    for j in range(len(rank)):
      rank[j]=rank[j]/maximum
    
  return rankArray

modified_y = []
groupedResults = facts.groupby('query')

for name, group in groupedResults:
    g = list(group['imp'].values)
    # print(len(g))
    modified_y.append(g)
modified_y = modifiedInvertRanks(modified_y)


queryWordEncodings = []
predWordEncodings = []
objWordEncodings = []

predictedQueryRanks = []
groundTruthRanks = []
integerValuedQueryRanks = []

count = 0

for query, group in groupedResults:

    ## name is the query used for grouping
    query = re.sub('[^a-zA-Z0-9 \n\.]', '', query)
    queryWords = query.lower().strip().split(" ")
    q = encode_sequence(queryWords)
    predictedRanksPerQuery = []
    groundTruthRanksPerQuery = []

    ## encoding predicate-objects
    i_ranks = []
    for i in range(len(group["obj"].values)):
        oW = group["obj"].values[i].lower().strip()
        if ('<' in oW and '>' in oW and 'dbp' in oW):
            oW = oW.split(":")
            oW = oW[1].split(">")
            oW = oW[0].split('_')
        elif ('www' in oW):
            oW = oW.split(".")
        else:
            oW = oW.split(" ")

        ## for removing special characters
        oW = [re.sub('[^a-zA-Z0-9 \n\.]', '', x) for x in oW]
        o = encode_sequence(oW)
        
        pred = group["pred"].values[i].lower().split(":")
        pred = pred[1].split(">")
        pred = splitCamelCasing(pred[0])
        pred = [re.sub('[^a-zA-Z0-9 \n\.]', '', x) for x in pred]
        p = encode_sequence(pred)
        
        q= tf.convert_to_tensor(q, dtype=tf.int32)
        p = tf.convert_to_tensor(p, dtype=tf.int32)
        o = tf.convert_to_tensor(o, dtype=tf.int32)
        
        predictions = model_2(q, p, o)
        predictedRanksPerQuery.append(predictions.numpy().tolist()[0][0])
        i_ranks.append(round(predictions.numpy().tolist()[0][0]*2))
    # i_ranks = [x if x < 4 else 4 for x in i_ranks]
    integerValuedQueryRanks.append(i_ranks)

    g_t = [x*2 for x in modified_y[count]]
    groundTruthRanks.append(g_t)
    predictedQueryRanks.append(predictedRanksPerQuery)
    count+=1

print(groundTruthRanks[0])

## NDCG implementation
from sklearn.metrics import ndcg_score, dcg_score 
 
ndcg_scores_5 = []
ndcg_scores_10 = []
count = 0
for x,y in zip(groundTruthRanks,integerValuedQueryRanks):
    
    
    if(len(x)>1):
        true_relevance = np.asarray([x]) 
        relevance_score = np.asarray([y]) 

        ndcg_scores_5.append(ndcg_score(true_relevance, relevance_score,k=5))
        ndcg_scores_10.append(ndcg_score(true_relevance, relevance_score,k=10))
    

print(sum(ndcg_scores_5)/len(ndcg_scores_5))
print(sum(ndcg_scores_10)/len(ndcg_scores_10))

