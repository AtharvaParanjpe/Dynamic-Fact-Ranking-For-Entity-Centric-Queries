
import pandas as pd
import numpy as np
import re
import keras
import numpy as np
import tensorflow as tf
import math

from sklearn.utils import shuffle
from sklearn.metrics import ndcg_score, dcg_score 

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
        self.dense_layer = tf.keras.layers.Dense(1, activation="sigmoid")
        ## 50 x 768

    def __call__(self, x_train):
        intermediate = self.bert_layer(x_train)
        output = self.dense_layer(intermediate[1])
        
        return output

## homogeneous encoding for all the queries,predicates, objects
maxLengthPadding = 50
# Note : maxLengthOfTokens = 48;

input_dictionary = {}

# df = pd.read_csv("/content/drive/My Drive/Independent Study/Modified_Data.csv")
# df = pd.read_csv("/content/drive/My Drive/Independent Study/URI_only.csv")
# df = pd.read_csv("/content/drive/My Drive/Independent Study/Complete_Data_With_Targets.csv")
# df = pd.read_csv("/content/drive/My Drive/Independent Study/URI_with_imp.csv")
df = pd.read_csv("/content/drive/My Drive/Independent Study/URI_with_targets.csv")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def getBertParameters(tokens):
    attn_mask = []
    seg_ids = []
    # pos_ids = []
    if(len(tokens)<maxLengthPadding):
        attn_mask = [1]*len(tokens) + [0]*(maxLengthPadding-len(tokens))
    else:
        attn_mask = [1]*maxLengthPadding
    
    segment = 0
    for x in tokens:
        seg_ids.append(segment)
        if(x=='[SEP]'):
            segment = 1
    seg_ids+=[0]*(maxLengthPadding-len(tokens))
    return attn_mask,seg_ids

## defining custom model parameters as per tensorflow2 guidelines

model = MyModel()

loss_obj = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam( learning_rate=0.00005)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')

groupedDataset = df.groupby('query')
queries = []
for q,g in groupedDataset:
  queries.append(q)

queries = shuffle(queries)

## data length = 95
data_length = len(groupedDataset)
split_length = int(0.8*data_length)

queries_for_training = queries[:split_length]
queries_for_testing = queries[split_length:]


test_data =[]
count = 0

queryTrainList = []
predTrainList = []
objTrainList = []

y_train = []

for query,group in groupedDataset:
    if(query in queries_for_testing):
      test_data.append([query,group["pred"].values,group["obj"].values,group["imp"].values])
    else: 
      for i in range(len(group["pred"].values)):
        queryTrainList.append(query)
        predTrainList.append(group["pred"].values[i])
        objTrainList.append(group["obj"].values[i])
        y_train.append(group["imp"].values[i])
    count+=1

## To give a higher importance to a higher utility value
def normalizeRanks(rankArray):
  maximum = 2
  for i in range(len(rankArray)):
    rankArray[i]=rankArray[i]/maximum
  return rankArray


y_train = normalizeRanks(y_train)



queryTrainList,predTrainList,objTrainList,y_train = shuffle(queryTrainList,predTrainList,objTrainList,y_train)


print(y_train[0:10])    
print(len(queryTrainList),len(predTrainList),len(objTrainList),len(test_data),len(y_train),data_length)

batch_size = 16
num_batches = math.floor(len(queryTrainList)/batch_size)
EPOCHS = 20


def train_step(query,pred,obj,target):
    input_ids = []
    attn_mask = []
    segment_ids = []

    for k in range(len(query)):
          in_id = tokenizer.encode(query[k],pred[k]+" "+obj[k], add_special_tokens=True)  # Batch size 1
          attn,seg = getBertParameters(tokenizer.convert_ids_to_tokens(in_id))
          in_id += [0]*(maxLengthPadding-len(in_id))
          input_ids.append(in_id)
          attn_mask.append(attn)
          segment_ids.append(seg)
          
    loss = 0
    with tf.GradientTape() as tape:
        input_dictionary['input_ids'] = tf.convert_to_tensor(np.array(input_ids)) ## [None, :]
        input_dictionary['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask))  ## [None, :]
        input_dictionary['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids))  ## [None, :]
        output = model(input_dictionary)
        loss = (target-output)**2
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # input()
    return loss
  
        
    
ndcg_epoch_array = {
    
}

## [CLS] + [Query] + [SEP] + [PRED + OBJ] +[SEP]
for j in range(EPOCHS):
    for i in range(num_batches-1):
        index = i*batch_size
        q = queryTrainList[index:index+batch_size]
        p = predTrainList[index:index+batch_size]
        o = objTrainList[index:index+batch_size]
        actual_y = np.array(y_train[index:index+batch_size]).reshape(batch_size,1)
        loss = train_step(q,p,o,actual_y)

    q = queryTrainList[num_batches*batch_size:]
    p = predTrainList[num_batches*batch_size:]
    o = objTrainList[num_batches*batch_size:]
    if(len(q)>0):
        actual_y = y_train[num_batches*batch_size:]
        actual_y = np.array(actual_y).reshape(len(actual_y),1)
        print(actual_y)
        
        loss = train_step(q,p,o,actual_y)
        print("Loss after epoch "+str(j)+" :", loss)   
    n5,n10 = test_step()

    print("n5,n10:",n5,n10)


    ndcg_epoch_array[j] = [n5,n10]

ndcg_epoch_array

# ## grouping the dataset by query to get the ndcg scores
def test_step():
    predictedQueryRanks = []
    groundTruthRanks = []

    integerValuedQueryRanks = []

    count = 0

    for x_test in test_data:
        query = x_test[0]
        predPerQuery = list(x_test[1])
        objPerQuery = list(x_test[2])
        target = list(x_test[3])
        g_t = []
        i_ranks = []
        predictedRanksPerQuery = []
        for i in range(len(predPerQuery)):
            input_ids = tokenizer.encode(query,predPerQuery[i]+" "+objPerQuery[i], add_special_tokens=True)  # Batch size 1
            attn_mask,segment_id = getBertParameters(tokenizer.convert_ids_to_tokens(input_ids))
            input_ids += [0]*(maxLengthPadding-len(input_ids))
            input_dictionary['input_ids'] = tf.constant(input_ids)[None, :]
            input_dictionary['attention_mask'] = tf.constant(attn_mask)[None, :]
            input_dictionary['token_type_ids'] = tf.constant(segment_id)[None, :]
            output = model(input_dictionary)
            predictedRanksPerQuery.append(output.numpy().tolist()[0][0])
            i_ranks.append(round(output.numpy().tolist()[0][0]*2))
        integerValuedQueryRanks.append(i_ranks)

        groundTruthRanks.append(target) 
        predictedQueryRanks.append(predictedRanksPerQuery)
        count+=1
    print(groundTruthRanks[0])
    print(integerValuedQueryRanks[0])
    
    return compute_ndcg_scores(groundTruthRanks,integerValuedQueryRanks)



def compute_ndcg_scores(groundTruthRanks,integerValuedQueryRanks): 
    ndcg_scores_5 = []
    ndcg_scores_10 = []
    count = 0
    for x,y in zip(groundTruthRanks,integerValuedQueryRanks):
        if(len(x)>1):
            true_relevance = np.asarray([x]) 
            relevance_score = np.asarray([y]) 
            ndcg_scores_5.append(ndcg_score(true_relevance, relevance_score,k=5))
            ndcg_scores_10.append(ndcg_score(true_relevance, relevance_score,k=10))
    return (sum(ndcg_scores_5)/len(ndcg_scores_5)),(sum(ndcg_scores_10)/len(ndcg_scores_10))

