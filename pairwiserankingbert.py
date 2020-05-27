import pandas as pd
import numpy as np
import tensorflow as tf
import math

from sklearn.utils import shuffle

from transformers import BertTokenizer, TFBertModel,TFBertForSequenceClassification

# tf.config.experimental.list_physical_devices('GPU')

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

maxLengthPadding = 50



train_data = open("./URI_5_fold_data/TrainingDataURI4.txt",'r')
# train_data = open("/content/drive/My Drive/Independent Study/Shuffled_Complete_Training_Data.txt",'r')
train_data = train_data.readlines()
train_data = shuffle(train_data)
test_data = open("./URI_5_fold_data/TestingDataURI4.txt",'r')
test_data = test_data.readlines()


for j in range(len(test_data)):
    test_data[j]=test_data[j].split(',')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def getBertParameters(tokens):
    attn_mask = []
    seg_ids = []
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



print(len(train_data),len(test_data))


model = MyModel()

loss_obj = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam( learning_rate=0.00005)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_accuracy')


batch_size = 32
num_batches = math.floor(len(train_data)/batch_size)
EPOCHS = 20


def train_step(data):
    maximum = 4
    input_ids1 = [] 
    attn_mask1 = []
    segment_ids1 = []

    input_ids2 = [] 
    attn_mask2 = []
    segment_ids2 = []
    input_dictionary = {}

    target_1 = []
    target_2 = []

    for k in range(len(data)):
        data[k] = data[k].split(',')
        
        in_id1 = tokenizer.encode(data[k][0],data[k][1]+" "+data[k][2], add_special_tokens=True)  # Batch size 1
        attn1,seg1 = getBertParameters(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))
        input_ids1.append(in_id1)
        attn_mask1.append(attn1)
        segment_ids1.append(seg1)
        
        target_1.append(int(data[k][3])/maximum)


        in_id2 = tokenizer.encode(data[k][4],data[k][5]+" "+data[k][6], add_special_tokens=True)  # Batch size 1
        attn2,seg2 = getBertParameters(tokenizer.convert_ids_to_tokens(in_id2))
        in_id2 += [0]*(maxLengthPadding-len(in_id2))
        input_ids2.append(in_id2)
        attn_mask2.append(attn2)
        segment_ids2.append(seg2)
        target_2.append(int(data[k][7])/maximum)


    
    
    target_1 = tf.convert_to_tensor(target_1)
    target_2 = tf.convert_to_tensor(target_2)
    target_1 = tf.reshape(target_1,shape=(target_1.shape[0],1))
    target_2 = tf.reshape(target_2,shape=(target_2.shape[0],1))
          
    loss = 0
    with tf.GradientTape() as tape:
        input_dictionary['input_ids'] = tf.convert_to_tensor(np.array(input_ids1)) ## [None, :]
        input_dictionary['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask1))  ## [None, :]
        input_dictionary['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids1))  ## [None, :]
        output1 = model(input_dictionary)
    
        input_dictionary['input_ids'] = tf.convert_to_tensor(np.array(input_ids2)) ## [None, :]
        input_dictionary['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask2))  ## [None, :]
        input_dictionary['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids2))  ## [None, :]
        output2 = model(input_dictionary)

        target = tf.math.subtract(target_1,target_2)

        # print("Target Shape: ",target.shape)

        output = tf.math.subtract(output1,output2)
        # print("Prediction Shape: ",output.shape)

        result = tf.math.subtract(target,output)
        squaredError = tf.math.square(result)
        
        
        loss = tf.math.divide(tf.math.reduce_sum(squaredError),squaredError.shape[0])
        # print(loss)
        

        # print(loss.shape)
        # loss = (target-output)**2
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
    return loss
  


## grouping the dataset by query to get the ndcg scores
def test_step():
    input_dictionary = {}
    predictedQueryRanks = []
    groundTruthRanks = []
    maximum = 4
    integerValuedQueryRanks = []
    target = []
    i = 0
    while( i < len(test_data)):
        i_ranks = []
        target = []
        count= 0
        query = test_data[i][0]
        while((i+count) < len(test_data) and query==test_data[i+count][0]):
            input_ids = tokenizer.encode(query,test_data[i+count][1]+" "+test_data[i+count][2], add_special_tokens=True)  # Batch size 1
            attn_mask,segment_id = getBertParameters(tokenizer.convert_ids_to_tokens(input_ids))
            input_ids += [0]*(maxLengthPadding-len(input_ids))
            input_dictionary['input_ids'] = tf.constant(input_ids)[None, :]
            input_dictionary['attention_mask'] = tf.constant(attn_mask)[None, :]
            input_dictionary['token_type_ids'] = tf.constant(segment_id)[None, :]
            output = model(input_dictionary)
            i_ranks.append(round(output.numpy().tolist()[0][0]*maximum))
            target.append(int(test_data[i+count][3]))
            count+=1
            # print(count)
        
        i+=count
        integerValuedQueryRanks.append(i_ranks)

        groundTruthRanks.append(target) 
    
    return compute_ndcg_scores(groundTruthRanks,integerValuedQueryRanks)

from sklearn.metrics import ndcg_score, dcg_score 

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



    
ndcg_epoch_array = {
    
}

## [CLS] + [Query] + [SEP] + [PRED + OBJ] +[SEP]
for j in range(EPOCHS):
    for i in range(num_batches-1):
        index = i*batch_size
        curr_batch = train_data[index:index+batch_size]
        loss = train_step(curr_batch)
        # print(i)
    curr_batch = train_data[num_batches*batch_size:]
    if(len(curr_batch)>0):
        loss = train_step(curr_batch)
        print("Loss after epoch "+str(j)+" :", loss)   
    model.save_weights(filepath="/common/users/ap1746/BertModel",overwrite=True)
    
    n5,n10 = test_step()
    
    print("After epoch "+str(j+1)+" n5,n10:",n5,n10)

    ndcg_epoch_array[j] = [n5,n10]
    


print(ndcg_epoch_array)
