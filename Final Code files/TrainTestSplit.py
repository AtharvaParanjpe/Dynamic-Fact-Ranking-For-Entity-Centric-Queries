import numpy as np
import pandas as pd
from sklearn.utils import shuffle

filename = './Complete_Data_With_Targets.csv'
train = open("./CompleteData/Utility/TrainingData4.txt", "w")

test = open("./CompleteData/Utility/TestingData4.txt", "w")


def getPairsWithLowRanking(query,pred,obj,target,groupedDataset):
    pairs = []
    for q,group in groupedDataset:
      if(query==q):
        for i in range(len(group['pred'].values)):
            if(group['utility'].values[i]<target):
                train.write(query+','+pred+','+obj+','+str(target)+','+
                query+','+group['pred'].values[i]+','+group['obj'].values[i]+','+str(group['utility'].values[i]))
                train.write("\n")
    

df = pd.read_csv(filename)


groupedDataset = df.groupby('query')


queries = []
for q,g in groupedDataset:
  queries.append(q)

queries = shuffle(queries)

data_length = len(groupedDataset)
split_length = int(0.8*data_length)

queries_for_training = queries[:split_length]
queries_for_testing = queries[split_length:]


for query,group in groupedDataset:
    if(query in queries_for_testing):
        for j in range(len(group['pred'].values)):
            test.write(query+','+group['pred'].values[j]+','+group['obj'].values[j]+','+str(group['utility'].values[j]))
            test.write("\n")
            
    else:
        for i in range(len(group['pred'].values)):
            getPairsWithLowRanking(query,group['pred'].values[i],group['obj'].values[i],group['utility'].values[i],groupedDataset)

train.close()
test.close()


