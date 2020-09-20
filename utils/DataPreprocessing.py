import pandas as pd
import numpy as np
import re

## Data Processing for the Complete Dataset including the URI objects
## Using the fact_ranking_coll.tsv
class DataPreprocess:
    def __init__(self,pathToFile):
        dataFrame = self.getData(pathToFile)
        dataFrame = self.cleanData(dataFrame)
        self.convertToExcel(dataFrame)
        self.printParams(dataFrame)

    def printParams(self,df):
        groupedDataset = df.groupby('query')
        ranksPerQuery = []
        for q,g in groupedDataset:
            ranksPerQuery.append(len(g))
        print("Number of queries: ",len(ranksPerQuery))
        print("Total Facts: ",sum(ranksPerQuery))
        print("Average facts per query:",sum(ranksPerQuery)/len(ranksPerQuery))



    def _splitCamelCasing(self,sentence):
        words = sentence.split(" ")
        for i in range(len(words)):
            split = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', words[i])).split()
            words[i] = " ".join(split)
        return " ".join(words)
    
    def getData(self,filePath):
        df = pd.read_csv(filePath,delimiter='\t', encoding='utf-8')
        df = df.drop(["id", "qid","en_id"],axis=1)
        return df
    
    def _dataModifierForObjects(self,data):
        if(not re.search('[a-zA-Z]', data)):
            return "NUM"
        elif(re.search('[0-9]',data)):
            return "ALPHANUM"
        data = re.sub("[^A-Za-z0-9]", " ",data)
        if("dbo" in data):
            data = data.replace("dbo", "")
        if("dbpedia" in data):
            data = data.replace("dbpedia", "")
        if("dbp" in data):
            data = data.replace("dbp", "")
        data = self._splitCamelCasing(data)
        return data.lower()

    def _dataModifier(self,data):
        data = re.sub("[^A-Za-z0-9]", " ",data)
        if("dbo" in data):
            data = data.replace("dbo", "")
        if("dbpedia" in data):
            data = data.replace("dbpedia", "")
        if("dbp" in data):
            data = data.replace("dbp", "")
        data = self._splitCamelCasing(data)
        return data.lower()
    
    def cleanData(self,df):
        for i in range(len(df.pred.values)):
            query = self._dataModifier(df['query'].values[i])
            df['query'].values[i] = query.strip()
            pred = self._dataModifier(df.pred.values[i])
            df.pred.values[i] = pred.strip()
            obj = self._dataModifierForObjects(df.obj.values[i])
            df.obj.values[i] = obj.strip()
        return df
    
    def convertToExcel(self,df):
        df.to_csv("../data/Complete_Data_With_Targets.csv",index = False)


dataPreprocess = DataPreprocess("../data/fact_ranking_coll.tsv")

        


