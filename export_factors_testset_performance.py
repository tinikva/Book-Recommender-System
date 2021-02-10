#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
import json
from als_model import groundTruth, predictionsPerUser, predictionsAndLabels, evaluate

def main(spark):
        #Export user and item factors learned by the best model learned on 1%
        best_als_model = ALSModel.load('hdfs:/user/mdv325/final_project/models/model_20200508093345')

        user_factors = best_als_model.userFactors.toPandas()
        users = list(user_factors['features'])
        with open('users.json', 'a') as ur:
            json.dump(users, ur)
        item_factors = best_als_model.itemFactors.toPandas()
        items = list(item_factors['features'])
        with open('items.json', 'a') as it:
            json.dump(items, it)

        #Evaluate the performance of best model learned on 100% on test set
        best_als_model = ALSModel.load('hdfs:/user/mdv325/final_project/models/model_20200509164104')
        test = spark.read.parquet('hdfs:/user/mdv325/final_project/train_data/100_percent/test_interactions')
        targetsPerUser = groundTruth(test)
        preds =  best_als_model.transform(test) # Intentionally to get a ranking for **all** the user items in val but will evaluate only on ground truth
        predsPerUser = predictionsPerUser(preds)
        evalRes = evaluate(predsPerUser, targetsPerUser)
        print(evalRes)

if __name__ == "__main__":
        spark = SparkSession.builder.appName('get_data').getOrCreate()
        main(spark)
