#!/usr/bin/env python
# -*- coding: utf-8 -*-

#### USAGE ######
# python als_model.py -t <dataName>
# Fits and train model with the dataname entered (should exist under .../final_project/train_data/)

import itertools
import sys, getopt
import time

from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import Window
from pyspark.sql.functions import *
from training_data import read_train_data, get_user, newSparkSession

##CONSTANTS##
MODEL_PATH_TEMPLATE = "final_project/models/model_{}"
RESULTS_PATH_TEMPLATE = "/home/{}/final-project-quaranteam/results/als_{}_{}.json" ## This is saving locally, not in hdfs!
SEED = 20

def getTimeStamp():
	timestamp = time.strftime("%Y%m%d%H%M%S")
	return timestamp
		
def loadModel(modelFileName):
	return ALSModel.load(modelFileName)
	
def saveModel(model):
	fileName = MODEL_PATH_TEMPLATE.format(getTimeStamp())
	model.save(fileName)
	return fileName

	
def groundTruth(userData):
	w = Window.partitionBy('user_id').orderBy(col('rating').desc())
	return userData.filter('rating = 4 or rating = 5').select('user_id', 'book_id', collect_list('book_id').over(w).alias('books')).groupBy('user_id').agg(max('books').alias('books')).withColumn('book_count',size('books'))
	
def predictionsPerUser(modelPredictions):
	wPred = Window.partitionBy('user_id').orderBy(col('prediction').desc())
	return modelPredictions.select('user_id', 'book_id', collect_list('book_id').over(wPred).alias('books')).groupBy('user_id').agg(max('books').alias('books')).withColumn('book_count',size('books'))

def predictionsAndLabels(predsPerUser, targetsPerUser):
	predsAndLabels = predsPerUser.join(targetsPerUser.select(col('user_id'),col('books').alias('books_t'),col('book_count').alias('book_count_t')), 'user_id', 'inner')
	return predsAndLabels
	
def evaluate(predsPerUser, targetsPerUser):
	limit = 500
	predsAndLabels = predictionsAndLabels(predsPerUser, targetsPerUser)
	rdd = predsAndLabels.rdd.map(lambda r: (r[1][:r[4]], r[3]) if r[4] < limit else (r[1][:limit], r[3][:limit]))
	rm =  RankingMetrics(rdd)
	return rm.meanAveragePrecision

def initResults():
	return {'GridSearch': [] }
	
def collectGridSearchResults( results, rank, regParam, maxIter, evalRes, trainingDataName, modelFileName, timeElapsed):
	results.get('GridSearch').append({'rank': rank,
			    	          'regParam' : regParam, 
					  'maxIter'  : maxIter,
					  'map'	     : evalRes,
					  'subsample': trainingDataName,
					  'modelFileName' : modelFileName,
					  'timeElapsed'   : timeElapsed})
										 
def saveResults(name, results):	
	import json
	try:
		fileName  = RESULTS_PATH_TEMPLATE.format(get_user(), name, getTimeStamp())
		with open(fileName, 'w') as f:
			json.dump(results, f)
		print("Output saved in: {}".format(fileName))
	except Exception as e:
		print(e)
		print(results)
		
def train(trainingDataName, maxIter = 10):
	
	spark = newSparkSession()	
	train, val, _ = read_train_data(trainingDataName) # eg: trainingDataName = '5_percent'
	targetsPerUser = groundTruth(val)
	
	rank = [ 5, 10, 20, 30, 50 ]
	regParam = [ .001, .01, .1, 1 ]


	allResults = initResults()
	paramGrid = itertools.product(rank, regParam)

	for rank,regParam in paramGrid:

		t_start = time.process_time()
		als = ALS(rank = rank, maxIter = maxIter, regParam = regParam, userCol = 'user_id', itemCol = 'book_id', ratingCol = "rating", coldStartStrategy = "drop", seed = SEED)
		model = als.fit(train)
		preds =  model.transform(val) # Intentionally to get a ranking for **all** the user items in val but will evaluate only on ground truth
		predsPerUser = predictionsPerUser(preds)				
		evalRes = evaluate(predsPerUser, targetsPerUser) # At the moment returns only our customized MAP, do we need more?
		t_end = time.process_time()		

		modelFileName = saveModel(model)
		collectGridSearchResults(allResults, rank, regParam, maxIter, evalRes, trainingDataName, modelFileName, t_end-t_start) #TODO: check time elapsed
		
	saveResults(trainingDataName, allResults)
	
		
def main(argv):

	try:
		opts, args = getopt.getopt(argv, 't:' )
	except getopt.GetoptError as err:
		print(err)
	
	for option,arg in opts:
		if option == '-t':
			print("Training...")
			train(str(arg))
			

if __name__ == "__main__":
	main(sys.argv[1:])
		
