#!/usr/bin/env python

#### USAGE ######
# python training_data.py -l     	#Bring source data to user hdfs folder (writes under final_project/all_data)
# python training_data.py -t 0.1 	#Creates all training data under final_project/train_data (subsampling the percentage entered)
# python training_data.py -r 		#Just for sample purposes on how you can read persisted train data and get them into spark data frames


import sys, getopt, random
from pyspark.sql import SparkSession

# CONSTANTS
SOURCE_DATA_PATH_TEMPLATE = 'hdfs:/user/{}/final_project/all_data/{}'
TRAIN_DATA_PATH_TEMPLATE = 'hdfs:/user/{}/final_project/train_data/{}/{}'
SEED = 20

def get_user():
	''' Returns user name (net id in this case) '''
	import getpass
	return getpass.getuser()

def newSparkSession():
	#return SparkSession.builder.getOrCreate()
	mem = "19GB"
	spark = (SparkSession.builder.appName("QuaranALS")
		.master("yarn")
		.config("sparn.executor.memory", mem)
		.config("sparn.driver.memory",mem)
		.getOrCreate())
	spark.sparkContext.setLogLevel("ERROR")
	return spark

def get_source_data( sparkSession = None ):
	''' Reads source data and writes to user hdfs '''
	spark = sparkSession or newSparkSession()

	filePath = 'hdfs:/user/bm106/pub/goodreads'
	fileNames = [ 'goodreads_interactions.csv', \
		      'user_id_map.csv', \
              	      'book_id_map.csv' ]

	for fileName in fileNames:
		table = spark.read.csv( "{}/{}".format(filePath, fileName), header = True, inferSchema = True )
		table.write.parquet(SOURCE_DATA_PATH_TEMPLATE.format(get_user(), fileName.split(".")[0]))

def save_train_data( name, train_interactions, val_interactions, test_interactions ):
	train_interactions.write.parquet(TRAIN_DATA_PATH_TEMPLATE.format(get_user(), name, 'train_interactions'))
	val_interactions.write.parquet(TRAIN_DATA_PATH_TEMPLATE.format(get_user(), name, 'val_interactions'))
	test_interactions.write.parquet(TRAIN_DATA_PATH_TEMPLATE.format(get_user(), name, 'test_interactions'))

def read_train_data( name, sparkSession = None):
	spark = sparkSession or newSparkSession()
	train_interactions = spark.read.parquet(TRAIN_DATA_PATH_TEMPLATE.format(get_user(), name, 'train_interactions'))
	val_interactions = spark.read.parquet(TRAIN_DATA_PATH_TEMPLATE.format(get_user(), name, 'val_interactions'))
	test_interactions = spark.read.parquet(TRAIN_DATA_PATH_TEMPLATE.format(get_user(), name, 'test_interactions'))
	return (train_interactions, val_interactions, test_interactions)
	
def split_users_interactions(interactions, users, splitProportion):
	""" Returns a split of the interactions of the users received. 
	    Used to include interactions from validation and test users in the training set. """
	import pyspark.sql.functions as F
	from pyspark.sql.window import Window
	window = Window.partitionBy('user_id').orderBy('book_id')
	ranked_interactions = interactions.join(users,'user_id','leftsemi').select("user_id","book_id","rating", F.percent_rank().over(window).alias("percent_rank"))
	return (ranked_interactions.filter(ranked_interactions.percent_rank <= splitProportion).drop('percent_rank'), ranked_interactions.filter(ranked_interactions.percent_rank > splitProportion).drop('percent_rank'))

def split_train_val_test(users,interactions):		
	training_users, validation_users, test_users = users.randomSplit([.6,.2,.2],seed=SEED)
	training_users_training_interactions = interactions.join(training_users,'user_id','leftsemi')
	validation_users_training_interactions, validation_users_validation_interactions = split_users_interactions(interactions, validation_users, 0.5)
	test_users_training_interactions, test_users_test_interactions = split_users_interactions(interactions, test_users, 0.5)

	interactions_training = training_users_training_interactions.union(validation_users_training_interactions).union(test_users_training_interactions)
	interactions_validation = validation_users_validation_interactions
	interactions_test = test_users_test_interactions

	return (interactions_training, interactions_validation, interactions_test)

def generate_train_data( sampleSize = .01, min_interactions = 10, sparkSession = None):

	random.seed(SEED)
	spark = sparkSession or newSparkSession()

	interactions = spark.read.parquet(SOURCE_DATA_PATH_TEMPLATE.format(get_user(), 'goodreads_interactions'))
	interactions = interactions.filter('rating > 0').drop('is_read').drop('is_reviewed')

	if min_interactions:
		interactions = interactions.join(interactions.groupBy(interactions.user_id).agg({'book_id':'count'}).filter('count(book_id) >= 10'),'user_id','leftsemi')
		#interactions = spark.sql('SELECT * FROM interactions WHERE user_id IN (SELECT user_id from interactions GROUP BY user_id HAVING COUNT(book_id)>={})'.format(min_interactions))

	users = interactions.select('user_id').distinct()
	if sampleSize > 0:
		users = users.sample(False, sampleSize, seed = SEED)
		interactions = interactions.join(users,'user_id','leftsemi')	
	interactions_split = split_train_val_test(users,interactions)
	save_train_data("{}_percent".format(int(sampleSize*100)), *interactions_split)	
	 
def main(argv):

	try:
		opts, args = getopt.getopt(argv, 'lt:r:' )
	except getopt.GetoptError as err:
		print(err)
		#usage()
		#sys.exit(2)

	for option,arg in opts:
		if option == '-l':
			print("Getting Source Data")
			get_source_data()
		if option == '-t':
			print("Generating Training Data")
			generate_train_data( sampleSize = float(arg))
		if option == '-r':
			train, val, test = read_train_data(arg)
			

if __name__ == "__main__":
	main(sys.argv[1:])


