Execute files from Github Respository in the Following order to replicate analysis:

For Baseline Model:

1) Run 'python training_data.py -l' to get source data to user hdfs folder (writes under final_project/all_data as parquet files)
2) Create these two directories in hdfs:
hfs -mkdir final_project/train_data
hfs -mldir final_project/models  
3) Run 'python training_data.py -t fraction' to preprocess and split data into training, validation and test sets. Replace fraction to include fraction of the data you want to subsample (100% will use the whole data set)
4) Run 'python als_model -t 5_percent' to generate a file with the grid search results that will be saved in your local drive (not hdfs), in git folder. 
So please create the folder "results" under final-project-quaranteam (replace this with the name of your folder)   (>>mkdir results)
After you run it you will be able to see where the results were saved. Replace 5_percent by whatever percentage of subsampled data you want to run hyperparameter 
search on. This py file will also save the ALS models for each hyperparameter combination under final_project/models/.
5) Run 'python export_factors_testset_performance.py' to export learned user and item factors to your local directory as a json file and check the performance of 
the best model on test data set.
6) Run 'hyperparameter_search_results.ipynb' to graph hyperparameter search results. Reads in results json file generated in step #4.

For Annoy Extension:

1) Install the Annoy package by running:
pip install --user annoy
2) Run 'Annoy_fast_search_extension.ipynb' to implement fast search extension. This reads in learned factors exported in step #5. 

For LightFM extension:
1) Install the LightFM package by running:
pip install lightfm
2) Repeat steps #3 from the baseline model procedure for the following subsamples: 0.1%, 0.5%, 1% and 5% to obtain the datasets and export these to a local machine.
3) Repeat step #4 from the baseline model on the subsets of data obtained previously, using the slightly modified 'comparison_als_model.py' which times the time taken for training and evaluation.
5) Run the 'light_fm_val.py' script to obtain results for LightFM implementation.



