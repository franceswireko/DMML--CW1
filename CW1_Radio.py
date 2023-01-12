import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import csv
import mlxtend.frequent_patterns
import mlxtend.preprocessing
import pandas_profiling

#read data set file
data = pd.read_csv('CW1_Last_FM.csv')
print(data.head)
radio_data = data.copy()
radio_data.info()

#statistics of the data
#profile = radio_data.profile_report(title ="Last Fm Report")
#profile.to_file(output_file="./Last_Fm_Report.html")
#statistics of the data
#radio_data.profile_report()

data_duplicate= radio_data[radio_data.duplicated(keep=False)]
print("Duplicate Data")
print(data_duplicate)

radio_data.drop_duplicates(inplace=True)
data_duplicate= radio_data[radio_data.duplicated(keep=False)]
print(data_duplicate)

print("Distinct Users: ",len(radio_data.user.unique()))
print("Distinct Artist: ",len(radio_data.artist.unique()))
print("Distinct Country: ",len(radio_data.country.unique()))

artist_count = radio_data.artist.value_counts(normalize=True)
print(artist_count)

#Grouping the data into 'artist basket' to redute the dataset
radio_data.groupby(['country', 'user', 'sex']).apply(lambda x: ','.join(x['artist'])).reset_index()
radio_data['artist'] = radio_data.groupby(['country', 'user', 'sex'])['artist'].transform(lambda x: ','.join(x))
print(radio_data)

#drop duplicate data
radio_data=radio_data.drop_duplicates()
artist_radio_data = radio_data['artist'].to_frame().reset_index()
artist_radio_data.drop('index', axis=1, inplace=True)
print(artist_radio_data)

with open('Playlist.csv', newline='') as f:
    reader = csv.reader(f)
    playlist = list(reader)
print(len(playlist))
#playlist.pop(0)
#print(len(playlist))

#Apriori Implementation
encode_=mlxtend.preprocessing.TransactionEncoder()
encode_arr=encode_.fit_transform(playlist)
print(encode_arr)
encode_df = pd.DataFrame(encode_arr, columns=encode_.columns_)
print(encode_df.head())

#Filtering the dataset to obtaim frequent itemset with chosen support of 0.03
frequent_artist = apriori(encode_df, min_support=0.03, use_colnames=True)
print("frequent artist with support of atleast 3%")
print(frequent_artist)

#Among all items, let’s will select the ones that have a minimum confidence of 0.025
rule = association_rules(frequent_artist, metric="confidence",min_threshold=0.025)
print("Confidence metric with 2.5% minimum threshold")
print(rule)
#rule.to_csv('First rule.csv', index=False)

#With this step, let’s impose a minimum threshold on the lift of .0.01:
rule = association_rules(frequent_artist, metric="lift", min_threshold=0.01)
print("Lift metric with 1% minimum threshold")
print(rule)
#rule.to_csv('Second rule.csv', index=False)




