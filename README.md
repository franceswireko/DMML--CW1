# DMML--CW1

I. Data Description – COVID-19 Global Spread Data
The COVID pandemic situation impacted us all during the past three years.
Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus.
Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. 
However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular 
disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illnesses. Anyone can get sick with COVID-19 and become 
seriously ill or die at any age.
The best way to prevent and slow down transmission is to be well informed about the disease and how the virus spreads. Students are offered to research 
the COVID-19 global data by means of clustering analysis.

II. Dataset Information – Radio Listeners Data: To understand what exactly a listener prefers listening to on the radio, every detail is recorded online. 
This recorded information is used for recommending music that the listener is likely to enjoy and to come up with a “focused” marketing strategy that 
sends out advertisements for music that a listener may wish to buy. However, this results in wasting money on scarce advertising.
Suppose that you are provided with data from a music community site, giving you details of each user. This will be further enhanced by getting access to 
a log of every artist that listed users have downloaded on their computer. With this data, you will also get information on the demographics of the listed 
users (such as age, sex, location, occupation, and interests).
The objective of providing this data lies in building a system that recommends new music to the users in this listed community. From the available 
information, it is usually not difficult to determine the support for various individual artists (that is, the frequencies of a specific music 
genre/artist or song that a user is listening to) as well as the joint support for pairs (or larger groupings) of artists.
You need to count the number of incidences across all your network members and divide it by the number of members.
In the mentioned data set, a large chunk of information close to 300,000 records of song (or artists) selections is listed that is per the listening 
frequency given by 15,000 users. Each row of the data set contains the name of the artist that the user has been listening to. The first user is a 
German lady, who has listened to 16 artists. This has resulted in the first 16 rows of the data matrix.
First, you need to transform the data given here into an incidence matrix, where each listener is represented by a row, with 0s and 1s across the columns. 
This indicates if a listener has chosen a certain artist or not.
Then, calculate the support for each of the listed 1004 artists and display the support for all artists with a support threshold greater than 0.08.
