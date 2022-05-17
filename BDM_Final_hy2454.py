#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import json
import numpy as np
import pandas as pd
import sys

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
sc = pyspark.SparkContext.getOrCreate()
spark = SparkSession(sc)
spark
from pyspark.sql.types import DateType, IntegerType, MapType, StringType, FloatType
from pyproj import Transformer
import shapely
from shapely.geometry import Point


# read nyc_cbg_centroids.csv file and make a nyc cbg dictionary, use to find the lon and lat
t = Transformer.from_crs(4326, 2263)
nyc_cbg_centroids = pd.read_csv('nyc_cbg_centroids.csv')
nyc_cbg_centroids['lat'] = t.transform(nyc_cbg_centroids['latitude'], nyc_cbg_centroids['longitude'])[0]
nyc_cbg_centroids['lon'] = t.transform(nyc_cbg_centroids['latitude'], nyc_cbg_centroids['longitude'])[1]
nyc_cbg_centroids['cbg_fips'] = nyc_cbg_centroids['cbg_fips'].astype(str)
nyc_cbg_centroids.set_index(["cbg_fips"], inplace=True)
nyc_cbg_centroids = nyc_cbg_centroids.transpose()

# read nyc_supermarkets
nyc_supermarkets = pd.read_csv('nyc_supermarkets.csv')
spark = SparkSession.builder.appName('pandasToSparkDF').getOrCreate()
nyc_supermarkets = spark.createDataFrame(nyc_supermarkets)


weekly_patterns_nyc = spark.read.csv('/tmp/bdm/weekly-patterns-nyc-2019-2020/*', header = True, escape = '"')

## filter out places that not belong to nyc_supermarkets for weekly_patterns_nyc
weekly_patterns_nyc2 = weekly_patterns_nyc.join(nyc_supermarkets, weekly_patterns_nyc['placekey'] == nyc_supermarkets['safegraph_placekey'],'left')                                .dropna(subset = ['safegraph_placekey'])                                .select('placekey','date_range_start','date_range_end','poi_cbg','visitor_home_cbgs')

# calculate number of visitors and total distance
def cbg(poi_cbg, visitor_home_cbgs):
  num_visitors = 0
  total_distance = 0
  for k,v in json.loads(visitor_home_cbgs).items():    
    if nyc_cbg_centroids.get(k,None) is not None:
      num_visitors += v
      total_distance += Point(nyc_cbg_centroids.get(k, None)[2], nyc_cbg_centroids.get(k, None)[3]).distance(Point(nyc_cbg_centroids.get(poi_cbg, None)[2], nyc_cbg_centroids.get(poi_cbg, None)[3]))/5280
  return [total_distance, float(num_visitors)]

udfExpand = F.udf(cbg,T.ArrayType(T.FloatType()))
weekly_patterns_nyc3 = weekly_patterns_nyc2.select('placekey', 'date_range_start', 'date_range_end', 'poi_cbg',
                 udfExpand(weekly_patterns_nyc2['poi_cbg'], weekly_patterns_nyc2['visitor_home_cbgs']).alias('visitor_home_cbgs'))


# get visitor_home_cbgs data for 2019-03,2019-10,2020-03,2020-10
df_19_03 = weekly_patterns_nyc3.filter(((weekly_patterns_nyc3['date_range_start'] >= '2019-03-01') & (weekly_patterns_nyc3['date_range_start'] <= '2019-03-31')) | ((weekly_patterns_nyc3['date_range_end'] >= '2019-03-01') & (weekly_patterns_nyc3['date_range_end'] <= '2019-03-31'))).select(weekly_patterns_nyc3['poi_cbg'], weekly_patterns_nyc3['visitor_home_cbgs'].alias('2019-03'))
df_19_10 = weekly_patterns_nyc3.filter(((weekly_patterns_nyc3['date_range_start'] >= '2019-10-01') & (weekly_patterns_nyc3['date_range_start'] <= '2019-10-31')) | ((weekly_patterns_nyc3['date_range_end'] >= '2019-10-01') & (weekly_patterns_nyc3['date_range_end'] <= '2019-10-31'))).select(weekly_patterns_nyc3['poi_cbg'], weekly_patterns_nyc3['visitor_home_cbgs'].alias('2019-10'))
df_20_03 = weekly_patterns_nyc3.filter(((weekly_patterns_nyc3['date_range_start'] >= '2020-03-01') & (weekly_patterns_nyc3['date_range_start'] <= '2020-03-31')) | ((weekly_patterns_nyc3['date_range_end'] >= '2020-03-01') & (weekly_patterns_nyc3['date_range_end'] <= '2020-03-31'))).select(weekly_patterns_nyc3['poi_cbg'], weekly_patterns_nyc3['visitor_home_cbgs'].alias('2020-03'))
df_20_10 = weekly_patterns_nyc3.filter(((weekly_patterns_nyc3['date_range_start'] >= '2020-10-01') & (weekly_patterns_nyc3['date_range_start'] <= '2020-10-31')) | ((weekly_patterns_nyc3['date_range_end'] >= '2020-10-01') & (weekly_patterns_nyc3['date_range_end'] <= '2020-10-31'))).select(weekly_patterns_nyc3['poi_cbg'], weekly_patterns_nyc3['visitor_home_cbgs'].alias('2020-10'))

# cobime different time period data into one dataframe, which date for one column
output = df_19_03.join(df_19_10,'poi_cbg', 'outer').join(df_20_03,'poi_cbg', 'outer').join(df_20_10,'poi_cbg', 'outer')

# group by poi_cbg
output2 = output.groupBy('poi_cbg').agg(F.collect_list('2019-03').alias('2019-03'),F.collect_list('2019-10').alias('2019-10'), F.collect_list('2020-03').alias('2020-03'),F.collect_list('2020-10').alias('2020-10'))

# calculate the average 
def calAvg(val):
  dis = 0
  vis = 0
  for x in val:
    if x[0]:
      dis += float(x[0])
    if x[1]:
      vis += x[1]
  if dis == 0 or vis == 0:
    return 0
  return dis/vis


udfExpand = F.udf(calAvg, FloatType())
output3 = output2.select('poi_cbg', 
                 udfExpand(output2['2019-03']).alias('2019-03'), udfExpand(output2['2019-10']).alias('2019-10'), udfExpand(output2['2020-03']).alias('2020-03'), udfExpand(output2['2020-10']).alias('2020-10'))
output3.write.option("header",True).csv(sys.argv[1])

