#imported the required packages
from pyspark import SparkContext, SparkConf
import numpy
from scipy import spatial

conf = SparkConf()
conf.setAppName('MovieRecommender')
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")
conf.set("spark.eventLog.enabled", "true")
conf.set("spark.serializer ", "org.apache.spark.serializer.KryoSerializer")
sc = SparkContext(conf=conf)

#created rdd on movies and ratings dataset
movies_data=sc.textFile("/home/hadoop/movies.csv").map(lambda x:x.split(",")).map(lambda x:(x[0],x[1]))
ratings_data=sc.textFile("/home/hadoop/ratings.csv").map(lambda x:x.split(",")).map(lambda x:(x[1],(x[0],x[2])))

#got the userid,(movie title,rating) from two datasets
combineddata=movies_data.join(ratings_data)
requireddata=combineddata.map(lambda x:(x[1][1][0],(x[1][0],x[1][1][1]))).cache()

#used self join on the rdd
joinedRatings = requireddata.join(requireddata)

#to remove duplicates 
def removeDuplicates( ratingvalues ):
    ratings = ratingvalues[1]
    (movie_title1, movie_rating1) = ratings[0]
    (movie_title2, movie_rating2) = ratings[1]
    return movie_title1 < movie_title2
uniqueratings = joinedRatings.filter(removeDuplicates) 

def formPairs( ratingvalues ):
    ratings = ratingvalues[1]
    (movie_title1, movie_rating1) = ratings[0]
    (movie_title2, movie_rating2) = ratings[1]
    return (movie_title1, movie_title2), (movie_rating1, movie_rating2)
movieratingPairs = uniqueratings.map(formPairs) # rdd o/p as (movie_title1,movie_title2),(movie_rating1, movie_rating2)
groupMoviepairRatings=movieratingPairs.groupByKey() # grouped all values of a respective key

#used to find correlation
def correlation(ratings):
    q1=[]
    q2=[]
    for r1 in ratings:
        q1.append(float(r1[0]))
        q2.append(float(r1[1]))
        cor = numpy.corrcoef(q1,q2)[0,1]
        cos_cor = 1-spatial.distance.cosine(q1,q2)
        avg_cor = 0.5*(cor+cos_cor)
        n=len(ratings)
    return cor,cos_cor,avg_cor,n    
moviePairSimilarities = groupMoviepairRatings.mapValues(correlation)

moviePairSimilarities.saveAsTextFile("/home/hadoop/output.txt")#saving the output file in the location


