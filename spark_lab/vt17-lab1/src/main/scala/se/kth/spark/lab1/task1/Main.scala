  package se.kth.spark.lab1.task1

import se.kth.spark.lab1._

import org.apache.spark._
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"

    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    rdd.take(5)

    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(line => line.split("\t"))

    //Step3: map each row into a Song object by using the year label and the first three features  
    val linesRdd = recordsRdd.map(line => (line(0).split(",")))
    val songsRdd = linesRdd.map(line => (line(0).toFloat.toInt,line(1).toFloat, line(2).toFloat, line(3).toFloat))

    //Step4: convert your rdd into a dataframe
    val songsDf = songsRdd.toDF("year","feat1","feat2","feat3")
    songsDf.createOrReplaceTempView("Songs")
    
    //1. How many songs there are in the DataFrame?
    songsDf.count()
    sqlContext.sql("SELECT count(*) FROM Songs").show()
    
    //2. How many songs were released between the years 1998 and 2000?
    songsDf.filter($"year">=1998 && $"year"<=2000).count()
    sqlContext.sql("SELECT count(*) FROM Songs WHERE year >= 1998 AND year <= 2000").show()
    
    //3. What is the min, max and mean value of the year column?
    songsDf.agg(avg("year"), max("year"), min("year")).show()
    sqlContext.sql("SELECT avg(year), max(year), min(year) FROM Songs").show()

    //4. Show the number of songs per year between the years 2000 and 2010?
    songsDf.filter($"year">=2000 && $"year"<=2010).groupBy("year").count().orderBy("year").show()
    sqlContext.sql("SELECT year, count(*) FROM Songs WHERE year >= 2000 AND year <= 2010 GROUP BY year ORDER BY year").show()    
  }
}