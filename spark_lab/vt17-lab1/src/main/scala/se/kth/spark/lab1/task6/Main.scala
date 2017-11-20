package se.kth.spark.lab1.task6

import se.kth.spark.lab1._

import org.apache.spark._
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer, PolynomialExpansion}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.{Row, SQLContext, DataFrame}
import org.apache.spark.sql.functions._

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF: DataFrame = sc.textFile(filePath).toDF()
    val Array(obsDF,testDF) = rawDF.randomSplit(Array[Double](0.8, 0.2))
    
    val rTokenizer = new RegexTokenizer().setInputCol("value").setOutputCol("tokens").setPattern(",")
    val arr2Vect = new Array2Vector().setInputCol("tokens").setOutputCol("vector")
    val lSlicer = new VectorSlicer().setInputCol("vector").setOutputCol("year").setIndices(Array(0))
    val v2d = new Vector2DoubleUDF(v => v(0)).setInputCol("year").setOutputCol("double")
    val minimum = v2d.transform(lSlicer.transform(arr2Vect.transform(rTokenizer.transform(obsDF)))).agg(min("double")).head.get(0).toString().toDouble
    val features = rTokenizer.transform(obsDF).select("tokens").head().toString().count(_ == ',')
    val lShifter = new DoubleUDF(v => v - minimum).setInputCol("double").setOutputCol("label")
    val fSlicer = new VectorSlicer().setInputCol("vector").setOutputCol("npFeatures").setIndices(1 to features toArray)
//  val fSlicer = new VectorSlicer().setInputCol("vector").setOutputCol("features").setIndices(1 to features toArray)    
    val pExpansion = new PolynomialExpansion().setInputCol("npFeatures").setOutputCol("features").setDegree(2)
    
    val begtime = System.nanoTime()
    
    val myLR = new MyLinearRegressionImpl()
    val pipeline = new Pipeline().setStages(Array(rTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, pExpansion, myLR))    
//  val pipeline = new Pipeline().setStages(Array(rTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR))
    val pipelineModel: PipelineModel = pipeline.fit(obsDF)
    val myLRModel = pipelineModel.stages(7).asInstanceOf[MyLinearModelImpl]
//  val myLRModel = pipelineModel.stages(6).asInstanceOf[MyLinearModelImpl]    
    val predictions = pipelineModel.transform(obsDF).select("label","prediction") 
    val RMSE = Helper.rmse(predictions.rdd.map(row => (row.getAs[Double](0), row.getAs[Double](1))))
    
    val endtime = System.nanoTime()
  
    //print RMSE of our model
    println(s"RMSE: ${RMSE}")
    println(s"numFeatures: ${features}")
    println(s"trainTime: ${(endtime-begtime)/1000000000.0} sec")
    //do prediction - print first k
    pipelineModel.transform(testDF).select("label","prediction").show(10)
  }
}