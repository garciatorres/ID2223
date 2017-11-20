/* Check: https://spark.apache.org/docs/1.6.1/ml-classification-regression.html#linear-regression */

package se.kth.spark.lab1.task3

import se.kth.spark.lab1._

import org.apache.spark._
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.sql.{SQLContext, DataFrame}
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
    
    /* split the raw dataset 80/20 */
    val Array(obsDF,testDF) = rawDF.randomSplit(Array[Double](0.8, 0.2))

    /* previous stages of pipeline: the same as in task 2 */
    
    val rTokenizer = new RegexTokenizer().setInputCol("value").setOutputCol("tokens").setPattern(",")
    val arr2Vect = new Array2Vector().setInputCol("tokens").setOutputCol("vector")
    val lSlicer = new VectorSlicer().setInputCol("vector").setOutputCol("year").setIndices(Array(0))
    val v2d = new Vector2DoubleUDF(v => v(0)).setInputCol("year").setOutputCol("double")
    val minimum = v2d.transform(lSlicer.transform(arr2Vect.transform(rTokenizer.transform(obsDF)))).agg(min("double")).head.get(0).toString().toDouble
    val features = rTokenizer.transform(obsDF).select("tokens").head().toString().count(_ == ',')
    val lShifter = new DoubleUDF(v => v - minimum).setInputCol("double").setOutputCol("label")
    val fSlicer = new VectorSlicer().setInputCol("vector").setOutputCol("features").setIndices(1 to features toArray)
    
    /* Linear Regression Basic parameters:
			• maximum iterations - set it to 10 or 50
			• regularization parameter - set it to 0.1 or 0.9
			• elastic net parameter - set it to 0.1
			*/

    val begtime = System.nanoTime()
    
    val myLR = new LinearRegression().setMaxIter(50).setRegParam(0.1).setElasticNetParam(0.1)
    val pipeline = new Pipeline().setStages(Array(rTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR)) 
    val pipelineModel: PipelineModel = pipeline.fit(obsDF)
    val lrModel = pipelineModel.stages(6).asInstanceOf[LinearRegressionModel]
    
    val endtime = System.nanoTime()
      
    //print RMSE of our model
    println(s"RMSE: ${lrModel.summary.rootMeanSquaredError}")
    println(s"MaxIter: ${lrModel.getMaxIter}")
    println(s"RegParam: ${lrModel.getRegParam}")
    println(s"numFeatures: ${lrModel.numFeatures}")
    println(s"trainTime: ${(endtime-begtime)/1000000000.0} sec")
    //do prediction - print first k
    pipelineModel.transform(testDF).select("label","prediction").show(10)
  }
}