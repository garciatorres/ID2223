/* Check: http://spark.apache.org/docs/latest/ml-tuning.html 	*/

package se.kth.spark.lab1.task4

import se.kth.spark.lab1._

import org.apache.spark._
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator
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
    val Array(obsDF,testDF) = rawDF.randomSplit(Array[Double](0.8, 0.2))

    /* Pipeline: the same as in task 3 */
      
    val rTokenizer = new RegexTokenizer().setInputCol("value").setOutputCol("tokens").setPattern(",")
    val arr2Vect = new Array2Vector().setInputCol("tokens").setOutputCol("vector")
    val lSlicer = new VectorSlicer().setInputCol("vector").setOutputCol("year").setIndices(Array(0))
    val v2d = new Vector2DoubleUDF(v => v(0)).setInputCol("year").setOutputCol("double")
    val minimum = v2d.transform(lSlicer.transform(arr2Vect.transform(rTokenizer.transform(obsDF)))).agg(min("double")).head.get(0).toString().toDouble   
    val features = rTokenizer.transform(obsDF).select("tokens").head().toString().count(_ == ',')
    val lShifter = new DoubleUDF(v => v - minimum).setInputCol("double").setOutputCol("label")
    val fSlicer = new VectorSlicer().setInputCol("vector").setOutputCol("features").setIndices(1 to features toArray)
    val myLR = new LinearRegression().setElasticNetParam(0.1)
    val pipeline = new Pipeline().setStages(Array(rTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR))
    val pipelineModel: PipelineModel = pipeline.fit(obsDF)
    
    /* Create a ParameterGridBuilder and add grids for the maxIter and regParam of the estimator. 
     * For each of the parameter chose three values above the base value and three bellow. 
     * */
    
    val paramGrid = new ParamGridBuilder()
    .addGrid(myLR.maxIter, Array(10, 50, 100, 150, 200, 250))
    .addGrid(myLR.regParam, Array(0.1, 0.25, 0.4, 0.6, 0.75, 0.9))
    .build()
    
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(6)

    val cvModel: CrossValidatorModel = cv.fit(obsDF)
    val lrModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(6).asInstanceOf[LinearRegressionModel]

    //print RMSE of our model
    println(s"RMSE: ${lrModel.summary.rootMeanSquaredError}")
    println(s"MaxIter: ${lrModel.getMaxIter}")
    println(s"RegParam: ${lrModel.getRegParam}")
    println(s"numFeatures: ${lrModel.numFeatures}")
    //do prediction - print first k
    pipelineModel.transform(testDF).select("label","prediction").show(10)
  }
}