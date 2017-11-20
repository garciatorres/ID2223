/* Check: https://spark.apache.org/docs/1.6.1/ml-features.html */

package se.kth.spark.lab1.task2

import se.kth.spark.lab1._

import org.apache.spark._
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
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
    val rawDF = sc.textFile(filePath).toDF()
    rawDF.cache()

    //Step1: tokenize each row
    val rTokenizer = new RegexTokenizer().setInputCol("value").setOutputCol("tokens").setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    val step2 = rTokenizer.transform(rawDF)
    step2.show(5)
    
    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector().setInputCol("tokens").setOutputCol("vector")
    val step3 = arr2Vect.transform(step2)

    //Step4: extract the label(year) into a new column
    val lSlicer = new VectorSlicer().setInputCol("vector").setOutputCol("year").setIndices(Array(0))
    val step4 = lSlicer.transform(step3)

    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val v2d = new Vector2DoubleUDF(v => v(0)).setInputCol("year").setOutputCol("double")
    val step5 = v2d.transform(step4)
        
    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)
    //how can you determine the minimum value of the label dynamically (and in a more elegant way)?
    val minimum = v2d.transform(lSlicer.transform(arr2Vect.transform(rTokenizer.transform(rawDF)))).agg(min("double")).head.get(0).toString().toDouble   
    val lShifter = new DoubleUDF(v => v - minimum).setInputCol("double").setOutputCol("label")
    val step6 = lShifter.transform(step5)

    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer().setInputCol("vector").setOutputCol("features").setIndices(Array(1,2,3))
    val step7 = fSlicer.transform(step6)

    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(Array(rTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer))

    //Step9: generate model by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)

    //Step10: transform data with the model - do predictions
    val step10 = pipelineModel.transform(rawDF) 

    //Step11: drop all columns from the dataframe other than label and features
    val step11 = step10.select("label", "features")
    step11.show(10)
  }
}