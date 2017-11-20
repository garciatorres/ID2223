package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._

import org.apache.spark.hack._
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Matrices
import org.apache.spark.mllib.evaluation.RegressionMetrics

case class Instance(label: Double, features: Vector)

object Helper {
  
  /* Computes the Root Mean Square Error of a given RDD of tuples of (label, prediction) */
  
  def rmse(labelsAndPreds: RDD[(Double, Double)]): Double = {
    Math.sqrt(labelsAndPreds.map{case (l,p) => Math.pow((l - p), 2)}.sum/labelsAndPreds.count)
  }

  /* Applies the formula: prediction = weightsT dot features */
  
  def predictOne(weights: Vector, features: Vector): Double = {
    VectorHelper.dot(weights,features)
  }

  /* This method receives as input a RDD of observation instances
   * and in the end returns an RDD of (label, prediction)
   * */
  
  def predict(weights: Vector, data: RDD[Instance]): RDD[(Double, Double)] = {
    data.map(i => (i.label, predictOne(weights, i.features)))
  }
}

class MyLinearRegressionImpl(override val uid: String)
    extends MyLinearRegression[Vector, MyLinearRegressionImpl, MyLinearModelImpl] {

  def this() = this(Identifiable.randomUID("mylReg"))

  override def copy(extra: ParamMap): MyLinearRegressionImpl = defaultCopy(extra)

  /* Computes the formula: GSij = (wiT.xj-yj).xj */  
  
  def gradientSummand(weights: Vector, lp: Instance): Vector = {
    VectorHelper.dot(lp.features,VectorHelper.dot(weights,lp.features) - lp.label)
  }

  /* Computes the formula: gradient = SUM GSij */  
  
  def gradient(d: RDD[Instance], weights: Vector): Vector = {
    d.map(i => gradientSummand(weights,i)).reduce((a,b) => VectorHelper.sum(a,b))
  }

  def linregGradientDescent(trainData: RDD[Instance], numIters: Int): (Vector, Array[Double]) = {

    val n = trainData.count()
    val d = trainData.take(1)(0).features.size
    var weights = VectorHelper.fill(d, 0)
    val alpha = 1.0
    val errorTrain = Array.fill[Double](numIters)(0.0)

    for (i <- 0 until numIters) {
      //compute this iterations set of predictions based on our current weights
      val labelsAndPredsTrain = Helper.predict(weights, trainData)
      //compute this iteration's RMSE
      errorTrain(i) = Helper.rmse(labelsAndPredsTrain)

      //compute gradient
      val g = gradient(trainData, weights)
      //update the gradient step - the alpha
      val alpha_i = alpha / (n * scala.math.sqrt(i + 1))
      val wAux = VectorHelper.dot(g, (-1) * alpha_i)
      //update weights based on gradient
      weights = VectorHelper.sum(weights, wAux)
    }
    (weights, errorTrain)
  }

  def train(dataset: Dataset[_]): MyLinearModelImpl = {
    //println("Training")

    val numIters = 100

    val instances: RDD[Instance] = dataset.select(
      col($(labelCol)), col($(featuresCol))).rdd.map {
        case Row(label: Double, features: Vector) =>
          Instance(label, features)
      }

    val (weights, trainingError) = linregGradientDescent(instances, numIters)
    new MyLinearModelImpl(uid, weights, trainingError)
  }
}

class MyLinearModelImpl(override val uid: String, val weights: Vector, val trainingError: Array[Double])
    extends MyLinearModel[Vector, MyLinearModelImpl] {

  override def copy(extra: ParamMap): MyLinearModelImpl = defaultCopy(extra)

  def predict(features: Vector): Double = {
  //println("Predicting")
    val prediction = Helper.predictOne(weights, features)
    prediction
  }
}