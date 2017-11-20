package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}

object VectorHelper {

  /* the dot product of two vectors */
  def dot(v1: Vector, v2: Vector): Double = {
    require(v1.size == v2.size) 
    (0 to v1.size-1).map((i) => v1(i) * v2(i)).sum
  }
  
  /* the dot product of a vector and a scalar */
  def dot(v: Vector, s: Double): Vector = {
    Vectors.dense(v.toArray.map(v => v*s))
  }

  /* addition of two vectors */
  def sum(v1: Vector, v2: Vector): Vector = {
    require(v1.size == v2.size)
    Vectors.dense((0 to v1.size-1).map((i) => v1(i) + v2(i)).toArray)
  }

  /* create a vector of predefined size and initialize it with the predefined value */
  def fill(size: Int, fillVal: Double): Vector = {
    require(size > 0)
    Vectors.dense((0 to size-1).map((_) => fillVal).toArray)
  }
}