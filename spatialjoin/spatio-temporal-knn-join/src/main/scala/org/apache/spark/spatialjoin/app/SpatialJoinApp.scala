package org.apache.spark.spatialjoin.app

import java.sql.Timestamp

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.spatialjoin.extractor.STExtractor
import org.apache.spark.spatialjoin.join.SpatioTemporalKnnJoin
import org.apache.spark.spatialjoin.serialize.GeometryKryoRegistrator
import org.apache.spark.spatialjoin.utils.WKTUtils
import org.locationtech.jts.geom.Geometry

/**
 * @author wangrubin3
 **/
object SpatialJoinApp {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .set("spark.driver.maxResultSize", "10g")
      .set("spark.serializer", classOf[KryoSerializer].getName)
      .set("spark.executor.memory", "50g")
      .set("spark.driver.memory", "6g")
//      .set("spark.executor.instances", "42")
      .set("spark.kryo.registrator", classOf[GeometryKryoRegistrator].getName)
      .setAppName("SpatialJoinApp")
//      .setMaster("spark://startserver02:7077")
//      .setMaster("local[*]")
    val spark = new SparkContext(conf)

    val alpha = args(0).toInt //experiment value is 200
    val beta = args(1).toInt //experiment value is 40
    val binNum = args(2).toInt //experiment value is 200
    val deltaMilli: Long = args(3).toInt * 60 * 1000 // time unit for args(3) is minute, experiment value is 30 minutes
    val k = args(4).toInt //experiment value is 15

    val leftPath = args(5) //data path of dataset R
    val rightPath = args(6) //data path of dataset S
    val storePath = args(7)

    val stKnnJoin = new SpatioTemporalKnnJoin(deltaMilli, k, alpha, beta, binNum)

    //function for parsing spatio-temporal record from file to rdd
    val readRdd = (filePath: String) => spark.textFile(filePath)
      .map(line => {
        val attrs = line.split(",")
        val geom = WKTUtils.read(attrs(4))
        val time = Timestamp.valueOf(attrs(3))
        (geom, time)
      })

    //parsing line
    val readRdd_line = (filePath: String) => spark.textFile(filePath)
      .map(line => {
        val attrs = line.split("\t")
        val geom = WKTUtils.read(attrs(0))
        val time_s = Timestamp.valueOf(attrs(1))
        val time_e = Timestamp.valueOf(attrs(2))
        (geom, time_s, time_e)
      })

    //if the left and right records are different
    //please define extractor for both respectively
    val extractor = new STExtractor[(Geometry, Timestamp)] {
      override def startTime(row: (Geometry, Timestamp)): Timestamp = row._2

      //end time equals start time for point data
      override def endTime(row: (Geometry, Timestamp)): Timestamp = row._2

      override def geom(row: (Geometry, Timestamp)): Geometry = row._1
    }

    val extractor_line = new STExtractor[(Geometry, Timestamp, Timestamp)] {
      override def startTime(row: (Geometry, Timestamp, Timestamp)): Timestamp = row._2

      //end time equals start time for point data
      override def endTime(row: (Geometry, Timestamp, Timestamp)): Timestamp = row._3

      override def geom(row: (Geometry, Timestamp, Timestamp)): Geometry = row._1
    }

    val joinTime = stKnnJoin.join(
      readRdd(leftPath), readRdd(rightPath),
      extractor, extractor)

    //println(statistic)
    spark.parallelize(Seq(joinTime))
      .repartition(1)
      .saveAsTextFile(storePath)

    spark.stop()
  }
}
