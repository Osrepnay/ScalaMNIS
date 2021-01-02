package io.github.osrepnay.scalamnist

import smile.base.mlp.{Layer, OutputFunction}
import smile.math.TimeFunction

import scala.io.Source

object ClassifyDigits {

	def main(args: Array[String]): Unit = {
		val bufferedSourceTr = Source.fromURL("https://pjreddie.com/media/files/mnist_train.csv")
		val trainData = bufferedSourceTr.getLines().map {line =>
			val splitLine = line.split(",")
			(splitLine.head.toInt, splitLine.tail.map(_.toDouble / 255))
		}
		val trainDataIterators = trainData.duplicate
		println("Done loading training data")
		val trainLabels = trainDataIterators._1.map(_._1).toArray
		val trainImgs = trainDataIterators._2.map(_._2.toArray).toArray
		val imgNN = smile.classification.mlp(trainImgs, trainLabels, Array(
			Layer.rectifier(30),
			Layer.rectifier(30),
			Layer.mle(10, OutputFunction.SOFTMAX)
		), epochs = 30,
			learningRate = TimeFunction.constant(0.3),
			momentum = TimeFunction.constant(0.1),
			weightDecay = 0)
		println("Finished training")
		val bufferedSourceTe = Source.fromURL("https://pjreddie.com/media/files/mnist_test.csv")
		val testData = bufferedSourceTe.getLines().map(_.split(","))
		val testDataIterators = testData.duplicate
		println("Done loading testing data")
		val testLabels = testDataIterators._1.map(_.head.toInt).toArray
		val testImgs = testDataIterators._2.map(_.tail.map(_.toDouble / 255)).toArray
		println("Testing random test data")
		val predictions =
			(0 until 1000).map {_ =>
				val randTestIdx = (math.random() * testLabels.length).toInt
				val predicted = imgNN.predict(testImgs(randTestIdx))
				(randTestIdx, predicted)
			}
		predictions.foreach(prediction => println(s"Expected ${testLabels(prediction._1)}, got ${prediction._2}"))
		println(s"Percent correct: ${
			predictions.count(prediction => testLabels(prediction._1) == prediction._2).
				toDouble / predictions.length * 100
		}%")
	}

}
