// Prediction utilities for FactRadar
import * as tf from "@tensorflow/tfjs"

class FactChecker {
  constructor(modelPath) {
    this.model = null
    this.loadModel(modelPath)
  }

  async loadModel(modelPath) {
    try {
      this.model = await tf.loadLayersModel(modelPath)
      console.log("Model loaded successfully")
    } catch (error) {
      console.error("Error loading model:", error)
    }
  }

  async predict(preprocessedText) {
    if (!this.model) {
      throw new Error("Model not loaded")
    }

    const inputTensor = tf.tensor2d([preprocessedText])
    const prediction = this.model.predict(inputTensor)
    const result = await prediction.data()

    // Clean up tensors
    inputTensor.dispose()
    prediction.dispose()

    return {
      confidence: result[0],
      prediction: result[0] > 0.5 ? "True" : "False",
      probability: {
        true: result[0],
        false: 1 - result[0],
      },
    }
  }

  async checkFact(text, preprocessor) {
    const preprocessedText = preprocessor.preprocessText(text)
    return await this.predict(preprocessedText)
  }
}

export { FactChecker }
