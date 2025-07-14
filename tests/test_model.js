// Tests for model functionality
import { FactChecker } from "../src/prediction.js"
import { describe, beforeEach, test, expect, jest } from "@jest/globals"

describe("FactChecker", () => {
  let factChecker

  beforeEach(() => {
    factChecker = new FactChecker()
    // Mock model for testing
    factChecker.model = {
      predict: jest.fn().mockReturnValue({
        data: jest.fn().mockResolvedValue([0.8]),
      }),
    }
  })

  test("should make predictions", async () => {
    const preprocessedText = [1, 2, 3, 4, 5]
    const result = await factChecker.predict(preprocessedText)

    expect(result.confidence).toBe(0.8)
    expect(result.prediction).toBe("True")
    expect(result.probability.true).toBe(0.8)
    expect(result.probability.false).toBe(0.2)
  })

  test("should classify as false for low confidence", async () => {
    factChecker.model.predict.mockReturnValue({
      data: jest.fn().mockResolvedValue([0.3]),
    })

    const preprocessedText = [1, 2, 3, 4, 5]
    const result = await factChecker.predict(preprocessedText)

    expect(result.prediction).toBe("False")
  })

  test("should throw error when model not loaded", async () => {
    factChecker.model = null
    const preprocessedText = [1, 2, 3, 4, 5]

    await expect(factChecker.predict(preprocessedText)).rejects.toThrow("Model not loaded")
  })
})
