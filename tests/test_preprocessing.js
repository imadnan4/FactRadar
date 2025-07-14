// Tests for preprocessing functionality
import { TextPreprocessor, describe, beforeEach, test, expect } from "../src/preprocessing.js"

describe("TextPreprocessor", () => {
  let preprocessor

  beforeEach(() => {
    // Mock vocab and params for testing
    const mockVocab = {
      word_index: {
        "<PAD>": 0,
        "<START>": 1,
        "<UNK>": 2,
        the: 3,
        is: 4,
        test: 5,
      },
    }

    const mockParams = {
      max_sequence_length: 10,
    }

    preprocessor = new TextPreprocessor()
    preprocessor.vocab = mockVocab
    preprocessor.params = mockParams
  })

  test("should tokenize text correctly", () => {
    const text = "This is a test!"
    const tokens = preprocessor.tokenize(text)
    expect(tokens).toEqual(["this", "is", "a", "test"])
  })

  test("should convert text to sequence", () => {
    const text = "the test is"
    const sequence = preprocessor.textToSequence(text)
    expect(sequence).toHaveLength(10) // padded to max_sequence_length
    expect(sequence.slice(0, 3)).toEqual([3, 5, 4]) // the, test, is
  })

  test("should handle unknown words", () => {
    const text = "unknown word"
    const sequence = preprocessor.textToSequence(text)
    expect(sequence[0]).toBe(2) // <UNK> token
    expect(sequence[1]).toBe(2) // <UNK> token
  })
})
