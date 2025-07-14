// Text preprocessing utilities for FactRadar
class TextPreprocessor {
  constructor(vocabPath, paramsPath) {
    this.vocab = null
    this.params = null
    this.loadResources(vocabPath, paramsPath)
  }

  async loadResources(vocabPath, paramsPath) {
    try {
      const vocabResponse = await fetch(vocabPath)
      this.vocab = await vocabResponse.json()

      const paramsResponse = await fetch(paramsPath)
      this.params = await paramsResponse.json()
    } catch (error) {
      console.error("Error loading preprocessing resources:", error)
    }
  }

  tokenize(text) {
    // Basic tokenization
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, " ")
      .split(/\s+/)
      .filter((token) => token.length > 0)
  }

  textToSequence(text) {
    const tokens = this.tokenize(text)
    const sequence = tokens.map((token) => this.vocab.word_index[token] || this.vocab.word_index["<UNK>"])

    // Pad or truncate sequence
    if (sequence.length < this.params.max_sequence_length) {
      const padding = new Array(this.params.max_sequence_length - sequence.length).fill(0)
      return sequence.concat(padding)
    } else {
      return sequence.slice(0, this.params.max_sequence_length)
    }
  }

  preprocessText(text) {
    return this.textToSequence(text)
  }
}

export { TextPreprocessor }
