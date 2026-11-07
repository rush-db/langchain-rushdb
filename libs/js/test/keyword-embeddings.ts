import { Embeddings } from '@langchain/core/embeddings'

/** Deterministic 4-d embeddings: one axis per topic word (same as the Python tests). */
export class KeywordEmbeddings extends Embeddings {
  static AXES = ['apple', 'ocean', 'graph', 'agent']

  constructor() {
    super({})
  }

  embed(text: string): number[] {
    const lowered = text.toLowerCase()
    const vector = KeywordEmbeddings.AXES.map((word) => (lowered.includes(word) ? 1 : 0))
    return vector.some(Boolean) ? vector : [0.1, 0.1, 0.1, 0.1]
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    return texts.map((t) => this.embed(t))
  }

  async embedQuery(text: string): Promise<number[]> {
    return this.embed(text)
  }
}
