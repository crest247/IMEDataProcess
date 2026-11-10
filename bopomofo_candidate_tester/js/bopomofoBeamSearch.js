export const TABLE_FILL_PENALTY = -12.0;
export const TABLE_FILL_UNIGRAM_LOG_PROB = -20.0;
const ROOT = -2147483648;
const STATE_CACHE_LIMIT = 4096;
const PATH_KEY_SHIFT = 1048576;

class Path {
  constructor(unitId, end, previous) {
    this.unitId = unitId;
    this.end = end;
    this.previous = previous;
  }

  get first() {
    return this.previous ? this.previous.first : this;
  }

  toList() {
    const list = [];
    for (let path = this; path; path = path.previous) list.push(path);
    return list.reverse();
  }
}

class Hypothesis {
  constructor(score, parentState, unitId, path) {
    this.score = score;
    this.parentState = parentState;
    this.unitId = unitId;
    this.path = path;
    this.state = null;
  }
}

function compareScoreDescending(a, b) {
  if (a > b) return -1;
  if (a < b) return 1;
  return 0;
}

export class BopomofoBeamSearch {
  constructor(lexicon, model) {
    this.lexicon = lexicon;
    this.model = model;
    this.stateCache = new Map();
    this.unkIndex = model.outputSize - 1;
  }

  getVocabIndex(unitId) {
    return unitId >= 0 ? this.lexicon.unitToVocab[unitId] : this.unkIndex;
  }

  getLogProb(state, unitId) {
    if (unitId >= 0) {
      return Math.fround(
        this.model.getLogit(state, this.lexicon.unitToVocab[unitId]) + this.lexicon.unkLogProbs[unitId],
      );
    }
    return Math.fround(this.model.getLogit(state, this.unkIndex) + TABLE_FILL_PENALTY);
  }

  getUnigramLogProb(unitId) {
    return unitId >= 0 ? this.lexicon.unigramLogProbs[unitId] : TABLE_FILL_UNIGRAM_LOG_PROB;
  }

  getState(hypothesis) {
    if (!hypothesis.state) {
      hypothesis.state =
        hypothesis.unitId === ROOT
          ? hypothesis.parentState
          : this.model.advance(hypothesis.parentState, this.getVocabIndex(hypothesis.unitId));
    }
    return hypothesis.state;
  }

  getCachedState(parentState, vocabIndex) {
    const key = `${parentState.id}:${vocabIndex}`;
    const cached = this.stateCache.get(key);
    if (cached) {
      this.stateCache.delete(key);
      this.stateCache.set(key, cached);
    }
    return cached;
  }

  putCachedState(parentState, vocabIndex, state) {
    const key = `${parentState.id}:${vocabIndex}`;
    this.stateCache.delete(key);
    this.stateCache.set(key, state);
    if (this.stateCache.size > STATE_CACHE_LIMIT) {
      this.stateCache.delete(this.stateCache.keys().next().value);
    }
  }

  decode(graph, length, contextState, beamWidth = 8, maxSentences = 5, isCancelled = () => false) {
    const { edges, target } = graph;
    const futureScores = new Array(target + 1).fill(-Infinity);
    futureScores[target] = 0;
    for (let start = target - 1; start >= 0; start--) {
      const startEdges = edges[start];
      for (let index = 0; index < startEdges.length; index += 2) {
        const score = this.getUnigramLogProb(startEdges[index]) + futureScores[startEdges[index + 1]];
        if (score > futureScores[start]) futureScores[start] = score;
      }
    }

    const hypotheses = Array.from({ length: target + 1 }, () => []);
    hypotheses[0].push(new Hypothesis(0, contextState, ROOT, null));
    let firstLogProbs = null;
    for (let start = 0; start < target; start++) {
      if (isCancelled()) return null;
      const startEdges = edges[start];
      if (hypotheses[start].length === 0 || startEdges.length === 0) continue;
      const kept = [...hypotheses[start]].sort((a, b) => compareScoreDescending(a.score, b.score)).slice(0, beamWidth);

      const pending = [];
      for (const hypothesis of kept) {
        if (hypothesis.state || hypothesis.unitId === ROOT) continue;
        const cached = this.getCachedState(hypothesis.parentState, this.getVocabIndex(hypothesis.unitId));
        if (cached) hypothesis.state = cached;
        else pending.push(hypothesis);
      }
      if (pending.length > 0) {
        const vocabIndices = pending.map((hypothesis) => this.getVocabIndex(hypothesis.unitId));
        const states = this.model.stepBatch(
          pending.map((hypothesis) => hypothesis.parentState.hidden),
          vocabIndices,
        );
        pending.forEach((hypothesis, index) => {
          hypothesis.state = states[index];
          this.putCachedState(hypothesis.parentState, vocabIndices[index], states[index]);
        });
      }
      for (const hypothesis of kept) {
        const state = this.getState(hypothesis);
        const logProbs = new Float32Array(startEdges.length / 2);
        for (let index = 0; index < logProbs.length; index++) logProbs[index] = this.getLogProb(state, startEdges[2 * index]);
        if (start === 0) firstLogProbs = logProbs;
        for (let index = 0; index < logProbs.length; index++) {
          const unitId = startEdges[2 * index];
          const end = startEdges[2 * index + 1];
          hypotheses[end].push(
            new Hypothesis(hypothesis.score + logProbs[index], state, unitId, new Path(unitId, end, hypothesis.path)),
          );
        }
      }
      hypotheses[start] = kept;
    }

    const finals = [...hypotheses[target]]
      .sort((a, b) => compareScoreDescending(a.score, b.score))
      .slice(0, Math.max(beamWidth, maxSentences));
    const sentences =
      target === length ? finals.slice(0, maxSentences).map((hypothesis) => [hypothesis.score, hypothesis.path]) : [];

    const exactScores = new Map();
    for (const hypothesis of finals) {
      const firstStep = hypothesis.path.first;
      const pathKey = firstStep.unitId * PATH_KEY_SHIFT + firstStep.end;
      if ((exactScores.get(pathKey) ?? -Infinity) < hypothesis.score) exactScores.set(pathKey, hypothesis.score);
    }
    const bestByText = new Map();
    const addCandidate = (score, text, unitId, end) => {
      const textKey = `${text}\u0000${end}`;
      const existing = bestByText.get(textKey);
      if (!existing || existing.score < score) bestByText.set(textKey, { score, text, unitId, end });
    };

    const startEdges = edges[0];
    for (let index = 0; index < startEdges.length / 2; index++) {
      const unitId = startEdges[2 * index];
      const end = startEdges[2 * index + 1];
      const score = exactScores.get(unitId * PATH_KEY_SHIFT + end) ?? firstLogProbs[index] + futureScores[end];
      if (unitId >= 0) addCandidate(score, this.lexicon.getText(unitId), unitId, end);
      else {
        this.lexicon.getTableChars(graph.tableFillTokens[-unitId - 1]).forEach(({ char, unitId: charUnitId }, order) => {
          addCandidate(score - 1e-6 * order, char, charUnitId, end);
        });
      }
    }

    const firstCandidates = [...bestByText.values()].sort((a, b) => {
      const byScore = compareScoreDescending(a.score, b.score);
      if (byScore !== 0) return byScore;
      const byUnigram = compareScoreDescending(this.getUnigramLogProb(a.unitId), this.getUnigramLogProb(b.unitId));
      if (byUnigram !== 0) return byUnigram;
      return a.unitId - b.unitId;
    });
    return {
      sentences: sentences.map(([score, path]) => ({ score, path })),
      firstCandidates,
    };
  }
}
