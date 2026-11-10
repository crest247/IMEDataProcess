import { BopomofoBeamSearch } from "./bopomofoBeamSearch.js";
import { BopomofoLexicon } from "./bopomofoLexicon.js";
import { BopomofoModel } from "./bopomofoModel.js";
import { BopomofoRules, InputKind } from "./bopomofoRules.js";

const CONTEXT_STATE_CACHE_LIMIT = 32;

function createCandidate(text, unitId, consumed, score = NaN) {
  return { text, unitId, consumed, score };
}

function createResult(input, kind, sentences = [], candidates = []) {
  return { input, kind, sentences, candidates };
}

export class BopomofoDecoder {
  constructor() {
    this.rules = null;
    this.lexicon = null;
    this.model = null;
    this.beamSearch = null;
    this.contextStateCache = new Map();
    this.isLoaded = false;
    this.beamWidth = 8;
    this.maxSentences = 3;
    this.maxCandidates = 60;
  }

  async loadData(baseUrl) {
    const [chars, allChars, units, model] = await Promise.all([
      fetch(`${baseUrl}bopomofo_chars.csv`).then((response) => response.text()),
      fetch(`${baseUrl}bopomofo_all_chars.csv`).then((response) => response.text()),
      fetch(`${baseUrl}bopomofo_units.bin`).then((response) => response.arrayBuffer()),
      fetch(`${baseUrl}bopomofo_model.bin`).then((response) => response.arrayBuffer()),
    ]);
    this.loadFrom({ chars, allChars, units, model });
  }

  loadFrom({ chars, allChars, units, model }) {
    this.rules = new BopomofoRules(chars, allChars);
    this.lexicon = new BopomofoLexicon(units, this.rules);
    this.model = new BopomofoModel(model);
    this.beamSearch = new BopomofoBeamSearch(this.lexicon, this.model);
    this.contextStateCache.clear();
    this.isLoaded = true;
  }

  decode(input, selectedUnitIds = [], isCancelled = () => false) {
    if (!this.isLoaded) return createResult("", InputKind.EMPTY);
    const kind = this.rules.classify(input, selectedUnitIds.length > 0);
    if (kind === InputKind.SINGLE_SYLLABLE) {
      const syllable = this.rules.getCompleteSyllable(input);
      return createResult(input, kind, [], this.getTableChars(syllable, input.length));
    }
    if (kind === InputKind.MIXED) return this.decodeMixed(input, selectedUnitIds, isCancelled);
    return createResult(input, kind);
  }

  decodeMixed(input, selectedUnitIds, isCancelled) {
    const graph = this.lexicon.buildGraph(input);
    if (graph.target === 0) return createResult(input, InputKind.INVALID);
    const searchResult = this.beamSearch.decode(
      graph,
      input.length,
      this.getContextState(selectedUnitIds),
      this.beamWidth,
      this.maxSentences,
      isCancelled,
    );
    if (searchResult === null) return null;
    const sentences = searchResult.sentences.map(({ score, path }) => {
      const units = path.toList().map((step) => {
        if (step.unitId >= 0) return createCandidate(this.lexicon.getText(step.unitId), step.unitId, step.end);
        const { char, unitId } = this.lexicon.getTableChars(graph.tableFillTokens[-step.unitId - 1])[0];
        return createCandidate(char, unitId, step.end);
      });
      return { text: units.map((unit) => unit.text).join(""), units, score };
    });
    const candidates = searchResult.firstCandidates
      .slice(0, this.maxCandidates)
      .map((candidate) => createCandidate(candidate.text, candidate.unitId, candidate.end, candidate.score));
    return createResult(input, InputKind.MIXED, sentences, [...candidates, ...this.getHomophones(input, candidates)]);
  }

  getHomophones(input, candidates) {
    const shown = new Set(candidates.map((candidate) => `${candidate.text}\u0000${candidate.consumed}`));
    let token = null;
    for (const candidateToken of this.rules.getTokensAt(input, 0)) {
      if (this.rules.getReadingSyllables(candidateToken).length === 0) continue;
      if (token === null || candidateToken.end > token.end) token = candidateToken;
    }
    if (token === null) return [];
    return this.lexicon
      .getReadingChars(token)
      .map(({ char, unitId }) => createCandidate(char, unitId, token.end))
      .filter((candidate) => {
        const key = `${candidate.text}\u0000${candidate.consumed}`;
        if (shown.has(key)) return false;
        shown.add(key);
        return true;
      });
  }

  getTableChars(syllable, consumed) {
    const { chars } = this.rules.charsBySyllable.get(`${syllable.initial}\u0000${syllable.remain}\u0000${syllable.tone}`);
    return chars.map((char) => createCandidate(char, this.lexicon.getCharUnitId(char, syllable), consumed));
  }

  getContextState(unitIds) {
    const validUnitIds = unitIds.filter((unitId) => unitId >= 0);
    const key = validUnitIds.join(",");
    const cached = this.contextStateCache.get(key);
    if (cached) {
      this.contextStateCache.delete(key);
      this.contextStateCache.set(key, cached);
      return cached;
    }
    const state =
      validUnitIds.length === 0
        ? this.model.getInitialState()
        : this.model.advance(
            this.getContextState(validUnitIds.slice(0, -1)),
            this.lexicon.unitToVocab[validUnitIds[validUnitIds.length - 1]],
          );
    this.contextStateCache.set(key, state);
    if (this.contextStateCache.size > CONTEXT_STATE_CACHE_LIMIT) {
      this.contextStateCache.delete(this.contextStateCache.keys().next().value);
    }
    return state;
  }
}
