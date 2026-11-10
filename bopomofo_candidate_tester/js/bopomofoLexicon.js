import { BufferReader } from "./bufferReader.js";
import { getSyllableKey, tokenMatches } from "./bopomofoRules.js";

const UNIT_ID_SHIFT = 1048576;

export class BopomofoLexicon {
  constructor(buffer, rules) {
    this.rules = rules;
    const reader = new BufferReader(buffer);
    this.unitCount = reader.readInt();
    this.vocabSize = reader.readInt();
    this.textOffsets = reader.readInts(this.unitCount + 1);
    this.textBytes = reader.readBytes(reader.readInt());
    this.syllableOffsets = reader.readInts(this.unitCount + 1);
    this.unitSyllableIds = reader.readShorts(this.syllableOffsets[this.unitCount]);
    this.unitToVocab = reader.readInts(this.unitCount);
    this.unkLogProbs = reader.readFloats(this.unitCount);
    this.unigramLogProbs = reader.readFloats(this.unitCount);
    const nodeCount = reader.readInt();
    this.childOffsets = reader.readInts(nodeCount + 1);
    this.childSyllableIds = reader.readShorts(this.childOffsets[nodeCount]);
    this.childNodes = reader.readInts(this.childOffsets[nodeCount]);
    this.nodeUnitOffsets = reader.readInts(nodeCount + 1);
    this.nodeUnits = reader.readInts(this.nodeUnitOffsets[nodeCount]);

    this.textDecoder = new TextDecoder("utf-8");
    this.texts = new Array(this.unitCount).fill(null);

    this.syllableIds = new Map();
    this.rules.syllables.forEach((syllable, id) => this.syllableIds.set(getSyllableKey(syllable), id));

    this.singleCharUnits = new Map();
    for (let unitId = 0; unitId < this.unitCount; unitId++) {
      if (this.getSyllableCount(unitId) === 1) {
        const key = `${this.getText(unitId)}\u0000${this.getSyllableIds(unitId)[0]}`;
        if (!this.singleCharUnits.has(key)) this.singleCharUnits.set(key, unitId);
      }
    }

    this.tableCharCache = new Map();
    this.readingCharCache = new Map();
  }

  getText(unitId) {
    let text = this.texts[unitId];
    if (text === null) {
      text = this.textDecoder.decode(this.textBytes.subarray(this.textOffsets[unitId], this.textOffsets[unitId + 1]));
      this.texts[unitId] = text;
    }
    return text;
  }

  getSyllableCount(unitId) {
    return this.syllableOffsets[unitId + 1] - this.syllableOffsets[unitId];
  }

  getSyllableIds(unitId) {
    const start = this.syllableOffsets[unitId];
    return Array.from(this.unitSyllableIds.subarray(start, start + this.getSyllableCount(unitId)));
  }

  getCharUnitId(char, syllable) {
    const syllableId = this.syllableIds.get(getSyllableKey(syllable));
    if (syllableId === undefined) return -1;
    return this.singleCharUnits.get(`${char}\u0000${syllableId}`) ?? -1;
  }

  getTableChars(token) {
    const key = `${token.initial}\u0000${token.remain}\u0000${token.tone}`;
    let result = this.tableCharCache.get(key);
    if (!result) {
      result = this.getChars(this.rules.getTableSyllables(token));
      this.tableCharCache.set(key, result);
    }
    return result;
  }

  getReadingChars(token) {
    const key = `${token.initial}\u0000${token.remain}\u0000${token.tone}`;
    let result = this.readingCharCache.get(key);
    if (!result) {
      result = this.getChars(this.rules.getReadingSyllables(token));
      this.readingCharCache.set(key, result);
    }
    return result;
  }

  getChars(syllables) {
    const chars = new Map();
    for (const syllable of syllables) {
      for (const char of this.rules.charsBySyllable.get(getSyllableKey(syllable)).chars) {
        if (!chars.has(char)) chars.set(char, this.getCharUnitId(char, syllable));
      }
    }
    return [...chars.entries()].map(([char, unitId]) => ({ char, unitId }));
  }

  buildGraph(input) {
    const length = input.length;
    const tokensByPosition = Array.from({ length }, (_, position) => this.rules.getTokensAt(input, position));
    const eaten = new Array(length).fill(false);
    for (let position = 0; position < length; position++) {
      if (eaten[position]) {
        tokensByPosition[position] = [];
        continue;
      }
      for (const token of tokensByPosition[position]) {
        for (let covered = position + 1; covered < Math.min(token.end, length); covered++) eaten[covered] = true;
      }
    }
    const edges = Array.from({ length }, () => []);
    const tableFillTokens = [];
    const reachable = new Array(length + 1).fill(false);
    if (length > 0) reachable[0] = true;
    let edgeBuffer = [];
    const seenEdges = new Set();
    const syllables = this.rules.syllables;

    const collectUnits = (position, node, depth) => {
      if (depth > 0) {
        for (let index = this.nodeUnitOffsets[node]; index < this.nodeUnitOffsets[node + 1]; index++) {
          const unitId = this.nodeUnits[index];
          const edgeKey = unitId * UNIT_ID_SHIFT + position;
          if (!seenEdges.has(edgeKey)) {
            seenEdges.add(edgeKey);
            edgeBuffer.push(unitId, position);
          }
        }
      }
      if (position === length) return;
      for (const token of tokensByPosition[position]) {
        for (let child = this.childOffsets[node]; child < this.childOffsets[node + 1]; child++) {
          if (tokenMatches(token, syllables[this.childSyllableIds[child]])) {
            collectUnits(token.end, this.childNodes[child], depth + 1);
          }
        }
      }
    };

    for (let start = 0; start < length; start++) {
      if (!reachable[start]) continue;
      edgeBuffer = [];
      seenEdges.clear();
      collectUnits(start, 0, 0);
      for (const token of tokensByPosition[start]) {
        if (this.rules.getTableSyllables(token).length > 0) {
          tableFillTokens.push(token);
          edgeBuffer.push(-tableFillTokens.length, token.end);
        }
      }
      edges[start] = edgeBuffer;
      for (let index = 1; index < edges[start].length; index += 2) reachable[edges[start][index]] = true;
    }

    let target = 0;
    if (reachable[length]) target = length;
    else {
      for (let position = length; position >= 1; position--) {
        if (reachable[position]) {
          target = position;
          break;
        }
      }
    }
    const tableFillCost = new Array(length + 1).fill(Infinity);
    tableFillCost[target] = 0;
    for (let start = target - 1; start >= 0; start--) {
      const startEdges = edges[start];
      for (let index = 0; index < startEdges.length; index += 2) {
        const end = startEdges[index + 1];
        if (end <= target && tableFillCost[end] !== Infinity) {
          const cost = (startEdges[index] < 0 ? 1 : 0) + tableFillCost[end];
          tableFillCost[start] = Math.min(tableFillCost[start], cost);
        }
      }
    }
    for (let start = 0; start < length; start++) {
      const startEdges = edges[start];
      const keptEdges = [];
      if (start < target && tableFillCost[start] !== Infinity) {
        for (let index = 0; index < startEdges.length; index += 2) {
          const end = startEdges[index + 1];
          if (
            end <= target &&
            tableFillCost[end] !== Infinity &&
            (startEdges[index] < 0 ? 1 : 0) + tableFillCost[end] === tableFillCost[start]
          ) {
            keptEdges.push(startEdges[index], end);
          }
        }
      }
      edges[start] = keptEdges;
    }
    return { edges, tableFillTokens, target };
  }
}
