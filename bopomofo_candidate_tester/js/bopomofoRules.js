export const TONE_MARK = ["", "ˉ", "ˊ", "ˇ", "ˋ", "˙"];
export const MARK_TONE = new Map([["ˉ", 1], ["ˊ", 2], ["ˇ", 3], ["ˋ", 4], ["˙", 5]]);
export const STANDALONE = new Set("ㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ");

export const InputKind = {
  EMPTY: "EMPTY",
  SINGLE_SYLLABLE: "SINGLE_SYLLABLE",
  MIXED: "MIXED",
  INVALID: "INVALID",
};

export function readCsvRows(text) {
  return text
    .split("\n")
    .slice(1)
    .map((line) => line.replace(/\r+$/, ""))
    .filter((line) => line.length > 0)
    .map((line) => line.split(","));
}

export function getSyllableKey(syllable) {
  return `${syllable.initial}\u0000${syllable.remain}\u0000${syllable.tone}`;
}

export function tokenMatches(token, syllable) {
  if (syllable.initial.length !== 1 || syllable.initial[0] !== token.initial) return false;
  if (token.remain !== null && syllable.remain !== token.remain) return false;
  return token.tone === 0 || syllable.tone === token.tone;
}

function createSyllable(row) {
  return { initial: row[1], remain: row[2], tone: Number(row[3]) };
}

function getTokenKey(token) {
  return `${token.initial}\u0000${token.remain}\u0000${token.tone}`;
}

export class BopomofoRules {
  constructor(charsCsv, allCharsCsv) {
    this.syllables = readCsvRows(charsCsv).map(createSyllable);

    this.charsBySyllable = new Map();
    for (const row of readCsvRows(allCharsCsv)) {
      const syllable = createSyllable(row);
      const key = getSyllableKey(syllable);
      let entry = this.charsBySyllable.get(key);
      if (!entry) {
        entry = { syllable, chars: [], seen: new Set() };
        this.charsBySyllable.set(key, entry);
      }
      if (!entry.seen.has(row[0])) {
        entry.seen.add(row[0]);
        entry.chars.push(row[0]);
      }
    }
    this.tableSyllables = [...this.charsBySyllable.values()].map((entry) => entry.syllable);
    this.tableSyllableKeys = new Set(this.charsBySyllable.keys());

    const allSyllables = new Map();
    for (const syllable of [...this.syllables, ...this.tableSyllables]) {
      allSyllables.set(getSyllableKey(syllable), syllable);
    }
    this.initials = new Set();
    this.tonesByInitial = new Map();
    this.tonesByFinal = new Map();
    for (const syllable of allSyllables.values()) {
      const initial = syllable.initial[0];
      this.initials.add(initial);
      if (!this.tonesByInitial.has(initial)) this.tonesByInitial.set(initial, new Set());
      this.tonesByInitial.get(initial).add(syllable.tone);
      const finalKey = `${initial}\u0000${syllable.remain}`;
      if (!this.tonesByFinal.has(finalKey)) {
        this.tonesByFinal.set(finalKey, { initial, remain: syllable.remain, tones: new Set() });
      }
      this.tonesByFinal.get(finalKey).tones.add(syllable.tone);
    }
    this.remainsByInitial = new Map();
    for (const { initial, remain } of this.tonesByFinal.values()) {
      if (remain.length === 0) continue;
      if (!this.remainsByInitial.has(initial)) this.remainsByInitial.set(initial, []);
      this.remainsByInitial.get(initial).push(remain);
    }

    this.completeSyllables = new Map();
    for (const syllable of this.tableSyllables) {
      this.completeSyllables.set(syllable.initial + syllable.remain + TONE_MARK[syllable.tone], syllable);
    }

    this.tableSyllableCache = new Map();
    this.readingSyllableCache = new Map();
  }

  getCompleteSyllable(input) {
    return this.completeSyllables.get(input) ?? null;
  }

  classify(input, continuation = false) {
    if (input.length === 0) return InputKind.EMPTY;
    if (!continuation && this.completeSyllables.has(input)) return InputKind.SINGLE_SYLLABLE;
    if (this.getTokensAt(input, 0).length > 0) return InputKind.MIXED;
    return InputKind.INVALID;
  }

  getTableSyllables(token) {
    const key = getTokenKey(token);
    let result = this.tableSyllableCache.get(key);
    if (!result) {
      const matched = this.tableSyllables.filter(
        (syllable) => syllable.initial[0] === token.initial && (token.remain === null || syllable.remain === token.remain),
      );
      result = token.tone !== 0 ? matched.filter((syllable) => syllable.tone === token.tone) : matched;
      this.tableSyllableCache.set(key, result);
    }
    return result;
  }

  getReadingSyllables(token) {
    const key = getTokenKey(token);
    let result = this.readingSyllableCache.get(key);
    if (!result) {
      const remain = token.remain ?? "";
      result = this.tableSyllables.filter(
        (syllable) =>
          syllable.initial[0] === token.initial &&
          syllable.remain === remain &&
          (token.tone === 0 || syllable.tone === token.tone),
      );
      this.readingSyllableCache.set(key, result);
    }
    return result;
  }

  getTokensAt(input, position) {
    const tokens = [];
    const initial = input[position];
    if (!this.initials.has(initial)) return tokens;
    const forms = [{ remain: null, end: position + 1 }];
    for (const remain of this.remainsByInitial.get(initial) ?? []) {
      if (input.startsWith(remain, position + 1)) forms.push({ remain, end: position + 1 + remain.length });
    }
    for (const { remain, end } of forms) {
      const tones =
        remain === null
          ? this.tonesByInitial.get(initial)
          : this.tonesByFinal.get(`${initial}\u0000${remain}`).tones;
      if (end < input.length && MARK_TONE.has(input[end])) {
        const tone = MARK_TONE.get(input[end]);
        const lockedKey = getSyllableKey({ initial, remain: "", tone });
        if (remain === null && STANDALONE.has(initial) && this.tableSyllableKeys.has(lockedKey)) {
          tokens.push({ end: end + 1, initial, remain: "", tone });
        } else if (tones.has(tone)) {
          tokens.push({ end: end + 1, initial, remain, tone });
        }
      } else {
        tokens.push({ end, initial, remain, tone: 0 });
      }
    }
    return tokens;
  }
}
