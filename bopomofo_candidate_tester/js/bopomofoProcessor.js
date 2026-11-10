import { MARK_TONE } from "./bopomofoRules.js";

const RADICAL_KEY_MAP = new Map([
  ["1", "ㄅ"], ["q", "ㄆ"], ["a", "ㄇ"], ["z", "ㄈ"],
  ["2", "ㄉ"], ["w", "ㄊ"], ["s", "ㄋ"], ["x", "ㄌ"],
  ["e", "ㄍ"], ["d", "ㄎ"], ["c", "ㄏ"],
  ["r", "ㄐ"], ["f", "ㄑ"], ["v", "ㄒ"],
  ["5", "ㄓ"], ["t", "ㄔ"], ["g", "ㄕ"], ["b", "ㄖ"],
  ["y", "ㄗ"], ["h", "ㄘ"], ["n", "ㄙ"],
  ["u", "ㄧ"], ["j", "ㄨ"], ["m", "ㄩ"],
  ["8", "ㄚ"], ["i", "ㄛ"], ["k", "ㄜ"], [",", "ㄝ"],
  ["9", "ㄞ"], ["o", "ㄟ"], ["l", "ㄠ"], [".", "ㄡ"],
  ["0", "ㄢ"], ["p", "ㄣ"], [";", "ㄤ"], ["/", "ㄥ"], ["-", "ㄦ"],
  ["6", "ˊ"], ["3", "ˇ"], ["4", "ˋ"], ["7", "˙"],
]);

export class BopomofoProcessor {
  constructor(decoder, onCommit, onChange) {
    this.decoder = decoder;
    this.onCommit = onCommit;
    this.onChange = onChange;
    this.radicals = [];
    this.selectedCandidates = [];
    this.selectedTokens = [];
    this.candidates = [];
    this.selectedCandidateIndex = 0;
    this.candidatesInput = "";
    this.lastResult = null;
    this.lastDecodeMillis = 0;
    this.hasSentenceCell = false;
  }

  isComposing() {
    return this.radicals.length > 0 || this.selectedCandidates.length > 0;
  }

  inputRadical(radical) {
    this.radicals.push(radical);
    this.updateCandidates();
  }

  setRadicals(radicals) {
    this.clearState();
    this.radicals = [...radicals];
    this.updateCandidates();
  }

  backspace() {
    if (this.radicals.length === 0) return false;
    this.radicals.pop();
    if (this.radicals.length === 0 && this.selectedCandidates.length > 0) {
      this.selectedCandidates.pop();
      this.radicals.push(...this.selectedTokens.pop());
    }
    this.updateCandidates();
    return true;
  }

  space() {
    if (!this.isComposing()) return false;
    this.selectCurrentCandidate();
    return true;
  }

  enter() {
    return this.space();
  }

  hardwareSpace() {
    if (this.radicals.length > 0 && !MARK_TONE.has(this.radicals[this.radicals.length - 1][0])) {
      this.inputRadical("ˉ");
      return true;
    }
    return this.space();
  }

  moveSelection(direction) {
    if (this.candidates.length === 0) return false;
    if (direction === 2) this.selectedCandidateIndex = Math.min(this.selectedCandidateIndex + 1, this.candidates.length - 1);
    if (direction === 4) this.selectedCandidateIndex = Math.max(this.selectedCandidateIndex - 1, 0);
    this.onChange();
    return true;
  }

  selectCurrentCandidate() {
    if (this.candidates.length > 0) this.selectCandidate(this.candidates[this.selectedCandidateIndex]);
  }

  selectCandidate(units) {
    const end = units[units.length - 1].consumed;
    const input = this.radicals.join("");
    if (end > input.length || !input.startsWith(this.candidatesInput.slice(0, end))) return;
    let start = 0;
    for (const unit of units) {
      this.selectedCandidates.push(unit);
      this.selectedTokens.push(this.radicals.slice(start, unit.consumed));
      start = unit.consumed;
    }
    this.radicals.splice(0, start);
    if (this.radicals.length === 0) {
      this.onCommit(this.selectedCandidates.map((candidate) => candidate.text).join(""));
      this.clearState();
      this.onChange();
    } else this.updateCandidates();
  }

  clearState() {
    this.radicals = [];
    this.selectedCandidates = [];
    this.selectedTokens = [];
    this.candidates = [];
    this.candidatesInput = "";
    this.selectedCandidateIndex = 0;
    this.lastResult = null;
    this.hasSentenceCell = false;
  }

  updateCandidates() {
    const input = this.radicals.join("");
    if (input.length === 0) {
      this.setCandidates("", [], null);
      return;
    }
    const start = performance.now();
    const result = this.decoder.decode(
      input,
      this.selectedCandidates.map((candidate) => candidate.unitId),
    );
    this.lastDecodeMillis = performance.now() - start;
    const sentence = result.sentences.length > 0 ? result.sentences[0].units : null;
    const singleSentenceUnit = sentence !== null && sentence.length === 1 ? sentence[0] : null;
    const list = result.candidates
      .filter(
        (candidate) =>
          !(singleSentenceUnit && singleSentenceUnit.text === candidate.text && singleSentenceUnit.consumed === candidate.consumed),
      )
      .map((candidate) => [candidate]);
    this.hasSentenceCell = sentence !== null;
    this.setCandidates(input, sentence ? [sentence, ...list] : list, result);
  }

  setCandidates(input, list, result) {
    this.candidatesInput = input;
    this.candidates = list;
    this.selectedCandidateIndex = 0;
    this.lastResult = result;
    this.onChange();
  }

  onHardwareKeyDown(key) {
    if (key === "ArrowRight") return this.moveSelection(2) || this.isComposing();
    if (key === "ArrowLeft") return this.moveSelection(4) || this.isComposing();
    if (key === "ArrowUp" || key === "ArrowDown") return this.isComposing();
    if (key === "Backspace") return this.backspace();
    if (key === " ") return this.hardwareSpace();
    if (key === "Enter") return this.enter();
    const radical = RADICAL_KEY_MAP.get(key);
    if (radical === undefined) return false;
    this.inputRadical(radical);
    return true;
  }
}
