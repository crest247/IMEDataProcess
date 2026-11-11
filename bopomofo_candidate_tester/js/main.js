import { BopomofoDecoder } from "./bopomofoDecoder.js";
import { KEYBOARD_ROWS, TONE_NAMES } from "./bopomofoKeyboardLayout.js";
import { BopomofoProcessor } from "./bopomofoProcessor.js";
import { InputKind } from "./bopomofoRules.js";

const DATA_URL = "../app_assets/";
const KIND_LABELS = new Map([
  [InputKind.SINGLE_SYLLABLE, { text: "單字完整注音", className: "b-single" }],
  [InputKind.MIXED, { text: "混打", className: "b-mix" }],
  [InputKind.INVALID, { text: "不合法", className: "b-bad" }],
]);

const decoder = new BopomofoDecoder();
let output = "";
const processor = new BopomofoProcessor(
  decoder,
  (text) => {
    output += text;
  },
  render,
);

function getElement(id) {
  return document.getElementById(id);
}

function createElement(tagName, className, text) {
  const element = document.createElement(tagName);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function buildKeyboard() {
  KEYBOARD_ROWS.forEach((row, index) => {
    const rowElement = createElement("div", `row r${index + 1}`);
    for (const radical of row) {
      const isTone = TONE_NAMES.has(radical);
      const button = createElement("button", isTone ? "key tone" : "key");
      if (isTone) {
        button.append(createElement("span", "tm", radical), createElement("small", "", TONE_NAMES.get(radical)));
      } else button.textContent = radical;
      button.addEventListener("click", () => processor.inputRadical(radical));
      rowElement.append(button);
    }
    getElement("keyboard").append(rowElement);
  });
}

function renderOutput() {
  getElement("output").textContent = output;
}

function renderCompose() {
  const chips = getElement("chips");
  chips.replaceChildren(
    ...processor.selectedCandidates.map((candidate, index) => {
      const chip = createElement("span", "chip", candidate.text);
      chip.append(createElement("small", "", processor.selectedTokens[index].join("")));
      return chip;
    }),
  );
  getElement("remain").textContent = processor.radicals.join("");
}

function renderMeta() {
  const meta = getElement("meta");
  const result = processor.lastResult;
  if (!result) {
    meta.replaceChildren();
    return;
  }
  const label = KIND_LABELS.get(result.kind);
  const items = [];
  if (label) items.push(createElement("span", `badge ${label.className}`, label.text));
  items.push(`候選 ${result.candidates.length} 個`);
  items.push(`${processor.lastDecodeMillis.toFixed(1)} ms`);
  meta.replaceChildren(...items.flatMap((item, index) => (index === 0 ? [item] : [createElement("span", "", "·"), item])));
}

function createCandidateButton(units, index) {
  const isSentence = index === 0 && processor.hasSentenceCell;
  const button = createElement("button", "cand");
  if (isSentence) button.classList.add("sent");
  if (index === processor.selectedCandidateIndex) button.classList.add("selected");
  const sublabel = isSentence
    ? units.map((unit) => unit.text).join("／")
    : processor.candidatesInput.slice(0, units[units.length - 1].consumed);
  button.append(createElement("span", "t", units.map((unit) => unit.text).join("")), createElement("span", "z", sublabel));
  button.addEventListener("click", () => processor.selectCandidate(units));
  return button;
}

function renderCandidates() {
  const candidatesElement = getElement("cands");
  const result = processor.lastResult;
  if (!processor.isComposing() || !result) {
    getElement("candLabel").textContent = "候選";
    candidatesElement.replaceChildren(createElement("span", "empty", "用下方鍵盤或實體鍵盤輸入注音"));
    return;
  }
  if (result.kind === InputKind.INVALID) {
    candidatesElement.replaceChildren(createElement("span", "empty", "不合法的輸入，沒有候選"));
    return;
  }
  getElement("candLabel").textContent =
    result.kind === InputKind.SINGLE_SYLLABLE ? `字表候選（共 ${result.candidates.length} 字）` : "候選（第一格是整句）";
  if (processor.candidates.length === 0) {
    candidatesElement.replaceChildren(createElement("span", "empty", "沒有候選"));
    return;
  }
  const buttons = processor.candidates.map((units, index) => createCandidateButton(units, index));
  candidatesElement.replaceChildren(...buttons);
  if (processor.selectedCandidateIndex > 0) buttons[processor.selectedCandidateIndex].scrollIntoView({ block: "nearest" });
}

function render() {
  renderOutput();
  renderCompose();
  renderMeta();
  renderCandidates();
}

function applySystemKey(key) {
  if (key === " ") output += " ";
  else if (key === "Enter") output += "\n";
  else if (key === "Backspace") output = [...output].slice(0, -1).join("");
  render();
}

function handleScreenKey(key, action) {
  if (!action()) applySystemKey(key);
}

function bindFunctionKeys() {
  getElement("backspace").addEventListener("click", () => handleScreenKey("Backspace", () => processor.backspace()));
  getElement("space").addEventListener("click", () => handleScreenKey(" ", () => processor.space()));
  getElement("enter").addEventListener("click", () => handleScreenKey("Enter", () => processor.enter()));
  getElement("clearComposition").addEventListener("click", () => {
    processor.clearState();
    render();
  });
  getElement("clearOutput").addEventListener("click", () => {
    output = "";
    render();
  });
  const applyPaste = () => {
    const value = getElement("pasteBox").value.replace(/\s+/g, "");
    if (value.length === 0) return;
    processor.setRadicals([...value]);
    getElement("pasteBox").value = "";
  };
  getElement("pasteButton").addEventListener("click", applyPaste);
  getElement("pasteBox").addEventListener("keydown", (event) => {
    if (event.key === "Enter") applyPaste();
  });
  for (const button of document.querySelectorAll("button")) {
    button.addEventListener("mousedown", (event) => event.preventDefault());
  }
}

function bindHardwareKeys() {
  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement) return;
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (processor.onHardwareKeyDown(event.key)) {
      event.preventDefault();
      return;
    }
    if (event.key === " " || event.key === "Enter" || event.key === "Backspace") {
      event.preventDefault();
      applySystemKey(event.key);
    }
  });
}

async function start() {
  buildKeyboard();
  bindFunctionKeys();
  render();
  const status = getElement("status");
  status.textContent = "載入資料中…";
  try {
    const startTime = performance.now();
    await decoder.loadData(DATA_URL);
    status.textContent = `資料已載入（${(performance.now() - startTime).toFixed(0)} ms）`;
    bindHardwareKeys();
    processor.updateCandidates();
  } catch (error) {
    status.textContent = `載入資料失敗：${error}`;
    status.classList.add("bad");
  }
}

start();
