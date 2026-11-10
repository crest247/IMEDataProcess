import { BufferReader } from "./bufferReader.js";

class QuantizedMatrix {
  constructor(rows, cols, scales, values) {
    this.rows = rows;
    this.cols = cols;
    this.scales = scales;
    this.values = values;
  }

  dotRow(row, vector) {
    let sum = 0;
    let offset = row * this.cols;
    for (let col = 0; col < this.cols; col++) sum += this.values[offset++] * vector[col];
    return sum * this.scales[row];
  }
}

class FloatMatrix {
  constructor(matrix) {
    this.rows = matrix.rows;
    this.cols = matrix.cols;
    this.weights = new Float32Array(matrix.rows * matrix.cols);
    for (let index = 0; index < this.weights.length; index++) {
      this.weights[index] = matrix.values[index] * matrix.scales[Math.floor(index / matrix.cols)];
    }
  }

  multiplyBatch(inputs, batchSize, bias, outputs) {
    const { rows, cols, weights } = this;
    for (let row = 0; row < rows; row++) {
      const rowOffset = row * cols;
      const rowBias = bias[row];
      for (let batch = 0; batch < batchSize; batch++) {
        let sum = 0;
        const inputOffset = batch * cols;
        for (let col = 0; col < cols; col++) sum += weights[rowOffset + col] * inputs[inputOffset + col];
        outputs[batch * rows + row] = sum + rowBias;
      }
    }
  }
}

let nextStateId = 0;

class ModelState {
  constructor(hidden, output) {
    this.id = nextStateId++;
    this.hidden = hidden;
    this.output = output;
  }
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

export class BopomofoModel {
  constructor(buffer) {
    const reader = new BufferReader(buffer);
    this.layerCount = reader.readInt();
    this.hiddenSize = reader.readInt();
    this.embeddingSize = reader.readInt();
    const readMatrix = () => {
      const rows = reader.readInt();
      const cols = reader.readInt();
      const scales = reader.readFloats(rows);
      const values = reader.readSignedBytes(rows * cols);
      return new QuantizedMatrix(rows, cols, scales, values);
    };
    const readVector = () => reader.readFloats(reader.readInt());
    this.embedding = readMatrix();
    this.layers = [];
    for (let layer = 0; layer < this.layerCount; layer++) {
      this.layers.push({
        inputWeights: new FloatMatrix(readMatrix()),
        hiddenWeights: new FloatMatrix(readMatrix()),
        inputBias: readVector(),
        hiddenBias: readVector(),
      });
    }
    this.projection = new FloatMatrix(readMatrix());
    this.projectionBias = readVector();
    this.outputBias = readVector();
    this.outputSize = this.outputBias.length;
    this.bosIndex = this.embedding.rows - 1;
  }

  getInitialState() {
    const hidden = Array.from({ length: this.layerCount }, () => new Float32Array(this.hiddenSize));
    return this.stepBatch([hidden], [this.bosIndex])[0];
  }

  advance(state, vocabIndex) {
    return this.stepBatch([state.hidden], [vocabIndex])[0];
  }

  stepBatch(hiddenStates, vocabIndices) {
    const { embeddingSize, hiddenSize, layerCount, embedding } = this;
    const batchSize = vocabIndices.length;
    let inputs = new Float32Array(batchSize * embeddingSize);
    for (let batch = 0; batch < batchSize; batch++) {
      const vocabIndex = vocabIndices[batch];
      const scale = embedding.scales[vocabIndex];
      for (let col = 0; col < embeddingSize; col++) {
        inputs[batch * embeddingSize + col] = embedding.values[vocabIndex * embeddingSize + col] * scale;
      }
    }
    const newHidden = Array.from({ length: batchSize }, () => new Array(layerCount));
    const inputGates = new Float32Array(batchSize * 3 * hiddenSize);
    const hiddenGates = new Float32Array(batchSize * 3 * hiddenSize);
    const previousHidden = new Float32Array(batchSize * hiddenSize);
    for (let layerIndex = 0; layerIndex < layerCount; layerIndex++) {
      const layer = this.layers[layerIndex];
      for (let batch = 0; batch < batchSize; batch++) {
        previousHidden.set(hiddenStates[batch][layerIndex], batch * hiddenSize);
      }
      layer.inputWeights.multiplyBatch(inputs, batchSize, layer.inputBias, inputGates);
      layer.hiddenWeights.multiplyBatch(previousHidden, batchSize, layer.hiddenBias, hiddenGates);
      const nextInputs = new Float32Array(batchSize * hiddenSize);
      for (let batch = 0; batch < batchSize; batch++) {
        const gateOffset = batch * 3 * hiddenSize;
        const hidden = new Float32Array(hiddenSize);
        for (let unit = 0; unit < hiddenSize; unit++) {
          const resetGate = sigmoid(inputGates[gateOffset + unit] + hiddenGates[gateOffset + unit]);
          const updateGate = sigmoid(
            inputGates[gateOffset + hiddenSize + unit] + hiddenGates[gateOffset + hiddenSize + unit],
          );
          const candidate = Math.tanh(
            inputGates[gateOffset + 2 * hiddenSize + unit] +
              resetGate * hiddenGates[gateOffset + 2 * hiddenSize + unit],
          );
          hidden[unit] = (1 - updateGate) * candidate + updateGate * previousHidden[batch * hiddenSize + unit];
        }
        newHidden[batch][layerIndex] = hidden;
        nextInputs.set(hidden, batch * hiddenSize);
      }
      inputs = nextInputs;
    }
    const outputs = new Float32Array(batchSize * embeddingSize);
    this.projection.multiplyBatch(inputs, batchSize, this.projectionBias, outputs);
    return Array.from(
      { length: batchSize },
      (_, batch) =>
        new ModelState(newHidden[batch], outputs.slice(batch * embeddingSize, (batch + 1) * embeddingSize)),
    );
  }

  getLogit(state, vocabIndex) {
    return this.embedding.dotRow(vocabIndex, state.output) + this.outputBias[vocabIndex];
  }
}
