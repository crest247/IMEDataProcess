export class BufferReader {
  constructor(buffer) {
    this.buffer = buffer;
    this.view = new DataView(buffer);
    this.position = 0;
  }

  readInt() {
    const value = this.view.getInt32(this.position, true);
    this.position += 4;
    return value;
  }

  readInts(count) {
    const result = new Int32Array(this.buffer.slice(this.position, this.position + 4 * count));
    this.position += 4 * count;
    return result;
  }

  readShorts(count) {
    const result = new Uint16Array(this.buffer.slice(this.position, this.position + 2 * count));
    this.position += 2 * count;
    return result;
  }

  readFloats(count) {
    const result = new Float32Array(this.buffer.slice(this.position, this.position + 4 * count));
    this.position += 4 * count;
    return result;
  }

  readBytes(count) {
    const result = new Uint8Array(this.buffer.slice(this.position, this.position + count));
    this.position += count;
    return result;
  }

  readSignedBytes(count) {
    const result = new Int8Array(this.buffer.slice(this.position, this.position + count));
    this.position += count;
    return result;
  }
}
