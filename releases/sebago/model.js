/* Sebago's two-layer causal decoder. Matrices use PyTorch row-major layout. */
const SebagoModel = (() => {
  const width = 48, headWidth = 12;

  function read(buffer) {
    const headerSize = Number(new DataView(buffer).getBigUint64(0, true));
    const dataStart = headerSize + 8;
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerSize)));
    const weights = {};
    let count = 0;
    for (const [name, info] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      const [start, end] = info.data_offsets;
      if (info.dtype !== "F32" || end - start !== info.shape.reduce((n, x) => n * x, 1) * 4) throw Error("Invalid model file");
      weights[name] = new Float32Array(buffer, dataStart + start, (end - start) / 4);
      count += weights[name].length;
    }
    if (count !== 40530) throw Error("Unexpected parameter count");
    return weights;
  }

  function linear(x, weights, bias) {
    const out = new Float32Array(bias.length);
    for (let i = 0; i < out.length; i++) {
      let y = bias[i], offset = i * x.length;
      for (let j = 0; j < x.length; j++) y += weights[offset + j] * x[j];
      out[i] = y;
    }
    return out;
  }

  function residualNorm(x, skip, weight, bias) {
    let mean = 0, variance = 0;
    for (let j = 0; j < width; j++) { x[j] += skip[j]; mean += x[j]; }
    mean /= width;
    for (const value of x) variance += (value - mean) ** 2;
    const scale = 1 / Math.sqrt(variance / width + 1e-5);
    return x.map((v, j) => (v - mean) * scale * weight[j] + bias[j]);
  }

  function lastLogits(tokens, w) {
    let states = tokens.map((id, i) => {
      const x = w["tokens.weight"].slice(id * width, (id + 1) * width);
      for (let j = 0; j < width; j++) x[j] += w["positions.weight"][i * width + j];
      return x;
    });
    for (let layer = 0; layer < 2; layer++) {
      const param = key => w["layers." + layer + "." + key];
      const qkv = states.map(x => linear(x, param("self_attn.in_proj_weight"), param("self_attn.in_proj_bias")));
      states = states.map((skip, t) => {
        const attention = new Float32Array(width);
        for (let h = 0; h < width; h += headWidth) {
          const scores = new Float64Array(t + 1);
          for (let i = 0; i <= t; i++) {
            let dot = 0;
            for (let j = h; j < h + headWidth; j++) dot += qkv[t][j] * qkv[i][width + j];
            scores[i] = dot / Math.sqrt(headWidth);
          }
          const peak = Math.max(...scores);
          let mass = 0;
          for (let i = 0; i <= t; i++) { scores[i] = Math.exp(scores[i] - peak); mass += scores[i]; }
          for (let i = 0; i <= t; i++)
            for (let j = h; j < h + headWidth; j++) attention[j] += scores[i] / mass * qkv[i][2 * width + j];
        }
        const mixed = linear(attention, param("self_attn.out_proj.weight"), param("self_attn.out_proj.bias"));
        const y = residualNorm(mixed, skip, param("norm1.weight"), param("norm1.bias"));
        const ff = linear(y, param("linear1.weight"), param("linear1.bias"));
        for (let j = 0; j < ff.length; j++) ff[j] = Math.max(0, ff[j]);
        return residualNorm(linear(ff, param("linear2.weight"), param("linear2.bias")), y, param("norm2.weight"), param("norm2.bias"));
      });
    }
    return linear(states[states.length-1], w["output.weight"], w["output.bias"]);
  }

  function generate(prefix, weights) {
    const seq = [...prefix], output = [];
    for (let i = 0; i < 4; i++) {
      const logits = lastLogits(seq, weights);
      const token = logits.indexOf(Math.max(...logits));
      output.push(token);
      if (token === 16) break;
      seq.push(token);
    }
    return output;
  }
  return { read, generate };
})();
if (typeof module !== "undefined") module.exports = SebagoModel;
