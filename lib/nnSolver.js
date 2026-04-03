/**
 * Fully-connected feedforward network: step-by-step training log
 * for assignment-style output (min-max norm, forward, E=d-Out, backprop, updates).
 */

function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

export function parseLayerSizes(numInputs, hiddenPart, numOutputs) {
  const hidden = String(hiddenPart || "")
    .split(/[,;\s]+/)
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => !Number.isNaN(n) && n > 0);
  if (numInputs < 1) throw new Error("En az 1 giriş özelliği gerekli.");
  if (numOutputs < 1) throw new Error("En az 1 çıkış nöronu gerekli.");
  return [numInputs, ...hidden, numOutputs];
}

export function activationFns(name) {
  const n = String(name || "sigmoid").toLowerCase();
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const table = {
    sigmoid: {
      label: "Sigmoid",
      f: (z) => sigmoid(z),
      fp: (z, a) => a * (1 - a),
      tex: String.raw`\sigma(z)=\frac{1}{1+e^{-z}}`,
      fpTex: String.raw`\sigma'(z)=\sigma(z)(1-\sigma(z))`,
    },
    tanh: {
      label: "Tanh",
      f: (z) => Math.tanh(z),
      fp: (z, a) => 1 - a * a,
      tex: String.raw`\tanh(z)`,
      fpTex: String.raw`1-\tanh^2(z)`,
    },
    relu: {
      label: "ReLU",
      f: (z) => (z > 0 ? z : 0),
      fp: (z, a) => (z > 0 ? 1 : 0),
      tex: String.raw`\mathrm{ReLU}(z)=\max(0,z)`,
      fpTex: String.raw`\mathrm{ReLU}'(z)=\mathbf{1}_{z>0}`,
    },
  };
  const act = table[n] || table.sigmoid;
  return { ...act, name: n in table ? n : "sigmoid" };
}

/** @param {number} x @param {number} [digits=3] */
export function fmt(x, digits = 3) {
  if (typeof x !== "number" || Number.isNaN(x)) return String(x);
  if (!Number.isFinite(x)) return x > 0 ? "∞" : "-∞";
  return x.toFixed(digits);
}

/**
 * Aktivasyon türevi: genel KaTeX + sayıların yerine konduğu satır.
 * @param {ReturnType<typeof activationFns>} act
 */
export function formatDerivativeKatex(act, zVal, aVal, fpiVal, fmtDigits = 3) {
  const fd = (x) => fmt(x, fmtDigits);
  const name = act.name || "sigmoid";
  const texGeneral = act.fpTex;
  let texNumeric = "";
  if (name === "sigmoid") {
    texNumeric = `\\sigma'(z) = \\sigma(z)(1-\\sigma(z)) = y(1-y) = ${fd(aVal)}\\cdot(1-${fd(aVal)}) = ${fd(fpiVal)}`;
  } else if (name === "tanh") {
    texNumeric = `f'(z) = 1-\\tanh^2(z) = 1-y^2 = 1-(${fd(aVal)})^2 = ${fd(fpiVal)}`;
  } else {
    texNumeric =
      zVal > 0
        ? `\\mathrm{ReLU}'(z)=\\mathbf{1}_{z>0} = 1 \\quad (z=${fd(zVal)}>0) \\Rightarrow ${fd(fpiVal)}`
        : `\\mathrm{ReLU}'(z)=\\mathbf{1}_{z>0} = 0 \\quad (z=${fd(zVal)}\\le 0) \\Rightarrow ${fd(fpiVal)}`;
  }
  return {
    texGeneral,
    texNumeric,
    plain: `f'(Net) = ${fd(fpiVal)}`,
  };
}

/** Column-wise min-max using all samples (features only). */
export function computeNormalization(samples) {
  if (!samples.length) throw new Error("Veri seti boş.");
  const dim = samples[0].features.length;
  const mins = Array(dim).fill(Infinity);
  const maxs = Array(dim).fill(-Infinity);
  for (const s of samples) {
    for (let j = 0; j < dim; j++) {
      const v = s.features[j];
      if (v < mins[j]) mins[j] = v;
      if (v > maxs[j]) maxs[j] = v;
    }
  }
  const spans = mins.map((mn, j) => {
    const mx = maxs[j];
    const sp = mx - mn;
    return sp === 0 ? 0 : sp;
  });
  return { mins, maxs, spans, dim };
}

export function normalizeFeatures(features, norm) {
  return features.map((x, j) => {
    const { mins, spans } = norm;
    if (spans[j] === 0) return 0.5;
    return (x - mins[j]) / spans[j];
  });
}

function matVecMul(W, v) {
  return W.map((row) => row.reduce((s, wij, j) => s + wij * v[j], 0));
}

function vecSub(a, b) {
  return a.map((x, i) => x - b[i]);
}

function vecHadamard(a, b) {
  return a.map((x, i) => x * b[i]);
}

function transpose(W) {
  const r = W.length;
  const c = W[0].length;
  const T = Array.from({ length: c }, () => Array(r).fill(0));
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) T[j][i] = W[i][j];
  }
  return T;
}

function forwardPass(xNorm, W, B, act) {
  const L = W.length;
  const a = [xNorm.slice()];
  const z = [null];
  for (let l = 1; l <= L; l++) {
    const net = matVecMul(W[l - 1], a[l - 1]).map((ni, i) => ni + B[l - 1][i]);
    const out = net.map((ni) => act.f(ni));
    z.push(net);
    a.push(out);
  }
  return { a, z, output: a[L] };
}

/**
 * İleri besleme: her nöron için Net = Σ a_j w_ij + b açılımı (ödev notasyonu).
 * @returns {{ layers: Array<{ layer: number, neurons: object[] }> }}
 */
export function explainForwardPass(a, z, W, B, act, fmtDigits = 3) {
  const f = (x) => fmt(x, fmtDigits);
  const L = W.length;
  const layers = [];
  for (let lv = 1; lv <= L; lv++) {
    const wi = lv - 1;
    const neurons = [];
    for (let i = 0; i < W[wi].length; i++) {
      const terms = [];
      for (let j = 0; j < W[wi][i].length; j++) {
        const aj = a[lv - 1][j];
        const wij = W[wi][i][j];
        const prevName = lv === 1 ? `x_${j + 1}` : `h_${j + 1}^{(${lv - 1})}`;
        const prevPlain = lv === 1 ? `x${j + 1}` : `h${lv - 1},${j + 1}`;
        terms.push({
          prevPlain,
          prevTex: prevName.replace(/\^/g, "^"),
          a: aj,
          w: wij,
          product: aj * wij,
        });
      }
      const bias = B[wi][i];
      const netVal = z[lv][i];
      const outVal = a[lv][i];
      const fpVal = act.fp(z[lv][i], a[lv][i]);
      const derivKatex = formatDerivativeKatex(act, netVal, outVal, fpVal, fmtDigits);
      const sumProducts = terms.reduce((s, t) => s + t.product, 0);
      const linearPart = f(sumProducts);
      const expandedPlain = [
        ...terms.map(
          (t) => `(${f(t.a)})·(${f(t.w)}) = ${f(t.product)}`
        ),
        `b = ${f(bias)}`,
      ].join(" ; ");
      const netEquationPlain = `Net_${i + 1}^{(${lv})} = ${terms
        .map((t) => `(${f(t.a)})(${f(t.w)})`)
        .join(" + ")} + ${f(bias)} = ${f(netVal)}`;
      neurons.push({
        neuronIndex: i + 1,
        layerIndex: lv,
        netSymbol: `\\mathrm{Net}_{${i + 1}}^{(${lv})}`,
        terms,
        bias,
        sumWeightedInputs: sumProducts,
        net: netVal,
        out: outVal,
        fprime: fpVal,
        expandedPlain,
        netEquationPlain,
        linearPart,
        outEquationPlain: `f(\\mathrm{Net}_{${i + 1}}^{(${lv})}) = ${f(outVal)}`,
        fpEquationPlain: `f'(\\mathrm{Net}) = ${f(fpVal)}`,
        activationTex: act.tex,
        derivativeKatex: derivKatex,
      });
    }
    layers.push({ layer: lv, neurons });
  }
  return { layers };
}

/**
 * Geri yayılım: zincir kuralı adımları (sayısal).
 */
export function explainBackwardPass(fwd, target, W, act, bp, fmtDigits = 3) {
  const f = (x) => fmt(x, fmtDigits);
  const { a, z } = fwd;
  const { E, deltas } = bp;
  const L = W.length;
  const blocks = [];

  const outNeurons = [];
  for (let i = 0; i < a[L].length; i++) {
    const yi = a[L][i];
    const zi = z[L][i];
    const Ei = E[i];
    const di = target[i];
    const fpi = act.fp(zi, yi);
    const del = deltas[L][i];
    const derK = formatDerivativeKatex(act, zi, yi, fpi, fmtDigits);
    outNeurons.push({
      neuronIndex: i + 1,
      layerIndex: L,
      delta: del,
      steps: [
        {
          title: "Çıkış hatası (ödev)",
          tex: `E_{${i + 1}} = d_{${i + 1}} - \\mathrm{Out}_{${i + 1}}^{(${L})},\\quad \\mathrm{Out}_{${i + 1}}^{(${L})}=f\\!\\big(\\mathrm{Net}_{${i + 1}}^{(${L})}\\big)`,
          plain: `E_${i + 1} = ${f(di)} - Out_${i + 1} = ${f(di)} - ${f(yi)} = ${f(Ei)}`,
        },
        {
          title: "Aktivasyon türevi",
          tex: `f'(\\mathrm{Net}_{${i + 1}}^{(${L})})`,
          texGeneral: derK.texGeneral,
          texNumeric: derK.texNumeric,
          activationTex: act.tex,
          plain: derK.plain,
        },
        {
          title: "Zincir kuralı (δ çıkış)",
          tex: `\\delta_{${i + 1}}^{(${L})} = \\frac{\\partial E}{\\partial \\mathrm{Net}_{${i + 1}}^{(${L})}} = \\frac{\\partial E}{\\partial \\mathrm{Out}_{${i + 1}}^{(${L})}}\\cdot f'\\!\\big(\\mathrm{Net}_{${i + 1}}^{(${L})}\\big),\\quad E_{${i + 1}}=d_{${i + 1}}-\\mathrm{Out}_{${i + 1}}^{(${L})},\\ \\mathrm{Out}_{${i + 1}}^{(${L})}=f\\!\\big(\\mathrm{Net}_{${i + 1}}^{(${L})}\\big)`,
          plain: `δ_${i + 1}^(${L}) = ${f(Ei)} × ${f(fpi)} = ${f(del)}`,
          chainDetails: [
            {
              label: "Çıkış: Out, Net ve E (tam ifadeler)",
              tex: String.raw`\mathrm{Out}_i^{(L)} = f\!\big(\mathrm{Net}_i^{(L)}\big),\qquad E_i = d_i - \mathrm{Out}_i^{(L)} = d_i - f\!\big(\mathrm{Net}_i^{(L)}\big)`,
              plain: `Bu nöron: Out_${i + 1}^{(${L})} = f(Net) = ${f(yi)};  E_${i + 1} = d - Out = ${f(Ei)}.`,
            },
            {
              label: "Tam zincir (∂E/∂Net)",
              tex: String.raw`\delta_i^{(L)}=\frac{\partial E}{\partial \mathrm{Net}_i^{(L)}}=\frac{\partial E}{\partial \mathrm{Out}_i^{(L)}}\cdot\frac{\partial \mathrm{Out}_i^{(L)}}{\partial \mathrm{Net}_i^{(L)}},\quad \mathrm{Out}_i^{(L)}=f\!\big(\mathrm{Net}_i^{(L)}\big),\quad \frac{\partial \mathrm{Out}_i^{(L)}}{\partial \mathrm{Net}_i^{(L)}}=f'\!\big(\mathrm{Net}_i^{(L)}\big)`,
              plain: `Ödev: hata terimi E_i = d_i - Out_i doğrudan kullanılır; ∂Out_i/∂Net_i = f'(Net_i) = ${f(fpi)}  ⇒  δ_i = E_i · f'(Net_i) = ${f(del)}.`,
            },
            {
              label: "Sayısal (ödev)",
              tex: String.raw`\delta_i^{(L)} = E_i\cdot f'\!\big(\mathrm{Net}_i^{(L)}\big),\qquad E_i = d_i-\mathrm{Out}_i^{(L)}`,
              plain: `δ_${i + 1} = ${f(Ei)} × ${f(fpi)} = ${f(del)}`,
            },
          ],
        },
      ],
    });
  }
  blocks.push({
    layer: L,
    title: `Çıkış katmanı (${L}) — zincir kuralı`,
    neurons: outNeurons,
  });

  for (let lv = L - 1; lv >= 1; lv--) {
    const hidden = [];
    for (let j = 0; j < a[lv].length; j++) {
      const backTerms = [];
      let sum = 0;
      for (let k = 0; k < deltas[lv + 1].length; k++) {
        const wkj = W[lv][k][j];
        const dk = deltas[lv + 1][k];
        const prod = dk * wkj;
        sum += prod;
        backTerms.push({
          kIndex: k + 1,
          deltaK: dk,
          weight: wkj,
          product: prod,
          plain: `δ_${k + 1}^(${lv + 1})·W_${k + 1},${j + 1} = ${f(dk)}×${f(wkj)} = ${f(prod)}`,
        });
      }
      const zlj = z[lv][j];
      const aj = a[lv][j];
      const fpl = act.fp(zlj, aj);
      const del = deltas[lv][j];
      const sumPlain = backTerms.map((t) => f(t.product)).join(" + ");
      const derH = formatDerivativeKatex(act, zlj, aj, fpl, fmtDigits);
      const partialChainPlain = backTerms
        .map(
          (t) =>
            `(∂E/∂Net_${t.kIndex}^{(${lv + 1})})·(∂Net_${t.kIndex}^{(${lv + 1})}/∂Out_${j + 1}^{(${lv})}) = δ_${t.kIndex}·W_${t.kIndex},${j + 1} = ${f(t.product)}`
        )
        .join("; ");
      const chainDetails = [
        {
          label: "Gizli katman: Out ve Net",
          tex: String.raw`\mathrm{Out}_j^{(\ell)} = f\!\big(\mathrm{Net}_j^{(\ell)}\big),\qquad \mathrm{Net}_k^{(\ell+1)} = \sum_r W_{k,r}^{(\ell+1)}\,\mathrm{Out}_r^{(\ell)} + b_k^{(\ell+1)}`,
          plain: `Net_${j + 1}^{(${lv})} = … → Out_${j + 1}^{(${lv})} = f(Net) = ${f(aj)}.`,
        },
        {
          label: "δ tanımı (zincir kuralı)",
          tex: String.raw`\delta_j^{(\ell)}=\frac{\partial E}{\partial \mathrm{Net}_j^{(\ell)}}=\frac{\partial E}{\partial \mathrm{Out}_j^{(\ell)}}\cdot\frac{\partial \mathrm{Out}_j^{(\ell)}}{\partial \mathrm{Net}_j^{(\ell)}}`,
          plain: "",
        },
        {
          label: "∂E/∂Out — W ile sonraki katman",
          tex: String.raw`\frac{\partial E}{\partial \mathrm{Out}_j^{(\ell)}}=\sum_k\frac{\partial E}{\partial \mathrm{Net}_k^{(\ell+1)}}\cdot\frac{\partial \mathrm{Net}_k^{(\ell+1)}}{\partial \mathrm{Out}_j^{(\ell)}}=\sum_k\delta_k^{(\ell+1)}\,W_{k,j}^{(\ell+1)}`,
          plain: `Çünkü ∂Net_k^(ℓ+1)/∂Out_j^(ℓ) = W_{k,j}^(ℓ+1) (Net_k toplamındaki Out_j katsayısı).  ${partialChainPlain}`,
        },
        {
          label: "∂Out/∂Net (aktivasyon türevi)",
          tex: String.raw`\frac{\partial \mathrm{Out}_j^{(\ell)}}{\partial \mathrm{Net}_j^{(\ell)}}=f'\!\big(\mathrm{Net}_j^{(\ell)}\big)`,
          plain: `= ${f(fpl)}`,
        },
        {
          label: "Birleşik δ",
          tex: String.raw`\delta_j^{(\ell)}=\Big(\sum_k\delta_k^{(\ell+1)}\,W_{k,j}^{(\ell+1)}\Big)\cdot f'\!\big(\mathrm{Net}_j^{(\ell)}\big)`,
          plain: `δ_${j + 1}^(${lv}) = ${f(sum)} × ${f(fpl)} = ${f(del)}`,
        },
      ];
      hidden.push({
        neuronIndex: j + 1,
        layerIndex: lv,
        delta: del,
        backTerms,
        steps: [
          {
            title: "Önceki katmandan gelen ağırlıklı δ toplamı",
            tex: `\\sum_{k} \\delta_{k}^{(${lv + 1})}\\,W_{k,${j + 1}}^{(${lv + 1})}\\quad\\big(\\mathrm{Net}_k^{(${lv + 1})}=\\sum_r W_{k,r}^{(${lv + 1})}\\,\\mathrm{Out}_r^{(${lv})}+b_k^{(${lv + 1})}\\big)`,
            plain: `Σ_k δ_k·W_{k,${j + 1}} = ${sumPlain} = ${f(sum)}`,
          },
          {
            title: "Aktivasyon türevi",
            tex: `f'(\\mathrm{Net}_{${j + 1}}^{(${lv})})`,
            texGeneral: derH.texGeneral,
            texNumeric: derH.texNumeric,
            activationTex: act.tex,
            plain: derH.plain,
          },
          {
            title: "Zincir kuralı (δ gizli)",
            tex: `\\delta_{${j + 1}}^{(${lv})} = \\Big(\\sum_k \\delta_k^{(${lv + 1})}\\,W_{k,${j + 1}}^{(${lv + 1})}\\Big)\\cdot f'\\!\\big(\\mathrm{Net}_{${j + 1}}^{(${lv})}\\big),\\ \\mathrm{Out}_{${j + 1}}^{(${lv})}=f\\big(\\mathrm{Net}_{${j + 1}}^{(${lv})}\\big)`,
            plain: `δ_${j + 1}^(${lv}) = ${f(sum)} × ${f(fpl)} = ${f(del)}`,
            chainDetails,
          },
        ],
      });
    }
    blocks.push({
      layer: lv,
      title: `Gizli katman ${lv} — zincir kuralı`,
      neurons: hidden,
    });
  }

  return { blocks };
}

function backwardPass({ a, z }, target, W, act) {
  const L = W.length;
  const deltas = Array(L + 1).fill(null);
  const E = vecSub(target, a[L]);
  const dL = vecHadamard(
    E,
    z[L].map((zi, i) => act.fp(zi, a[L][i]))
  );
  deltas[L] = dL;
  for (let l = L - 1; l >= 1; l--) {
    const sum = matVecMul(transpose(W[l]), deltas[l + 1]);
    const dh = vecHadamard(
      sum,
      z[l].map((zi, i) => act.fp(zi, a[l][i]))
    );
    deltas[l] = dh;
  }
  return { E, deltas };
}

function applySampleUpdate(W, B, deltas, a, eta) {
  const L = W.length;
  const dW = W.map((Wl) => Wl.map((row) => row.slice()));
  const dB = B.map((bl) => bl.slice());
  for (let l = 1; l <= L; l++) {
    const wi = l - 1;
    for (let i = 0; i < W[wi].length; i++) {
      for (let j = 0; j < W[wi][i].length; j++) {
        const upd = eta * deltas[l][i] * a[l - 1][j];
        dW[wi][i][j] = upd;
        W[wi][i][j] += upd;
      }
      const bud = eta * deltas[l][i];
      dB[wi][i] = bud;
      B[wi][i] += bud;
    }
  }
  return { dW, dB };
}

function evaluateMse(samples, norm, W, B, act) {
  let sum = 0;
  let n = 0;
  for (const s of samples) {
    const x = normalizeFeatures(s.features, norm);
    const { output } = forwardPass(x, W, B, act);
    const e = vecSub(s.target, output);
    for (const ei of e) {
      sum += ei * ei;
      n += 1;
    }
  }
  return n ? sum / n : 0;
}

/**
 * @param {object} opts
 * @param {{features:number[], target:number[]}[]} opts.samples
 * @param {number[][][]} opts.weights - W[layer] shape n_l x n_{l-1}
 * @param {number[][]} opts.biases - B[layer] length n_l
 */
export function runTraining({
  samples,
  weights: W0,
  biases: B0,
  activationName,
  learningRate,
  epochs,
}) {
  const act = activationFns(activationName);
  const norm = computeNormalization(samples);
  let W = deepClone(W0);
  let B = deepClone(B0);

  const epochLogs = [];
  const mseStart = evaluateMse(samples, norm, W, B, act);

  for (let ep = 0; ep < epochs; ep++) {
    const sampleSteps = [];
    for (let s = 0; s < samples.length; s++) {
      const sample = samples[s];
      const xNorm = normalizeFeatures(sample.features, norm);
      const fwd = forwardPass(xNorm, W, B, act);
      const bp = backwardPass(fwd, sample.target, W, act);
      const forwardExplanation = explainForwardPass(
        fwd.a,
        fwd.z,
        W,
        B,
        act
      );
      const backwardExplanation = explainBackwardPass(
        fwd,
        sample.target,
        W,
        act,
        bp
      );
      const upd = applySampleUpdate(W, B, bp.deltas, fwd.a, learningRate);
      sampleSteps.push({
        sampleIndex: s + 1,
        rawFeatures: sample.features.slice(),
        target: sample.target.slice(),
        xNorm,
        forwardExplanation,
        backwardExplanation,
        nets: fwd.z.slice(1).map((row, li) => ({
          layer: li + 1,
          values: row.slice(),
        })),
        activations: fwd.a.map((row, li) => ({
          layer: li,
          values: row ? row.slice() : row,
        })),
        output: fwd.output.slice(),
        E: bp.E.slice(),
        deltas: bp.deltas
          .map((d, li) => (d ? { layer: li, values: d.slice() } : null))
          .filter(Boolean),
        deltaW: upd.dW.map((M, li) => ({
          fromLayer: li,
          toLayer: li + 1,
          matrix: M.map((r) => r.slice()),
        })),
        deltaB: upd.dB.map((v, li) => ({
          layer: li + 1,
          values: v.slice(),
        })),
        weightsAfter: deepClone(W),
        biasesAfter: deepClone(B),
      });
    }
    const mseAfter = evaluateMse(samples, norm, W, B, act);
    epochLogs.push({
      epoch: ep + 1,
      samples: sampleSteps,
      mseAfterEpoch: mseAfter,
    });
  }

  const layerSizes = [samples[0].features.length];
  for (let i = 0; i < W.length; i++) layerSizes.push(W[i].length);

  return {
    activation: act,
    learningRate,
    layerSizes,
    normalization: norm,
    normalizedTable: samples.map((s) => ({
      features: normalizeFeatures(s.features, norm),
      target: s.target.slice(),
    })),
    mseBeforeTraining: mseStart,
    epochs: epochLogs,
    finalWeights: W,
    finalBiases: B,
  };
}

export function parseDatasetText(text, numOutputs = 1) {
  const lines = text.trim().split(/\r?\n/).filter((l) => l.trim().length);
  if (!lines.length) throw new Error("Veri seti boş.");
  const samples = [];
  for (const line of lines) {
    const parts = line.split(/[,;\t]+/).map((x) => x.trim()).filter(Boolean);
    const nums = parts.map((p) => {
      const v = parseFloat(p);
      if (Number.isNaN(v)) throw new Error(`Geçersiz sayı: "${p}"`);
      return v;
    });
    if (nums.length < numOutputs + 1) {
      throw new Error(
        `Her satırda en az ${numOutputs + 1} sayı olmalı (özellikler + hedef).`
      );
    }
    samples.push({
      features: nums.slice(0, -numOutputs),
      target: nums.slice(-numOutputs),
    });
  }
  return samples;
}

export function buildEmptyWeights(layerSizes) {
  const weights = [];
  const biases = [];
  for (let i = 1; i < layerSizes.length; i++) {
    const rows = layerSizes[i];
    const cols = layerSizes[i - 1];
    weights.push(
      Array.from({ length: rows }, () => Array.from({ length: cols }, () => 0))
    );
    biases.push(Array.from({ length: rows }, () => 0));
  }
  return { weights, biases };
}

/** Copy overlapping values when layer sizes change; new cells stay 0. */
export function resizeWeightsToArchitecture(prevWeights, prevBiases, newLayerSizes) {
  const { weights, biases } = buildEmptyWeights(newLayerSizes);
  if (!prevWeights?.length || !prevBiases?.length) {
    return { weights, biases };
  }
  const L = weights.length;
  for (let l = 0; l < L; l++) {
    const pw = prevWeights[l];
    const pb = prevBiases[l];
    if (!pw) continue;
    for (let i = 0; i < weights[l].length && i < pw.length; i++) {
      if (!pw[i]) continue;
      for (let j = 0; j < weights[l][i].length && j < pw[i].length; j++) {
        weights[l][i][j] = pw[i][j];
      }
    }
    if (pb) {
      for (let i = 0; i < biases[l].length && i < pb.length; i++) {
        biases[l][i] = pb[i];
      }
    }
  }
  return { weights, biases };
}

export function validateWeights(layerSizes, weights, biases) {
  if (!weights || !biases) throw new Error("Ağırlık veya bias eksik.");
  const L = layerSizes.length - 1;
  if (weights.length !== L || biases.length !== L) {
    throw new Error(`Beklenen ${L} ağırlık katmanı.`);
  }
  for (let l = 0; l < L; l++) {
    const rows = layerSizes[l + 1];
    const cols = layerSizes[l];
    const Wl = weights[l];
    if (!Wl || Wl.length !== rows) {
      throw new Error(`W${l + 1}: ${rows} satır bekleniyordu.`);
    }
    for (let i = 0; i < rows; i++) {
      if (!Wl[i] || Wl[i].length !== cols) {
        throw new Error(`W${l + 1}[${i}]: ${cols} sütun bekleniyordu.`);
      }
    }
    if (!biases[l] || biases[l].length !== rows) {
      throw new Error(`B${l + 1}: ${rows} bias bekleniyordu.`);
    }
  }
}

export function defaultAssignmentWeights() {
  const W1 = [
    [0.1, 0.2, -0.1, 0.3],
    [0.2, -0.1, 0.4, 0.1],
    [-0.2, 0.3, 0.1, 0.2],
  ];
  const W2 = [
    [0.3, -0.2, 0.1],
    [0.1, 0.4, -0.3],
  ];
  const W3 = [[0.2, 0.5]];
  const B1 = [0.1, 0.1, 0.1];
  const B2 = [0.1, 0.1];
  const B3 = [0.1];
  return {
    weights: [W1, W2, W3],
    biases: [B1, B2, B3],
  };
}

export function defaultAssignmentSamples() {
  return [
    { features: [148, 72, 35, 33.6], target: [1] },
    { features: [85, 66, 29, 26.6], target: [0] },
  ];
}
