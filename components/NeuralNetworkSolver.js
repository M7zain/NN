"use client";

import Image from "next/image";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import NetworkArchitectureVisual from "@/components/NetworkArchitectureVisual";
import WeightBiasEditor from "@/components/WeightBiasEditor";
import {
  activationFns,
  buildEmptyWeights,
  defaultAssignmentWeights,
  fmt,
  parseDatasetText,
  parseLayerSizes,
  resizeWeightsToArchitecture,
  runTraining,
  validateWeights,
} from "@/lib/nnSolver";

function KaTeXBlock({ math, display = true }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(math, {
        throwOnError: false,
        displayMode: display,
      });
    } catch {
      return math;
    }
  }, [math, display]);
  const Tag = display ? "div" : "span";
  return (
    <Tag
      className={
        display
          ? "my-2 max-w-full min-w-0 overflow-x-auto text-zinc-800 dark:text-zinc-100"
          : "inline-block max-w-full min-w-0 overflow-x-auto align-middle text-zinc-800 dark:text-zinc-100"
      }
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function MatrixTable({ rows }) {
  if (!rows?.length) return null;
  return (
    <div className="max-w-full min-w-0 overflow-x-auto rounded border border-zinc-200 bg-white p-2 font-mono text-xs dark:border-zinc-700 dark:bg-zinc-900">
      <table className="min-w-full border-collapse">
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td key={j} className="px-1 text-right tabular-nums">
                  {fmt(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function NeuralNetworkSolver() {
  const [datasetText, setDatasetText] = useState(
    `148,72,35,33.6,1\n85,66,29,26.6,0`
  );
  const [hiddenLayers, setHiddenLayers] = useState([3, 2]);
  const [numOutputs, setNumOutputs] = useState(1);
  const [learningRate, setLearningRate] = useState(0.5);
  const [epochs, setEpochs] = useState(2);
  const [activation, setActivation] = useState("sigmoid");
  const [wb, setWb] = useState(() => ({ ...defaultAssignmentWeights() }));
  const [showAdvancedJson, setShowAdvancedJson] = useState(false);
  const [jsonImport, setJsonImport] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const resultRef = useRef(null);

  const actInfo = useMemo(() => activationFns(activation), [activation]);

  useEffect(() => {
    if (!result) return;
    const id = requestAnimationFrame(() => {
      if (typeof window === "undefined") return;
      const narrow = window.matchMedia("(max-width: 1023px)").matches;
      if (!narrow) return;
      const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      resultRef.current?.scrollIntoView({
        behavior: reduce ? "auto" : "smooth",
        block: "start",
      });
    });
    return () => cancelAnimationFrame(id);
  }, [result]);

  const layerSizes = useMemo(() => {
    try {
      const samples = parseDatasetText(datasetText, numOutputs);
      return parseLayerSizes(
        samples[0].features.length,
        hiddenLayers.join(","),
        numOutputs
      );
    } catch {
      return null;
    }
  }, [datasetText, numOutputs, hiddenLayers]);

  const layerSizesKey = layerSizes?.join("-") ?? "";

  const numInputs = useMemo(() => {
    try {
      return parseDatasetText(datasetText, numOutputs)[0].features.length;
    } catch {
      return 0;
    }
  }, [datasetText, numOutputs]);

  useEffect(() => {
    if (!layerSizesKey) return;
    try {
      const samples = parseDatasetText(datasetText, numOutputs);
      const sizes = parseLayerSizes(
        samples[0].features.length,
        hiddenLayers.join(","),
        numOutputs
      );
      if (sizes.join("-") !== layerSizesKey) return;
      setWb((prev) =>
        resizeWeightsToArchitecture(prev.weights, prev.biases, sizes)
      );
    } catch {
      /* ignore */
    }
    // Yalnızca mimari (nöron sayıları) değişince yeniden boyutlandır
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layerSizesKey]);

  const jsonExport = useMemo(
    () => JSON.stringify(wb, null, 2),
    [wb]
  );

  const loadExample = useCallback(() => {
    setDatasetText(`148,72,35,33.6,1\n85,66,29,26.6,0`);
    setHiddenLayers([3, 2]);
    setNumOutputs(1);
    setLearningRate(0.5);
    setEpochs(2);
    setActivation("sigmoid");
    setWb({ ...defaultAssignmentWeights() });
    setError("");
    setResult(null);
  }, []);

  const generateEmptyMatrices = useCallback(() => {
    setError("");
    try {
      const samples = parseDatasetText(datasetText, numOutputs);
      const sizes = parseLayerSizes(
        samples[0].features.length,
        hiddenLayers.join(","),
        numOutputs
      );
      setWb(buildEmptyWeights(sizes));
    } catch (e) {
      setError(e.message || String(e));
    }
  }, [datasetText, hiddenLayers, numOutputs]);

  const applyJsonImport = useCallback(() => {
    setError("");
    try {
      if (!layerSizes) throw new Error("Önce geçerli bir veri seti girin.");
      const parsed = JSON.parse(jsonImport);
      validateWeights(layerSizes, parsed.weights, parsed.biases);
      setWb({ weights: parsed.weights, biases: parsed.biases });
    } catch (e) {
      setError(e.message || String(e));
    }
  }, [jsonImport, layerSizes]);

  const copyJson = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(jsonExport);
    } catch {
      setError("Panoya kopyalanamadı.");
    }
  }, [jsonExport]);

  const compute = useCallback(() => {
    setError("");
    setResult(null);
    try {
      if (!layerSizes) {
        throw new Error("Veri setini kontrol edin (özellik + hedef sütunları).");
      }
      const samples = parseDatasetText(datasetText, numOutputs);
      validateWeights(layerSizes, wb.weights, wb.biases);
      const out = runTraining({
        samples,
        weights: wb.weights,
        biases: wb.biases,
        activationName: activation,
        learningRate: Number(learningRate),
        epochs: Math.max(1, Math.min(500, parseInt(epochs, 10) || 2)),
      });
      setResult(out);
    } catch (e) {
      setError(e.message || String(e));
    }
  }, [
    datasetText,
    numOutputs,
    layerSizes,
    wb.weights,
    wb.biases,
    activation,
    learningRate,
    epochs,
  ]);

  return (
    <div className="mx-auto w-full min-w-0 max-w-5xl px-4 py-10 text-zinc-900 dark:text-zinc-100">
      <header className="mb-10 min-w-0 max-w-full overflow-hidden rounded-2xl border border-zinc-900 bg-zinc-950 text-zinc-100 shadow-sm">
        <div className="flex flex-col gap-4 px-5 py-5 sm:flex-row sm:items-center sm:gap-8 sm:px-6">
          <Image
            src="/karacode-logo-white.png"
            alt="Karacode Labs"
            width={260}
            height={52}
            className="h-10 w-auto max-w-[min(100%,280px)] shrink-0 object-contain object-left"
            priority
          />
          <div className="min-w-0 flex-1 border-t border-zinc-800/80 pt-4 sm:border-l sm:border-t-0 sm:pl-8 sm:pt-0">
            <h1 className="text-xl font-semibold tracking-tight text-white sm:text-2xl">
              Sinir ağı — adım adım çözümü
            </h1>
          </div>
        </div>
        <p className="border-t border-zinc-800/80 px-5 py-4 text-sm leading-relaxed text-zinc-400 sm:px-6">
          Aşağıdan mimariyi görsel olarak seçin; ağırlık ve bias değerlerini
          tablolardan girin. İsterseniz gelişmiş bölümden JSON da
          kullanabilirsiniz.
        </p>
      </header>

      <section className="mb-8 min-w-0 max-w-full rounded-xl border border-zinc-200 bg-zinc-50/80 p-5 dark:border-zinc-800 dark:bg-zinc-900/40">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-500">
          Kullanılan formüller
        </h2>
        <div className="mt-3 grid gap-4 md:grid-cols-2">
          <div>
            <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
              Lokal min–max (her sütun için)
            </p>
            <KaTeXBlock math="x_{\mathrm{norm}}=\dfrac{x-\min}{\max-\min}" />
          </div>
          <div>
            <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
              Çıkış hatası
            </p>
            <KaTeXBlock math="E = d - \mathrm{Out}" />
          </div>
          <div>
            <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
              Aktivasyon: {actInfo.label}
            </p>
            <KaTeXBlock math={actInfo.tex} />
            <KaTeXBlock math={actInfo.fpTex} />
          </div>
          <div>
            <p className="text-xs font-medium text-zinc-600 dark:text-zinc-400">
              Ağırlık ve bias güncelleme
            </p>
            <KaTeXBlock math="\Delta w_{ij} = \eta \cdot \delta_j \cdot x_i" />
            <KaTeXBlock math="\Delta b_j = \eta \cdot \delta_j" />
          </div>
        </div>
      </section>

      <div className="grid min-w-0 gap-8 lg:grid-cols-2 lg:gap-10">
        <div className="min-w-0 space-y-6">
          <label className="block">
            <span className="text-sm font-medium">
              Eğitim verisi (CSV: özellikler…, hedef)
            </span>
            <textarea
              className="mt-1 w-full min-h-[120px] rounded-lg border border-zinc-300 bg-white p-3 font-mono text-sm dark:border-zinc-600 dark:bg-zinc-950"
              value={datasetText}
              onChange={(e) => setDatasetText(e.target.value)}
              spellCheck={false}
            />
          </label>

          <NetworkArchitectureVisual
            numInputs={numInputs}
            hiddenLayers={hiddenLayers}
            setHiddenLayers={setHiddenLayers}
            numOutputs={numOutputs}
            setNumOutputs={setNumOutputs}
          />

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <label className="block">
              <span className="text-xs font-medium text-zinc-600">
                Öğrenme η
              </span>
              <input
                type="number"
                step="any"
                className="mt-1 w-full rounded border border-zinc-300 bg-white px-2 py-1.5 text-sm dark:border-zinc-600 dark:bg-zinc-950"
                value={learningRate}
                onChange={(e) => setLearningRate(e.target.value)}
              />
            </label>
            <label className="block">
              <span className="text-xs font-medium text-zinc-600">
                Epoch sayısı
              </span>
              <input
                type="number"
                min={1}
                max={500}
                className="mt-1 w-full rounded border border-zinc-300 bg-white px-2 py-1.5 text-sm dark:border-zinc-600 dark:bg-zinc-950"
                value={epochs}
                onChange={(e) => setEpochs(e.target.value)}
              />
            </label>
            <label className="block sm:col-span-1">
              <span className="text-xs font-medium text-zinc-600">
                Aktivasyon
              </span>
              <select
                className="mt-1 w-full rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-600 dark:bg-zinc-950"
                value={activation}
                onChange={(e) => setActivation(e.target.value)}
              >
                <option value="sigmoid">Sigmoid</option>
                <option value="tanh">Tanh</option>
                <option value="relu">ReLU</option>
              </select>
            </label>
          </div>

          {layerSizes ? (
            <div>
              <h3 className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">
                Ağırlık ve bias matrisleri
              </h3>
              <p className="mt-1 text-xs text-zinc-500">
                Mimari:{" "}
                <span className="font-mono tabular-nums">
                  {layerSizes.join(" → ")}
                </span>
              </p>
              <div className="mt-3">
                <WeightBiasEditor
                  layerSizes={layerSizes}
                  weights={wb.weights}
                  biases={wb.biases}
                  onChange={setWb}
                />
              </div>
            </div>
          ) : (
            <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
              Veri seti okunamıyor — satır başına yeterli sayıda virgülle ayrılmış
              değer girin.
            </p>
          )}

          <div>
            <button
              type="button"
              onClick={() => setShowAdvancedJson((s) => !s)}
              className="text-sm font-medium text-zinc-600 underline decoration-zinc-300 underline-offset-2 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-200"
            >
              {showAdvancedJson ? "Gelişmiş JSON ▲" : "Gelişmiş: JSON içe/dışa aktar ▼"}
            </button>
            {showAdvancedJson ? (
              <div className="mt-3 min-w-0 max-w-full space-y-3 rounded-lg border border-zinc-200 bg-zinc-50 p-3 dark:border-zinc-700 dark:bg-zinc-900/50">
                <div>
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-xs font-medium text-zinc-600">
                      Mevcut model (kopyala)
                    </span>
                    <button
                      type="button"
                      onClick={copyJson}
                      className="rounded border border-zinc-300 bg-white px-2 py-1 text-xs dark:border-zinc-600 dark:bg-zinc-800"
                    >
                      Panoya kopyala
                    </button>
                  </div>
                  <pre className="mt-2 max-h-40 max-w-full min-w-0 overflow-auto rounded border border-zinc-200 bg-white p-2 font-mono text-[10px] break-words whitespace-pre-wrap dark:border-zinc-600 dark:bg-zinc-950">
                    {jsonExport}
                  </pre>
                </div>
                <div>
                  <label className="text-xs font-medium text-zinc-600">
                    JSON yapıştır ve uygula
                  </label>
                  <textarea
                    className="mt-1 w-full min-h-[100px] rounded border border-zinc-300 bg-white p-2 font-mono text-xs dark:border-zinc-600 dark:bg-zinc-950"
                    value={jsonImport}
                    onChange={(e) => setJsonImport(e.target.value)}
                    placeholder='{"weights":[[...]],"biases":[[...]]}'
                    spellCheck={false}
                  />
                  <button
                    type="button"
                    onClick={applyJsonImport}
                    className="mt-2 rounded border border-zinc-300 bg-white px-3 py-1.5 text-xs font-medium dark:border-zinc-600 dark:bg-zinc-800"
                  >
                    JSON&apos;dan yükle
                  </button>
                </div>
              </div>
            ) : null}
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={compute}
              className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
            >
              Hesapla — adım adım çıktı
            </button>
            <button
              type="button"
              onClick={loadExample}
              className="rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm dark:border-zinc-600 dark:bg-zinc-900"
            >
              Ödev örneğini yükle
            </button>
            <button
              type="button"
              onClick={generateEmptyMatrices}
              className="rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm dark:border-zinc-600 dark:bg-zinc-900"
            >
              Sıfır matrisler
            </button>
          </div>

          {error ? (
            <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/50 dark:text-red-200">
              {error}
            </p>
          ) : null}
        </div>

        <aside className="h-fit min-w-0 rounded-xl border border-dashed border-zinc-300 p-4 text-sm text-zinc-600 dark:border-zinc-700 dark:text-zinc-400">
          <p className="font-medium text-zinc-800 dark:text-zinc-200">
            Nasıl doldurulur?
          </p>
          <ul className="mt-2 list-inside list-disc space-y-1">
            <li>
              Her satır: virgülle sayılar; son{" "}
              <strong>{numOutputs}</strong> sütun hedef (d).
            </li>
            <li>
              Gizli katman sayısını ve nöron sayılarını üstteki şema ile
              ayarlayın.
            </li>
            <li>
              Ağırlık tabloları otomatik boyutlanır; mimariyi küçültürseniz
              ortak hücreler korunur.
            </li>
          </ul>
        </aside>
      </div>

      {result ? (
        <output
          ref={resultRef}
          className="mt-12 block w-full min-w-0 max-w-full scroll-mt-4 print:mt-4"
          aria-live="polite"
        >
          <h2 className="text-lg font-semibold print:break-inside-avoid">
            Çözüm çıktısı
          </h2>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
            Mimari (nöron sayıları):{" "}
            <strong>{result.layerSizes.join(" - ")}</strong>. MSE (tüm çıkışlar
            ortalaması), epoch sonunda güncel ağırlıklarla tüm örnekler
            üzerinden — başlangıç{" "}
            <strong>{fmt(result.mseBeforeTraining)}</strong>
          </p>

          <section className="mt-6 min-w-0 max-w-full print:break-inside-avoid">
            <h3 className="text-base font-semibold">
              a) Lokal min–max normalizasyonu
            </h3>
            <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
              Her özellik sütunu için tablodaki min / max:
            </p>
            <div className="mt-2 min-w-0 max-w-full overflow-x-auto">
              <table className="min-w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-zinc-200 dark:border-zinc-700">
                    <th className="py-1 pr-4 text-left">Özellik</th>
                    <th className="py-1 pr-4">min</th>
                    <th className="py-1 pr-4">max</th>
                  </tr>
                </thead>
                <tbody>
                  {result.normalization.mins.map((mn, j) => (
                    <tr
                      key={j}
                      className="border-b border-zinc-100 dark:border-zinc-800"
                    >
                      <td className="py-1 pr-4">x{j + 1}</td>
                      <td className="tabular-nums">{fmt(mn)}</td>
                      <td className="tabular-nums">
                        {fmt(result.normalization.maxs[j])}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-sm font-medium">
              Normalize edilmiş örnekler:
            </p>
            <ul className="mt-1 max-w-full font-mono text-xs break-words">
              {result.normalizedTable.map((row, i) => (
                <li key={i} className="overflow-x-auto">
                  Örnek {i + 1}: x_norm = [
                  {row.features.map((v) => fmt(v)).join(", ")}], d = [
                  {row.target.map((v) => fmt(v)).join(", ")}]
                </li>
              ))}
            </ul>
          </section>

          {result.epochs.map((ep) => (
            <section
              key={ep.epoch}
              className="mt-10 min-w-0 max-w-full border-t border-zinc-200 pt-8 dark:border-zinc-800 print:break-inside-avoid"
            >
              <h3 className="text-lg font-semibold text-emerald-800 dark:text-emerald-300">
                Epoch {ep.epoch}
              </h3>
              <p className="mt-1 text-sm">
                Bu epoch sonunda MSE:{" "}
                <strong className="tabular-nums">{fmt(ep.mseAfterEpoch)}</strong>
              </p>

              {ep.samples.map((sm) => (
                <article
                  key={`${ep.epoch}-${sm.sampleIndex}`}
                  className="mt-8 min-w-0 max-w-full rounded-lg border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-700 dark:bg-zinc-950/50"
                >
                  <h4 className="break-words font-semibold">
                    Örnek {sm.sampleIndex} — ham giriş: [
                    {sm.rawFeatures.map((v) => fmt(v)).join(", ")}], hedef d = [
                    {sm.target.map((v) => fmt(v)).join(", ")}]
                  </h4>

                  <p className="mt-3 text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    b) İleri besleme (Net ve Out) — nöron nöron
                  </p>
                  <p className="max-w-full break-words text-xs text-zinc-500">
                    Normalize giriş (katman 0 / önceki katman çıkışları): [
                    {sm.xNorm.map((v) => fmt(v)).join(", ")}]
                  </p>
                  <KaTeXBlock math="\mathrm{Net}_i^{(\ell)}=\sum_j w_{ij}^{(\ell)}\,a_j^{(\ell-1)}+b_i^{(\ell)},\quad a_i^{(\ell)}=f\!\big(\mathrm{Net}_i^{(\ell)}\big)" />
                  {sm.forwardExplanation?.layers?.map((fl) => (
                    <div
                      key={fl.layer}
                      className="mt-4 min-w-0 max-w-full rounded-lg border border-sky-200/80 bg-sky-50/50 p-3 dark:border-sky-900/50 dark:bg-sky-950/30"
                    >
                      <p className="text-xs font-semibold text-sky-900 dark:text-sky-200">
                        Katman {fl.layer}
                      </p>
                      {fl.neurons.map((n) => (
                        <div
                          key={n.neuronIndex}
                          className="mt-3 min-w-0 max-w-full border-t border-sky-100 pt-3 first:mt-2 first:border-0 first:pt-0 dark:border-sky-900/40"
                        >
                          <KaTeXBlock
                            display={false}
                            math={`${n.netSymbol}=\\sum_j w_{${n.neuronIndex},j}^{(${n.layerIndex})}\\,a_j^{(${n.layerIndex - 1})}+b_${n.neuronIndex}^{(${n.layerIndex})}`}
                          />
                          <p className="mt-1 max-w-full break-words font-mono text-[11px] leading-relaxed text-zinc-800 dark:text-zinc-200">
                            {n.netEquationPlain}
                          </p>
                          <div className="mt-2 min-w-0 max-w-full overflow-x-auto">
                            <table className="min-w-full border-collapse text-[10px]">
                              <thead>
                                <tr className="border-b border-zinc-200 dark:border-zinc-700">
                                  <th className="py-1 pr-2 text-left font-normal text-zinc-500">
                                    Kaynak
                                  </th>
                                  <th className="py-1 pr-2 font-normal text-zinc-500">
                                    a
                                  </th>
                                  <th className="py-1 pr-2 font-normal text-zinc-500">
                                    w
                                  </th>
                                  <th className="py-1 font-normal text-zinc-500">
                                    a·w
                                  </th>
                                </tr>
                              </thead>
                              <tbody>
                                {n.terms.map((t, ti) => (
                                  <tr
                                    key={ti}
                                    className="border-b border-zinc-100 dark:border-zinc-800"
                                  >
                                    <td className="py-0.5 pr-2 font-mono">
                                      {t.prevPlain}
                                    </td>
                                    <td className="tabular-nums">{fmt(t.a)}</td>
                                    <td className="tabular-nums">{fmt(t.w)}</td>
                                    <td className="tabular-nums">{fmt(t.product)}</td>
                                  </tr>
                                ))}
                                <tr>
                                  <td
                                    className="py-0.5 pr-2 font-medium text-zinc-600"
                                    colSpan={3}
                                  >
                                    Bias b
                                  </td>
                                  <td className="tabular-nums font-medium">
                                    {fmt(n.bias)}
                                  </td>
                                </tr>
                              </tbody>
                            </table>
                          </div>
                          <p className="mt-2 max-w-full break-words font-mono text-[11px] text-zinc-700 dark:text-zinc-300">
                            Toplam (Σ a·w) + b = {fmt(n.sumWeightedInputs)} +{" "}
                            {fmt(n.bias)} ={" "}
                            <strong>{fmt(n.net)}</strong> ⇒ Out = f(Net) ={" "}
                            <strong>{fmt(n.out)}</strong>
                          </p>
                          {n.activationTex ? (
                            <div className="mt-3 min-w-0 max-w-full rounded border border-sky-100 bg-white/80 p-2 dark:border-sky-900/50 dark:bg-zinc-900/40">
                              <p className="text-[10px] font-medium text-zinc-500">
                                Aktivasyon f (seçilen fonksiyon)
                              </p>
                              <KaTeXBlock display={false} math={n.activationTex} />
                            </div>
                          ) : null}
                          {n.derivativeKatex?.texGeneral ? (
                            <div className="mt-2 min-w-0 max-w-full rounded border border-amber-100 bg-amber-50/60 p-2 dark:border-amber-900/40 dark:bg-amber-950/20">
                              <p className="text-[10px] font-medium text-amber-900 dark:text-amber-200">
                                f&apos; — genel formül
                              </p>
                              <KaTeXBlock
                                display={false}
                                math={n.derivativeKatex.texGeneral}
                              />
                              <p className="mt-1 text-[10px] font-medium text-amber-900 dark:text-amber-200">
                                f&apos; — bu nöron için sayılarla
                              </p>
                              {n.derivativeKatex.texNumeric ? (
                                <KaTeXBlock
                                  display={false}
                                  math={n.derivativeKatex.texNumeric}
                                />
                              ) : null}
                            </div>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  ))}
                  <p className="mt-3 text-xs font-medium text-zinc-600">
                    Özet vektörler (aynı katman)
                  </p>
                  {sm.nets.map((layer) => (
                    <div key={layer.layer} className="mt-1 min-w-0 max-w-full">
                      <p className="max-w-full break-words font-mono text-[11px] text-zinc-500">
                        Katman {layer.layer} Net: [
                        {layer.values.map((v) => fmt(v)).join(", ")}] · Out: [
                        {sm.activations
                          .find((a) => a.layer === layer.layer)
                          ?.values.map((v) => fmt(v))
                          .join(", ") ?? ""}
                        ]
                      </p>
                    </div>
                  ))}
                  <p className="mt-2 max-w-full break-words text-xs text-zinc-500">
                    Çıkış vektörü Out: [
                    {sm.output.map((v) => fmt(v)).join(", ")}]
                  </p>

                  <p className="mt-4 text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    c) Hata E = d − Out
                  </p>
                  <p className="max-w-full break-words font-mono text-xs">
                    E = [{sm.E.map((v) => fmt(v)).join(", ")}]
                  </p>

                  <p className="mt-4 text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    d) Geri yayılım — δ terimleri ve zincir kuralı
                  </p>
                  <KaTeXBlock math="\delta_i^{(L)} = E_i \cdot f'\!\big(\mathrm{Net}_i^{(L)}\big),\qquad \delta_j^{(\ell)} = \Big(\sum_k \delta_k^{(\ell+1)} W_{k,j}^{(\ell+1)}\Big)\cdot f'\!\big(\mathrm{Net}_j^{(\ell)}\big)" />
                  {sm.backwardExplanation?.blocks?.map((blk) => (
                    <div
                      key={blk.layer}
                      className="mt-4 min-w-0 max-w-full rounded-lg border border-violet-200/80 bg-violet-50/50 p-3 dark:border-violet-900/50 dark:bg-violet-950/25"
                    >
                      <p className="text-xs font-semibold text-violet-900 dark:text-violet-200">
                        {blk.title}
                      </p>
                      {blk.neurons.map((neu) => (
                        <div
                          key={`${blk.layer}-${neu.neuronIndex}`}
                          className="mt-3 min-w-0 max-w-full border-t border-violet-100 pt-3 first:mt-2 first:border-0 first:pt-0 dark:border-violet-900/40"
                        >
                          <p className="text-[11px] font-medium text-zinc-700 dark:text-zinc-300">
                            Nöron {neu.neuronIndex} — δ = {fmt(neu.delta)}
                          </p>
                          {neu.backTerms?.length ? (
                            <ul className="mt-1 max-w-full space-y-0.5 break-words font-mono text-[10px] text-zinc-600 dark:text-zinc-400">
                              {neu.backTerms.map((bt, bi) => (
                                <li key={bi}>{bt.plain}</li>
                              ))}
                            </ul>
                          ) : null}
                          {neu.steps.map((st, si) => (
                            <div key={si} className="mt-2 min-w-0 max-w-full">
                              <p className="text-[10px] font-medium uppercase tracking-wide text-zinc-500">
                                {st.title}
                              </p>
                              {st.tex ? (
                                <KaTeXBlock display={false} math={st.tex} />
                              ) : null}
                              {st.title === "Aktivasyon türevi" &&
                              st.activationTex ? (
                                <div className="mt-2 min-w-0 max-w-full rounded border border-zinc-200 bg-white/90 p-2 dark:border-zinc-600 dark:bg-zinc-900/50">
                                  <p className="text-[10px] text-zinc-500">
                                    Önce f (aktivasyon) — seçilen fonksiyon
                                  </p>
                                  <KaTeXBlock
                                    display={false}
                                    math={st.activationTex}
                                  />
                                </div>
                              ) : null}
                              {st.texGeneral ? (
                                <div className="mt-2 min-w-0 max-w-full rounded border border-amber-100 bg-amber-50/70 p-2 dark:border-amber-900/40 dark:bg-amber-950/25">
                                  <p className="text-[10px] font-medium text-amber-900 dark:text-amber-200">
                                    f&apos; — genel türev formülü
                                  </p>
                                  <KaTeXBlock
                                    display={false}
                                    math={st.texGeneral}
                                  />
                                  <p className="mt-2 text-[10px] font-medium text-amber-900 dark:text-amber-200">
                                    f&apos; — zincir kuralında kullanılan sayısal
                                    değer
                                  </p>
                                  {st.texNumeric ? (
                                    <KaTeXBlock
                                      display={false}
                                      math={st.texNumeric}
                                    />
                                  ) : null}
                                </div>
                              ) : null}
                              <p className="mt-1 max-w-full break-words font-mono text-[11px] leading-relaxed text-zinc-800 dark:text-zinc-200">
                                {st.plain}
                              </p>
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  ))}
                  <p className="mt-3 text-xs font-medium text-zinc-600">
                    δ vektör özeti
                  </p>
                  {sm.deltas.map((d) => (
                    <p
                      key={d.layer}
                      className="max-w-full break-words font-mono text-[11px] text-zinc-500"
                    >
                      Katman {d.layer}: δ = [
                      {d.values.map((v) => fmt(v)).join(", ")}]
                    </p>
                  ))}

                  <p className="mt-4 text-sm font-medium text-zinc-700 dark:text-zinc-300">
                    e) Ağırlık ve bias güncellemeleri (η ={" "}
                    {fmt(result.learningRate ?? Number(learningRate))})
                  </p>
                  {sm.deltaW.map((dw) => (
                    <div key={dw.fromLayer} className="mt-2 min-w-0 max-w-full">
                      <p className="text-xs text-zinc-600">
                        ΔW{dw.fromLayer + 1} ({dw.matrix.length}×
                        {dw.matrix[0]?.length ?? 0}):
                      </p>
                      <MatrixTable rows={dw.matrix} />
                    </div>
                  ))}
                  {sm.deltaB.map((db) => (
                    <div key={db.layer} className="mt-2 min-w-0 max-w-full">
                      <p className="text-xs text-zinc-600">ΔB{db.layer}:</p>
                      <p className="max-w-full break-words font-mono text-xs">
                        [{db.values.map((v) => fmt(v)).join(", ")}]
                      </p>
                    </div>
                  ))}

                  <p className="mt-4 text-xs text-zinc-500">
                    Bu örnekten sonra güncel W ve B bir sonraki örnek / epoch için
                    kullanılır.
                  </p>
                </article>
              ))}
            </section>
          ))}

          <section className="mt-10 min-w-0 max-w-full print:break-inside-avoid">
            <h3 className="text-base font-semibold">f) Epoch’lara göre MSE</h3>
            <ul className="mt-2 space-y-1 text-sm">
              <li>
                Başlangıç (eğitim öncesi): MSE ={" "}
                <strong>{fmt(result.mseBeforeTraining)}</strong>
              </li>
              {result.epochs.map((ep) => (
                <li key={ep.epoch}>
                  Epoch {ep.epoch} sonu: MSE ={" "}
                  <strong>{fmt(ep.mseAfterEpoch)}</strong>
                </li>
              ))}
            </ul>
          </section>

          <section className="mt-8 min-w-0 max-w-full print:break-inside-avoid">
            <h3 className="text-base font-semibold">
              Son ağırlıklar ve bias değerleri
            </h3>
            {result.finalWeights.map((W, li) => (
              <div key={li} className="mt-3">
                <p className="text-xs font-medium text-zinc-600">
                  W{li + 1}
                </p>
                <MatrixTable rows={W} />
                <p className="mt-1 text-xs font-medium text-zinc-600">
                  B{li + 1}: [
                  {result.finalBiases[li].map((v) => fmt(v)).join(", ")}]
                </p>
              </div>
            ))}
          </section>
        </output>
      ) : null}

      <footer className="mt-16 min-w-0 max-w-full border-t border-zinc-200 pt-8 text-center text-xs text-zinc-500 dark:border-zinc-800">
        <a
          href="https://karacode.com.tr"
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium text-zinc-700 underline decoration-zinc-300 underline-offset-2 hover:text-zinc-900 dark:text-zinc-300 dark:decoration-zinc-600 dark:hover:text-zinc-100"
        >
          karacode.com.tr
        </a>
        <p className="mt-3 text-zinc-500">
          Yazdırmak için tarayıcıdan Yazdır (Ctrl+P) kullanabilirsiniz.
        </p>
      </footer>
    </div>
  );
}
