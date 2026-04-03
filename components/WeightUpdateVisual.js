"use client";

import {
  useCallback,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

/**
 * Flat list of per-parameter updates (same order as nested loops in applySampleUpdate).
 */
export function buildWeightUpdateSteps(sm) {
  const steps = [];
  for (const dw of sm.deltaW) {
    const li = dw.fromLayer;
    dw.matrix.forEach((row, i) => {
      row.forEach((val, j) => {
        steps.push({
          key: `w-${li}-${i}-${j}`,
          type: "weight",
          matrixIndex: li,
          destNeuron: i,
          srcNeuron: j,
          value: val,
          label: `ΔW${li + 1}[${i + 1},${j + 1}]`,
        });
      });
    });
  }
  for (const db of sm.deltaB) {
    db.values.forEach((val, i) => {
      steps.push({
        key: `b-${db.layer}-${i}`,
        type: "bias",
        layer: db.layer,
        neuronIndex: i,
        value: val,
        label: `ΔB${db.layer}[${i + 1}]`,
      });
    });
  }
  return steps;
}

function layerTitle(layerIndex, totalLayers) {
  if (layerIndex === 0) return "Giriş";
  if (layerIndex === totalLayers - 1) return "Çıkış";
  return `Gizli ${layerIndex}`;
}

function ArrowBetween() {
  return (
    <div
      className="flex h-full min-h-[3rem] shrink-0 items-center self-stretch px-0.5 text-zinc-400 dark:text-zinc-500"
      aria-hidden
    >
      <svg
        width="20"
        height="14"
        viewBox="0 0 24 14"
        className="overflow-visible sm:h-4 sm:w-6"
      >
        <path
          d="M0 7h18"
          stroke="currentColor"
          strokeWidth="1.5"
          fill="none"
          strokeLinecap="round"
        />
        <path
          d="M18 2l5 5-5 5"
          stroke="currentColor"
          strokeWidth="1.5"
          fill="none"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

/**
 * @param {object} props
 * @param {number[]} props.layerSizes
 * @param {ReturnType<typeof buildWeightUpdateSteps>} props.steps
 * @param {(n: number) => string} props.fmt
 * @param {number} props.eta
 * @param {object} props.sm - sample step (for δ, a captions)
 */
export default function WeightUpdateVisual({
  layerSizes,
  steps,
  fmt,
  eta,
  sm,
}) {
  const svgMarkerId = useId().replace(/:/g, "");
  const [idx, setIdx] = useState(0);
  const containerRef = useRef(null);
  const fromRef = useRef(null);
  const toRef = useRef(null);
  const biasRef = useRef(null);
  const [line, setLine] = useState(null);

  const step = steps[idx] ?? null;
  const total = steps.length;

  const updateLine = useCallback(() => {
    if (!step || step.type !== "weight") {
      setLine(null);
      return;
    }
    const c = containerRef.current;
    const a = fromRef.current;
    const b = toRef.current;
    if (!c || !a || !b) {
      setLine(null);
      return;
    }
    const cr = c.getBoundingClientRect();
    const ar = a.getBoundingClientRect();
    const br = b.getBoundingClientRect();
    setLine({
      x1: ar.left + ar.width / 2 - cr.left,
      y1: ar.top + ar.height / 2 - cr.top,
      x2: br.left + br.width / 2 - cr.left,
      y2: br.top + br.height / 2 - cr.top,
    });
  }, [step]);

  useLayoutEffect(() => {
    updateLine();
  }, [updateLine, idx, layerSizes]);

  useLayoutEffect(() => {
    const ro =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(() => updateLine())
        : null;
    if (ro && containerRef.current) ro.observe(containerRef.current);
    const onResize = () => updateLine();
    window.addEventListener("resize", onResize);
    return () => {
      ro?.disconnect();
      window.removeEventListener("resize", onResize);
    };
  }, [updateLine]);

  const caption = useMemo(() => {
    if (!step || !sm) return null;
    if (step.type === "weight") {
      const l = step.matrixIndex + 1;
      const d = sm.deltas.find((x) => x.layer === l);
      const prevA = sm.activations.find(
        (x) => x.layer === step.matrixIndex
      );
      const di = d?.values[step.destNeuron];
      const aj = prevA?.values[step.srcNeuron];
      const deltaStr = di !== undefined ? fmt(di) : "—";
      const aStr = aj !== undefined ? fmt(aj) : "—";
      return `Δw = η·δ_${step.destNeuron + 1}^{(${l})}·a_${step.srcNeuron + 1}^{(${step.matrixIndex})} = ${fmt(eta)}×${deltaStr}×${aStr} = ${fmt(step.value)}`;
    }
    const d = sm.deltas.find((x) => x.layer === step.layer);
    const di = d?.values[step.neuronIndex];
    const deltaStr = di !== undefined ? fmt(di) : "—";
    return `Δb = η·δ_${step.neuronIndex + 1}^{(${step.layer})} = ${fmt(eta)}×${deltaStr} = ${fmt(step.value)}`;
  }, [step, sm, fmt, eta]);

  if (!layerSizes?.length || total === 0) return null;

  const totalLayers = layerSizes.length;

  return (
    <div className="mt-3 min-w-0 max-w-full rounded-xl border border-teal-200/90 bg-teal-50/40 p-3 dark:border-teal-900/50 dark:bg-teal-950/20">
      <p className="text-[11px] font-semibold text-teal-900 dark:text-teal-200">
        Güncelleme — hangi nöronlar?
      </p>
      <p className="mt-1 text-[10px] text-zinc-600 dark:text-zinc-400">
        Ağırlık için: önceki katmandaki nöron (a) ile sonraki katmandaki nöron
        (δ) arasındaki bağlantı güncellenir. Bias için: ilgili katmandaki tek
        nöron vurgulanır.
      </p>

      <div className="mt-2 flex flex-wrap items-center gap-2">
        <label className="sr-only" htmlFor="wu-step">
          Güncelleme adımı
        </label>
        <select
          id="wu-step"
          className="max-w-full min-w-0 flex-1 rounded-lg border border-teal-200 bg-white px-2 py-1.5 text-xs font-medium text-zinc-800 dark:border-teal-800 dark:bg-zinc-900 dark:text-zinc-100 sm:max-w-md"
          value={idx}
          onChange={(e) => setIdx(Number(e.target.value))}
        >
          {steps.map((s, i) => (
            <option key={s.key} value={i}>
              {s.label} = {fmt(s.value)}
            </option>
          ))}
        </select>
        <span className="text-[10px] tabular-nums text-zinc-500">
          {idx + 1} / {total}
        </span>
      </div>

      <div
        ref={containerRef}
        className="relative mt-3 min-w-0 overflow-x-auto rounded-lg border border-teal-100 bg-white/80 p-2 dark:border-teal-900/40 dark:bg-zinc-950/40"
      >
        {line ? (
          <svg
            className="pointer-events-none absolute inset-0 z-[1] h-full w-full overflow-visible"
            aria-hidden
          >
            <defs>
              <marker
                id={svgMarkerId}
                markerWidth="8"
                markerHeight="8"
                refX="6"
                refY="4"
                orient="auto"
              >
                <path d="M0,0 L8,4 L0,8 z" fill="rgb(13 148 136)" />
              </marker>
            </defs>
            <line
              x1={line.x1}
              y1={line.y1}
              x2={line.x2}
              y2={line.y2}
              stroke="rgb(13 148 136)"
              strokeWidth="2.5"
              strokeLinecap="round"
              markerEnd={`url(#${svgMarkerId})`}
              opacity="0.85"
            />
          </svg>
        ) : null}

        <div className="relative z-[2] flex w-max min-w-full flex-nowrap items-stretch justify-center gap-0 sm:w-full sm:justify-start">
          {layerSizes.map((count, colIdx) => (
            <div
              key={colIdx}
              className="flex shrink-0 items-stretch sm:min-w-0 sm:shrink sm:flex-1 sm:basis-[72px]"
            >
              {colIdx > 0 ? <ArrowBetween /> : null}
              <div className="flex w-[100px] flex-col rounded-md border border-zinc-200 bg-zinc-50/90 p-2 dark:border-zinc-600 dark:bg-zinc-900/80 sm:min-w-0 sm:max-w-none sm:flex-1 sm:basis-0">
                <p className="text-center text-[10px] font-semibold text-zinc-700 dark:text-zinc-200">
                  {layerTitle(colIdx, totalLayers)}
                </p>
                <p className="text-center text-[9px] text-zinc-500">
                  katman {colIdx}
                </p>
                <div className="mt-2 flex max-h-48 flex-col gap-1 overflow-y-auto">
                  {Array.from({ length: count }).map((_, ni) => {
                    const isWeight = step?.type === "weight";
                    const isBias = step?.type === "bias";
                    let ring = "border-zinc-300 dark:border-zinc-600";
                    let cellRef = undefined;
                    if (isWeight && step) {
                      if (colIdx === step.matrixIndex && ni === step.srcNeuron) {
                        ring =
                          "border-sky-500 ring-2 ring-sky-400 ring-offset-1 dark:ring-sky-500";
                        cellRef = fromRef;
                      } else if (
                        colIdx === step.matrixIndex + 1 &&
                        ni === step.destNeuron
                      ) {
                        ring =
                          "border-amber-500 ring-2 ring-amber-400 ring-offset-1 dark:ring-amber-500";
                        cellRef = toRef;
                      }
                    } else if (
                      isBias &&
                      step &&
                      colIdx === step.layer &&
                      ni === step.neuronIndex
                    ) {
                      ring =
                        "border-violet-500 ring-2 ring-violet-400 ring-offset-1 dark:ring-violet-500";
                      cellRef = biasRef;
                    }
                    return (
                      <div
                        key={ni}
                        ref={cellRef}
                        className={`flex h-7 w-7 shrink-0 items-center justify-center self-center rounded-full border bg-white text-[10px] font-bold tabular-nums text-zinc-800 dark:bg-zinc-950 dark:text-zinc-100 ${ring}`}
                        title={`Nöron ${ni + 1}`}
                      >
                        {ni + 1}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {step?.type === "weight" ? (
        <p className="mt-2 text-[10px] text-zinc-600 dark:text-zinc-400">
          <span className="font-medium text-sky-700 dark:text-sky-300">
            Mavi halka
          </span>
          : önceki katman (a kaynağı) ·{" "}
          <span className="font-medium text-amber-700 dark:text-amber-300">
            Turuncu halka
          </span>
          : sonraki katman (δ hedefi)
        </p>
      ) : step?.type === "bias" ? (
        <p className="mt-2 text-[10px] text-zinc-600 dark:text-zinc-400">
          <span className="font-medium text-violet-700 dark:text-violet-300">
            Mor halka
          </span>
          : bias güncellenen nöron (katman {step.layer}, nöron{" "}
          {step.neuronIndex + 1})
        </p>
      ) : null}

      {caption ? (
        <p className="mt-2 max-w-full break-words font-mono text-[10px] leading-relaxed text-zinc-700 dark:text-zinc-300">
          {caption}
        </p>
      ) : null}
    </div>
  );
}
