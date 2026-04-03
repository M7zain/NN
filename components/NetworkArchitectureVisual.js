"use client";

function NeuronStack({ count, accent }) {
  const show = Math.min(count, 10);
  const dot =
    "h-2.5 w-2.5 shrink-0 rounded-full border border-current opacity-90";
  return (
    <div
      className={`flex flex-col items-center gap-1 ${accent}`}
      aria-hidden
    >
      <div className="flex max-h-24 flex-col flex-wrap content-center gap-1.5 overflow-hidden py-1">
        {Array.from({ length: show }).map((_, i) => (
          <div key={i} className={dot} />
        ))}
      </div>
      {count > show ? (
        <span className="text-[10px] font-medium text-zinc-500">
          +{count - show}
        </span>
      ) : null}
    </div>
  );
}

function Arrow() {
  return (
    <div
      className="flex shrink-0 items-center self-center px-1 text-zinc-400 dark:text-zinc-500"
      aria-hidden
    >
      <svg width="28" height="16" viewBox="0 0 28 16" className="overflow-visible">
        <path
          d="M0 8h22"
          stroke="currentColor"
          strokeWidth="1.5"
          fill="none"
          strokeLinecap="round"
        />
        <path
          d="M22 3l5 5-5 5"
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
 * @param {number} props.numInputs - from dataset (feature count)
 * @param {number[]} props.hiddenLayers
 * @param {(v: number[] | ((prev: number[]) => number[])) => void} props.setHiddenLayers
 * @param {number} props.numOutputs
 * @param {(n: number) => void} props.setNumOutputs
 */
export default function NetworkArchitectureVisual({
  numInputs,
  hiddenLayers,
  setHiddenLayers,
  numOutputs,
  setNumOutputs,
}) {
  const addHidden = () => {
    setHiddenLayers((prev) => [...prev, 2]);
  };

  const removeHidden = (index) => {
    setHiddenLayers((prev) => prev.filter((_, i) => i !== index));
  };

  const setHiddenAt = (index, value) => {
    const n = Math.max(1, Math.min(64, parseInt(value, 10) || 1));
    setHiddenLayers((prev) => {
      const next = [...prev];
      next[index] = n;
      return next;
    });
  };

  const columns = [];

  columns.push({
    key: "in",
    title: "Giriş",
    subtitle: `${numInputs} özellik`,
    count: numInputs,
    accent: "text-sky-600 dark:text-sky-400",
    controls: null,
  });

  hiddenLayers.forEach((n, idx) => {
    columns.push({
      key: `h-${idx}`,
      title: `Gizli ${idx + 1}`,
      subtitle: `${n} nöron`,
      count: n,
      accent: "text-violet-600 dark:text-violet-400",
      controls: (
        <div className="mt-2 flex flex-wrap items-center gap-1">
          <label className="sr-only" htmlFor={`hidden-${idx}`}>
            Gizli katman {idx + 1} nöron sayısı
          </label>
          <input
            id={`hidden-${idx}`}
            type="number"
            min={1}
            max={64}
            className="w-14 rounded border border-zinc-300 bg-white px-1.5 py-0.5 text-center text-xs tabular-nums dark:border-zinc-600 dark:bg-zinc-950"
            value={n}
            onChange={(e) => setHiddenAt(idx, e.target.value)}
          />
          <button
            type="button"
            onClick={() => removeHidden(idx)}
            className="rounded border border-red-200 px-1.5 py-0.5 text-[10px] text-red-700 hover:bg-red-50 dark:border-red-900 dark:text-red-400 dark:hover:bg-red-950/40"
            title="Bu gizli katmanı kaldır"
          >
            Sil
          </button>
        </div>
      ),
    });
  });

  columns.push({
    key: "out",
    title: "Çıkış",
    subtitle: `${numOutputs} nöron`,
    count: numOutputs,
    accent: "text-emerald-600 dark:text-emerald-400",
    controls: (
      <div className="mt-2">
        <label className="sr-only" htmlFor="num-out">
          Çıkış nöron sayısı
        </label>
        <input
          id="num-out"
          type="number"
          min={1}
          max={32}
          className="w-14 rounded border border-zinc-300 bg-white px-1.5 py-0.5 text-center text-xs tabular-nums dark:border-zinc-600 dark:bg-zinc-950"
          value={numOutputs}
          onChange={(e) =>
            setNumOutputs(Math.max(1, Math.min(32, parseInt(e.target.value, 10) || 1)))
          }
        />
      </div>
    ),
  });

  const archStr = [
    numInputs,
    ...hiddenLayers,
    numOutputs,
  ].join(" — ");

  return (
    <div className="rounded-xl border border-zinc-200 bg-gradient-to-b from-white to-zinc-50/80 p-4 dark:border-zinc-700 dark:from-zinc-950 dark:to-zinc-900/80">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">
            Ağ mimarisi
          </h3>
          <p className="mt-0.5 font-mono text-xs text-zinc-500 tabular-nums">
            {archStr}
          </p>
        </div>
        <button
          type="button"
          onClick={addHidden}
          className="rounded-lg border border-dashed border-zinc-300 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 hover:bg-zinc-50 dark:border-zinc-600 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
        >
          + Gizli katman ekle
        </button>
      </div>

      <div
        className="mt-4 flex flex-wrap items-stretch justify-center gap-0 sm:justify-start"
        role="img"
        aria-label={`Sinir ağı: ${archStr}`}
      >
        {columns.map((col, i) => (
          <div key={col.key} className="flex items-stretch">
            {i > 0 ? <Arrow /> : null}
            <div className="flex min-w-[104px] max-w-[140px] flex-1 flex-col rounded-lg border border-zinc-200 bg-white p-3 shadow-sm dark:border-zinc-600 dark:bg-zinc-900/90">
              <p className={`text-xs font-semibold ${col.accent}`}>{col.title}</p>
              <p className="text-[10px] text-zinc-500 dark:text-zinc-400">
                {col.subtitle}
              </p>
              <div className="mt-2 flex flex-1 flex-col items-center justify-center gap-2">
                <NeuronStack count={col.count} accent={col.accent} />
                {col.controls}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
