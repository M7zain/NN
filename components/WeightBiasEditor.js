"use client";

import { fmt } from "@/lib/nnSolver";

/**
 * @param {object} props
 * @param {number[]} props.layerSizes
 * @param {number[][][]} props.weights
 * @param {number[][]} props.biases
 * @param {(next: { weights: number[][][]; biases: number[][] }) => void} props.onChange
 */
export default function WeightBiasEditor({
  layerSizes,
  weights,
  biases,
  onChange,
}) {
  const L = layerSizes.length - 1;
  if (L < 1) {
    return (
      <p className="text-sm text-zinc-500">
        Önce veri setinden en az bir giriş özelliği ve bir çıkış tanımlayın.
      </p>
    );
  }

  const setWeight = (l, i, j, val) => {
    const v = typeof val === "number" && Number.isFinite(val) ? val : 0;
    const nextW = weights.map((M, li) => {
      if (li !== l) return M.map((row) => [...row]);
      return M.map((row, ri) => {
        if (ri !== i) return [...row];
        return row.map((c, ci) => (ci === j ? v : c));
      });
    });
    onChange({ weights: nextW, biases: biases.map((b) => [...b]) });
  };

  const setBias = (l, i, val) => {
    const v = typeof val === "number" && Number.isFinite(val) ? val : 0;
    const nextB = biases.map((b, li) =>
      li === l ? b.map((x, bi) => (bi === i ? v : x)) : [...b]
    );
    onChange({
      weights: weights.map((M) => M.map((r) => [...r])),
      biases: nextB,
    });
  };

  return (
    <div className="space-y-8">
      {Array.from({ length: L }).map((_, l) => {
        const rows = layerSizes[l + 1];
        const cols = layerSizes[l];
        const Wl = weights[l];
        const Bl = biases[l];
        const prevLabels =
          l === 0
            ? Array.from({ length: cols }, (_, j) => `x${j + 1}`)
            : Array.from({ length: cols }, (_, j) => `h${l}_${j + 1}`);

        return (
          <div
            key={l}
            className="overflow-x-auto rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-950/50"
          >
            <p className="text-sm font-semibold text-zinc-800 dark:text-zinc-100">
              Katman {l + 1}:{" "}
              <span className="font-mono font-normal text-zinc-600 dark:text-zinc-400">
                W<sub>{l + 1}</sub> ({rows}×{cols}) · B<sub>{l + 1}</sub> ({rows}
                ×1)
              </span>
            </p>
            <p className="mt-1 text-xs text-zinc-500">
              Satırlar bu katmanın nöronları; sütunlar bir önceki katmanın
              çıkışları (w<sub>ij</sub>: hedef nöron i, kaynak j).
            </p>

            <div className="mt-3 overflow-x-auto">
              <table className="border-collapse text-xs">
                <thead>
                  <tr>
                    <th className="border border-zinc-200 bg-zinc-50 px-1 py-1 dark:border-zinc-600 dark:bg-zinc-800/80" />
                    {prevLabels.map((lab, j) => (
                      <th
                        key={j}
                        className="min-w-[4.25rem] border border-zinc-200 bg-zinc-50 px-1 py-1 font-mono text-[10px] font-normal text-zinc-600 dark:border-zinc-600 dark:bg-zinc-800/80 dark:text-zinc-400"
                      >
                        {lab}
                      </th>
                    ))}
                    <th className="min-w-[4.25rem] border border-zinc-200 bg-amber-50 px-2 py-1 text-[10px] font-medium text-amber-900 dark:border-zinc-600 dark:bg-amber-950/40 dark:text-amber-200">
                      b
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from({ length: rows }).map((__, i) => (
                    <tr key={i}>
                      <td className="border border-zinc-200 bg-zinc-50 px-2 py-0.5 font-mono text-[10px] text-zinc-600 dark:border-zinc-600 dark:bg-zinc-800/80">
                        n{l + 1},{i + 1}
                      </td>
                      {Array.from({ length: cols }).map((__, j) => (
                        <td
                          key={j}
                          className="border border-zinc-100 p-0 dark:border-zinc-800"
                        >
                          <input
                            type="number"
                            step="any"
                            className="w-full min-w-[3.75rem] border-0 bg-transparent px-1 py-1 text-center font-mono text-[11px] tabular-nums text-zinc-900 focus:bg-sky-50 focus:ring-1 focus:ring-sky-400 dark:text-zinc-100 dark:focus:bg-sky-950/50"
                            value={Wl?.[i]?.[j] ?? 0}
                            onChange={(e) => {
                              const v = parseFloat(e.target.value);
                              setWeight(
                                l,
                                i,
                                j,
                                Number.isFinite(v) ? v : 0
                              );
                            }}
                            aria-label={`W${l + 1} satır ${i + 1} sütun ${j + 1}`}
                          />
                        </td>
                      ))}
                      <td className="border border-amber-100 bg-amber-50/50 p-0 dark:border-amber-900/40 dark:bg-amber-950/20">
                        <input
                          type="number"
                          step="any"
                          className="w-full min-w-[3.75rem] border-0 bg-transparent px-1 py-1 text-center font-mono text-[11px] tabular-nums text-amber-950 focus:bg-amber-100 focus:ring-1 focus:ring-amber-400 dark:text-amber-100 dark:focus:bg-amber-950/50"
                          value={Bl?.[i] ?? 0}
                          onChange={(e) => {
                            const v = parseFloat(e.target.value);
                            setBias(l, i, Number.isFinite(v) ? v : 0);
                          }}
                          aria-label={`B${l + 1} nöron ${i + 1}`}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <p className="mt-2 text-[10px] text-zinc-400">
              Örnek satır (yuvarlı): [
              {(Wl?.[0] ?? []).map((x) => fmt(x, 4)).join(", ")}]
            </p>
          </div>
        );
      })}
    </div>
  );
}
