# Vendored frontend assets

Pinned third-party JS the Coach UI loads directly from
`/coach/static/vendor/`. No build step, no npm — these files are checked
in as-is and served to the browser.

## chart.umd.min.js

| Field | Value |
|---|---|
| Library | Chart.js |
| Version | 4.4.0 |
| Source | https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js |
| License | MIT |
| Used by | `coach/static/index.html` (renders the live loss curve in step 4) |
| Fallback | If the file is missing or fails to load, `coach.js` checks `typeof Chart === "undefined"` and degrades to a metrics-only table — the page does not break. |

The repo treats this file as a vendored binary artifact: do not edit it
in place. To upgrade, replace the file from the upstream CDN URL above
and update this row plus the SHA256 below.

### SHA256

| File | Hash |
|---|---|
| `chart.umd.min.js` | `0e2326c6868072bec1592760c6729043caeea2960a2b46cee6a2192aac6abff0` |
| Size | 201 KB |
| Vendored on | 2026-05-08 |

Verify locally:

```sh
sha256sum mindxtrain/operator/coach/static/vendor/chart.umd.min.js
# → 0e2326c6868072bec1592760c6729043caeea2960a2b46cee6a2192aac6abff0  …
```

To upgrade, replace the file from the upstream CDN URL above and update
this row.

## d3.v7.min.js

| Field | Value |
|---|---|
| Library | D3.js |
| Version | 7.9.0 |
| Source | https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js |
| License | ISC |
| Used by | `coach/static/index.html` (chronos drift sparkline + 24h anchor density bar in `#step-chronos`) |
| Fallback | If d3 isn't loaded, `coach.js` checks `typeof d3 === "undefined"` and the chronos card renders text-only (UTC + consensus tier + confidence band) — the page does not break. |

### SHA256

| File | Hash |
|---|---|
| `d3.v7.min.js` | `f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539` |
| Size | 274 KB |
| Vendored on | 2026-05-19 |

Verify locally:

```sh
sha256sum mindxtrain/operator/coach/static/vendor/d3.v7.min.js
# → f2094bbf6141b359722c4fe454eb6c4b0f0e42cc10cc7af921fc158fceb86539  …
```

three.js intentionally **not** vendored. The time-drift story is 2D
(sparkline + density bars) and d3 covers it cleanly. A future
agent-topology view would justify three.js (~600 KB) on its own merits.
