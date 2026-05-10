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
