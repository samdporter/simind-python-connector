---
orphan: true
---

# SPECTUB all-views-cache interference

Separately configured SPECTUB models can share cached views, producing
incorrect residual corrections. Disable all-views caching on every
participating matrix: `matrix.set_keep_all_views_in_cache(False)`. The CIL
Monte Carlo examples use this setting. With SIRF 3.10.1, STIR 6.4 and SIMIND
V8.0, disabling the cache reduced the maximum first-update residual discrepancy
from 0.0222821 to 1.55e-6 against a fresh projection at the same image. These
measurements were recorded on 2026-10-09; other versions were not tested.
