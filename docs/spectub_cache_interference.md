---
orphan: true
---

# SPECTUB all-views-cache interference

This page documents a known limitation of the SIMIND-corrected reconstruction
path shipped in **2.0.1**. It records what was observed, how it can be
reproduced, and the remedies that were considered but **not applied**. It is a
historical record, not a claim about every backend version.

## Status and impact

- **Known limitation, shipped in 2.0.1, not fixed.**
- When two separately set-up SPECTUB acquisition models are used together
  (for example a `fast` model and an `accurate` model) and all-views caching is
  left enabled, the models can interfere through SPECTUB's shared all-views
  cache. This shows up as a residual correction that is inconsistent with a
  freshly recomputed projection on the same, unchanged update image, and as
  run-to-run variability in the reconstruction endpoint.
- The effect is small at the operator level (a handful of bins differ at the
  first update) but it can shift the SIMIND quantised source map by one
  integer, which changes the random stream and therefore the recorded
  reconstruction result.
- This is reported for the environment below only. It is **not** a general
  statement about every SIRF/STIR release, and it should not be generalised to
  historical endpoint differences that were never reproduced.

## Evidence and provenance

All results in this section are **historical**, recorded on **2026-10-09** in
the environment:

| Component | Version |
|---|---|
| SIRF | 3.10.1 |
| STIR | 6.4 |
| SIMIND | V8.0 |

They were not re-run for this page.

### Direct isolation (cache ON versus cache OFF)

Two separately set-up SPECTUB acquisition models were driven through the
`ResidualCorrection` / `run_osem_with_corrections` path:

- `fast`: no point-spread function;
- `accurate`: `fast` plus the recorded **6 mm** `SeparableGaussianImageFilter`.

The captured additive was compared with a freshly recomputed
`background + accurate.direct(x) - fast.direct(x)` on the **same unchanged
update image**.

- **Cache ON:** at the first update, **32 of 32768 bins** differed, with a
  maximum absolute difference of approximately **0.0922077**. At the later
  recorded updates the difference was **zero**.
- **Cache OFF** on all SPECTUB matrices (`set_keep_all_views_in_cache(False)`):
  the difference was **exactly zero at all three recorded updates**.

This isolates the SPECTUB all-views cache as the cause of the 32-bin operator
interference.

### Cache-off example-10 reproduction

Two `example 10` / base-100 runs with the cache disabled on all SPECTUB
matrices agreed:

- hot-sphere contrast error **22.4% → 16.1%** in both runs;
- refresh scales **1193 / 964.5 / 1011**;
- scatter sums **3.543e5 / 2.79e5 / 3.024e5**.

The cache-ON endpoint split between 16.4% and 18.14% disappeared.

A third identical run confirmed SIMIND-side reproducibility:

- the integer source map, the raw `b01`/`b02` outputs and the unscaled
  acquisition arrays were identical across runs;
- only the SIMIND-generated `.h00` header differed, and only in its study
  date/time fields.

### Provenance notes

- An earlier log statement that the cache-off reproduction "was not run" was
  **superseded**; the two cache-off base-100 runs above were subsequently
  performed.
- These results are **separate** from the earlier `example 09` floating
  differences and the earlier unreproduced endpoint observations. Those remain
  unconfirmed observations in their own right; this page does not claim the
  cache explains them.

## Reproduction protocol

1. **Set up two separate models.** Build a `fast` SPECTUB acquisition model
   (no PSF) and an `accurate` model that adds the recorded 6 mm
   `SeparableGaussianImageFilter`, each set up independently.
2. **Compare at each update.** Capture the additive used during
   reconstruction and compare it, on the same unchanged update image, against a
   freshly recomputed `background + accurate.direct(x) - fast.direct(x)`.
   Record, at each update, the **number of changed bins** and the **maximum
   absolute difference**.
3. **Toggle the cache without changing shipped code.** Run the same
   comparison with all-views caching enabled and disabled on all SPECTUB
   matrices. Keep everything else fixed.
4. **Repeat example 10 under a fixed seed sequence.** Hold configuration,
   input data, environment and the seed base fixed, with the seed sequence
   `[100, 101, 102]` applied at updates 10 / 20 / 30 (that is, the actual `RR`
   sequence `[base, base+1, base+2]` at those iterations).
5. **Capture raw artifacts.** For each update record the float source image,
   the effective configuration and the actual `RR`. Hash the generated `.smi`
   after creation and before launch, and also hash the raw `b01`/`b02` outputs
   and the unscaled acquisition arrays.
6. **Do not use timestamp-bearing headers as raw reproducibility evidence.**
   The generated `.h00` header carries a study date/time and is expected to
   differ between runs even when every raw input and output is identical.

## Potential fixes — not applied

None of the following was implemented or shipped; they are recorded for a
future, separately approved change.

- **Disable all-views caching in the affected owned model factories.** The
  smallest change would sit around the reconstruction model factories that
  create the interleaved SPECTUB models.
- **Establish cache isolation or reset across all participating models.** A
  mechanism that resets or partitions the all-views cache so that independently
  set-up models cannot observe each other's cached views.
- **Investigate an upstream cache-lifecycle correction.** Determine whether
  SIRF or STIR should isolate this state per model and, if so, correct it
  upstream rather than in this package.
- **Constructing a fresh OSEM model is not demonstrated sufficient.** That
  pattern already exists in the current code while other models remain live, so
  it has not been shown to remove the interference.
