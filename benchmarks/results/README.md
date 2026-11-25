# Benchmark evidence

`thor/` contains synthetic inputs measured through the real initialization and refinement training entry points, isolated kernel checks, and summaries of profiler samples. `rtx3090/` and `rtx4090/` contain their respective run records. The face folders publish aggregate results across holdout groups. See [benchmark interpretation](../../docs/BENCHMARKS.md) for which pairs share input data, shapes, and training settings.

The training profile schema stores per-microbatch CUDA and wall timings, loss, sample identity, optional parameter audit slices, and allocator peaks. Profiler summaries measure GPU ranges in separate runs. For example, the following reproduces the loss comparison for the same 16 microbatches of the historical effective B32 update.

```sh
python -m benchmarks.compare_training_profiles --loss-only \
  benchmarks/results/thor/full/init-32scenes-local.json \
  benchmarks/results/thor/full/init-32scenes-mixed.json
```
