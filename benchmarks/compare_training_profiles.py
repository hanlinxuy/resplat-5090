"""Validate alignment and quantify numerical differences between two profile runs."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--loss-only", action="store_true",
                        help="Compare aligned losses for runs with different parameter audit coverage")
    args = parser.parse_args()
    reference, candidate = (json.loads(p.read_text()) for p in (args.reference, args.candidate))
    for field in ("gpu", "capability", "total_params", "trainable_params", "accumulation"):
        if reference.get(field) != candidate.get(field):
            parser.error(f"runs differ on {field}")
    if len(reference["samples"]) != len(candidate["samples"]) or not reference["samples"]:
        parser.error("runs must have the same nonempty sample count")
    losses = []
    for a, b in zip(reference["samples"], candidate["samples"]):
        for field in ("step", "update", "scenes", "scene_keys", "context_views", "refinement_iterations"):
            if a.get(field) != b.get(field):
                parser.error(f"mismatched update or data in {field}")
        if not (math.isfinite(a["loss"]) and math.isfinite(b["loss"])):
            parser.error("found a non-finite loss")
        losses.append(abs(a["loss"] - b["loss"]))
    if args.loss_only:
        print(json.dumps(dict(max_absolute_loss_diff=max(losses), aligned_microbatches=len(losses)), indent=2))
        return
    audits = reference.get("parameter_audit", {}), candidate.get("parameter_audit", {})
    if audits[0].keys() != audits[1].keys():
        parser.error("parameter audit fields differ")
    errors = {}
    for name in audits[0]:
        errors[name] = {}
        if audits[0][name].keys() != audits[1][name].keys():
            parser.error(f"optimizer snapshot fields differ for {name}")
        for field, a in audits[0][name].items():
            b = audits[1][name][field]
            if len(a) != len(b):
                parser.error(f"sample length differs for {name}/{field}")
            if not all(math.isfinite(x) for x in (*a, *b)):
                parser.error(f"non-finite sample for {name}/{field}")
            norm = max(1e-30, math.sqrt(sum(x * x for x in a)))
            error = math.sqrt(sum((x - y)**2 for x, y in zip(a, b))) / norm
            errors[name][field] = error
    print(json.dumps(dict(max_absolute_loss_diff=max(losses), aligned_microbatches=len(losses),
                          sampled_relative_l2=errors), indent=2))


if __name__ == "__main__":
    main()
