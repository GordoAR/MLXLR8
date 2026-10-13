"""Choose fixed hot experts on train routes; score separate held-out routes."""

import argparse
from collections import Counter
import json
from pathlib import Path


def _counts(routes, hot, shared):
    selected = set(hot)
    return {"hits": sum(e in selected for row in routes for e in row),
            "routed_hits": sum(e in selected for row in routes for e in row if e != shared),
            "all_selected_hit_tokens": sum(all(e in selected for e in row) for row in routes)}


def fixed_hot_lists(trace, split=256, shared=256, capacities=(1, 9, 12, 18)):
    layers, totals = {}, {"train": {}, "heldout": {}}
    for name in totals:
        totals[name] = {str(c): dict(tokens=0, selections=0, routed_selections=0,
                                    hits=0, routed_hits=0, all_selected_hit_tokens=0)
                        for c in capacities}
    for number, source in trace["layers"].items():
        routes = source["routes"]
        if len(routes) != 2 * split:
            raise ValueError(f"layer {number} needs {2 * split} routes, got {len(routes)}")
        train, heldout = routes[:split], routes[split:]
        frequency = Counter(e for row in train for e in row if e != shared)
        ranked = sorted(frequency, key=lambda e: (-frequency[e], e))
        layers[number] = {}
        for capacity in capacities:
            hot = [shared] + ranked[:capacity - 1]
            row = {"expert_ids": hot}
            for name, section in (("train", train), ("heldout", heldout)):
                counts = _counts(section, hot, shared)
                row[name] = counts
                total = totals[name][str(capacity)]
                total["tokens"] += len(section)
                total["selections"] += sum(map(len, section))
                total["routed_selections"] += sum(e != shared for r in section for e in r)
                for key, value in counts.items():
                    total[key] += value
            layers[number][str(capacity)] = row
    for section in totals.values():
        for total in section.values():
            total["hit_rate"] = total["hits"] / total["selections"]
            total["routed_hit_rate"] = total["routed_hits"] / total["routed_selections"]
            total["all_selected_hit_token_rate"] = total["all_selected_hit_tokens"] / total["tokens"]
    return {"train_tokens_per_layer": split, "heldout_tokens_per_layer": split,
            "shared_expert": shared, "tie_break": "lowest expert ID",
            "totals": totals, "layers": layers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = fixed_hot_lists(json.loads(Path(args.trace).read_text()))
    result["source"] = args.trace
    Path(args.output).write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
