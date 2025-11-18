#!/usr/bin/env python3
"""
dedupe_geojson.py

Remove duplicate station features from a FeatureCollection GeoJSON.

Usage:
    python dedupe_geojson.py input.geojson output.geojson
Options:
    --tol TOL        : coordinate tolerance in degrees (default 1e-5)
    --no-merge-lines : if set, duplicates are removed (keep first) without merging 'lines'
"""

import json
import argparse
import math
from collections import defaultdict

def parse_args():
    p = argparse.ArgumentParser(description="Deduplicate GeoJSON station features (merge lines).")
    p.add_argument("input", nargs='?', help="Input GeoJSON file (FeatureCollection)")
    p.add_argument("output", nargs='?', help="Output cleaned GeoJSON file")
    p.add_argument("--tol", type=float, default=1e-5,
                   help="Coordinate tolerance in degrees for merging (default 1e-5)")
    p.add_argument("--no-merge-lines", action="store_true",
                   help="Do NOT merge `lines` lists; just keep first occurrence")
    return p.parse_args()

def coord_close(c1, c2, tol):
    # c = (lon, lat)
    return abs(c1[0] - c2[0]) <= tol and abs(c1[1] - c2[1]) <= tol

def normalize_name(name):
    if name is None:
        return ""
    return " ".join(name.strip().lower().split())

def main():
    args = parse_args()
    with open("delhi_metro_stations.geojson", "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") != "FeatureCollection":
        raise SystemExit("Input GeoJSON must be a FeatureCollection")

    features = data.get("features", [])
    kept_features = []
    # index by normalized name -> list of indices in kept_features
    name_index = defaultdict(list)
    tol = args.tol

    removed_count = 0
    merged_count = 0

    for feat in features:
        # basic validation
        geom = feat.get("geometry", {})
        props = feat.get("properties", {}) or {}
        if geom.get("type") != "Point" or not isinstance(geom.get("coordinates", None), (list, tuple)):
            # treat non-Point as unique (or you could extend merge logic)
            kept_features.append(feat)
            continue

        coords = tuple(geom["coordinates"])  # (lon, lat)
        name = normalize_name(props.get("name", ""))

        is_duplicate = False
        if name in name_index:
            # check existing kept features with same normalized name for coordinate closeness
            for idx in name_index[name]:
                kept = kept_features[idx]
                kcoords = tuple(kept["geometry"]["coordinates"])
                if coord_close(coords, kcoords, tol):
                    # Found duplicate — merge or skip
                    is_duplicate = True
                    removed_count += 1

                    if not args.no_merge_lines:
                        # merge 'lines' property (union) when available
                        lines_new = props.get("lines", [])
                        lines_old = kept.get("properties", {}).get("lines", [])
                        # normalize items, preserve order-ish (old first then new uniques)
                        merged = []
                        for x in (lines_old or []):
                            if x not in merged:
                                merged.append(x)
                        for x in (lines_new or []):
                            if x not in merged:
                                merged.append(x)
                        if merged != (lines_old or []):
                            kept_features[idx]["properties"] = dict(kept_features[idx].get("properties", {}))
                            kept_features[idx]["properties"]["lines"] = merged
                            merged_count += 1
                    # we keep first geometry/properties otherwise
                    break

        if not is_duplicate:
            # add this feature
            name_index[name].append(len(kept_features))
            kept_features.append(feat)

    out = {
        "type": "FeatureCollection",
        "features": kept_features
    }
    with open("delhi_metro_stations_cleaned.geojson", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Input features : {len(features)}")
    print(f"Output features: {len(kept_features)}")
    print(f"Removed duplicates: {removed_count}")
    print(f"Features merged (lines updated): {merged_count}")
    print(f"Saved cleaned GeoJSON to: {args.output}")

if __name__ == "__main__":
    main()
