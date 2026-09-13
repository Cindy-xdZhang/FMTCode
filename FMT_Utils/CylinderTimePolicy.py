"""Metadata-only late-time selection for cylinder visualizations."""
import json
from pathlib import Path


def choose_cylinder_slice(manifest, dataset, policy):
    ranges = policy['simulation_time_ranges']
    if dataset not in ranges:
        raise ValueError(f'Original simulation time range must be registered for {dataset}.')
    start, end = map(float, ranges[dataset])
    cutoff = max(float(policy['minimum_physical_start']),
                 start + float(policy['minimum_original_simulation_fraction']) * (end - start))
    eligible = [s for s in manifest['slices'] if float(s['source_time']) >= cutoff]
    if not eligible:
        raise ValueError(f'No cached time meets {dataset} cutoff t>={cutoff}; create a new late-time cache.')
    selected = min(eligible, key=lambda s: (float(s['source_time']), int(s['ordinal'])))
    return int(selected['ordinal']), {
        'time_policy_version': policy['policy_version'], 'minimum_start_time': cutoff,
        'original_simulation_range': [start, end],
        'time_selection_rule': policy['cached_slice_selection'],
        'source_file_range': [manifest['source']['time_min'], manifest['source']['time_max']],
        'selected_physical_start': float(selected['source_time'])}


def select_from_files(cache_dir, dataset, policy_path):
    manifest = json.loads((Path(cache_dir) / 'manifest.json').read_text())
    policy = json.loads(Path(policy_path).read_text())
    return choose_cylinder_slice(manifest, dataset, policy)
