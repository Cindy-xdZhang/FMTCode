"""Apply the currently authorized 16h arc-length rule to new merged geometry."""
from pathlib import Path
import numpy as np
from FMT_Utils.Task6CoreLength_3D import filter_core_lengths
from experiments.Filter_Task6_CoreLength_1_4 import read_cores,save_json,sha256
from experiments.Extract_Task6_VortexCore_1_2 import write_lines


def filter_merged_corelines(merged,h):
    retained,lengths,keep=filter_core_lengths(merged,h,minimum_h=16.)
    return retained,dict(version='Task6CPPExtraction_2.5',stage='after strict gap<4h merging; before sampling',
        h=float(h),minimum_length_h=16.,minimum_length=16*h,keep_comparison='>=',
        merged_count=len(merged),retained_count=len(retained),lengths=lengths.tolist(),
        retained_merged_ids=np.flatnonzero(keep).tolist(),removed_merged_ids=np.flatnonzero(~keep).tolist(),
        total_retained_length=float(lengths[keep].sum()),coreline_ivd_filter=False,winding_filter=False,
        ivd_use='original-v IVD > mean for sample seeding only')


def save_filtered_corelines(folder,h):
    folder=Path(folder);source=folder/'vtk_merged.vtp'
    retained,report=filter_merged_corelines(read_cores(source),h)
    write_lines(folder/'corelines.vtp',retained)
    report.update(merged_sha256=sha256(source),corelines_sha256=sha256(folder/'corelines.vtp'))
    save_json(folder/'length_filter.json',report)
    return retained,report
