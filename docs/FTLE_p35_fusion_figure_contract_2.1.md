# FTLE P35 2.1 figure contract

> **2026-09-17撤销状态：** 本文涉及的Raw卷积＋FMT/残差或二维FMT卷积网络，已按用户要求从可用方案及有效FMT证据中删除；旧数值仅为撤销历史，不得用于当前主表、附录或“最佳FMT”结论。固定傅里叶特征本身及明确无卷积的Task1/Task4-c分支不受此撤销影响。以 [FMT禁止空间卷积协议1.1](FMT_no_spatial_convolution_protocol_1.1.md) 为准。


Question: Does the frozen P35 fusion improve held-out FTLE accuracy beyond ESPCN and beyond a decoder with the same architecture, and where does it fail?

The signed measured differences determine the conclusion; zero and negative effects remain visible. No diagram or selected image is used as a substitute for the full paired measurements.

- Main quantitative figure: two panels. Left compares P35 against ESPCN, the stronger historical scalar baseline. Right compares P35 against matched no-geometry and Raw inputs, separating architecture from geometric representation. Every flow and the equally weighted four-flow mean appear at both scales. Points are means over three optimizer seeds; error bars are one sample standard deviation of paired seed differences. Three time slices are averaged within each run, not counted as independent experimental replicates.
- Field plates: one figure per flow and scale. Always use the first prescribed final seed and first test slice. Ground truth and the shared evaluation support establish context; each displayed method contributes its reconstructed field and absolute error. Include ESPCN, U-Net, matched no-geometry, Raw, and the selected P35 method. Additional signed-only controls remain in the full numeric tables. These plates are illustrative, with no uncertainty bars or best-case selection.
- Backend: existing saved Python preference; matplotlib only. Original source script, no external plotting template.
- Export: 180 mm wide PDF with editable text, SVG, 600 dpi TIFF, and PNG preview. All glyphs at least5 pt. White background, shared sequential FTLE and error color scales within a plate, no clipped residual outliers. Axis transposition for tall domains is recorded and coordinate labels are exchanged accordingly. Invalid/excluded evaluation pixels are grey; no values in the scored region are omitted.
- Each plate preserves physical aspect ratio, comparable plot-area dimensions, physical coordinates, and full source arrays. The quantitative figure uses the complete paired data and exports its plotting table. Source prediction/config/selection hashes and exact times/seed are recorded.
- Mandatory QA: source validator, final-render plot-area measurement at1.5 pt tolerance, PDF glyph-size and collision audits, visual inspection of every panel and the whole figure. Negative and zero-gain cases cannot be excluded for appearance.

Figures are generated only after the final independent metric audit passes. This is an internal research report export, not a claim of journal submission compliance.
