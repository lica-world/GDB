"""Shared utilities for the design benchmarks.

``data_helpers``
    Data loading primitives shared across task modules: CSV sample loading,
    JSON task-file loading, and ``ModelInput`` construction.

``text_helpers``
    Model-output text processing: strip thinking blocks, code fences,
    extract JSON, and normalized edit distance.

``image_helpers``
    Image array conversion, mask handling, OCR, and statistical helpers.

``template_layout_paths``
    Resolve layout/image/annotation paths from the Lica dataset tree.
"""
