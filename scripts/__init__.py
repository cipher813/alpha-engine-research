"""Maintenance + reporting scripts.

Most modules here are CLI tools (``python scripts/<name>.py``); a subset
(``aggregate_costs``) is also imported from ``lambda/handler.py`` so the
Research Lambda can run the daily cost-aggregation step in-process at
the end of every successful invocation. This file exists to mark
``scripts/`` as an explicit Python package so the import path resolves
inside the Lambda Docker image — implicit namespace packages would also
work, but the explicit marker keeps the import contract visible.
"""
