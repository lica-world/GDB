# gdb verify fixture

Tiny bundled dataset used by `gdb verify` to confirm an install is
functional without any downloads or API keys. Covers the
`v0-smoke` suite only.

**Do not edit by hand.** Regenerate with:

```bash
python scripts/build_verify_dataset.py
```

Images are downsampled to 128px on the long edge; scores
produced against this fixture are **meaningless** by design.
