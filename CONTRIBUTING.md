# Contributing to bezpy

Contributions are welcome, especially reproducible validation cases, data
reader improvements, and complete geoelectric-hazard workflows.

## Development setup

```bash
git clone https://github.com/greglucas/bezpy
cd bezpy
python -m pip install -e ".[test,docs]"
pytest
ruff check bezpy tests scripts benchmarks
mkdocs build --strict
```

Tests must not depend on live web services. Use a small synthetic or archived
fixture and mock the transport layer, following the acquisition tests in
`tests/test_data.py`.

## Scientific changes

A change that can alter a numerical result should include:

- a focused regression test with a physical invariant, independently computed
  reference, or comparison against a named external implementation;
- explicit units, coordinates, Fourier sign, missing-data, and uncertainty
  assumptions in its public docstring;
- an entry under **Result-changing fixes** in `CHANGELOG.md`;
- an update to `docs/validation.md` when it changes a validated method or default.

Do not silently replace an established method. Preserve it behind an explicit
option when scientifically defensible, or document why recreating it would
reintroduce a bug.

## Pull requests

Keep changes scoped and explain the scientific motivation, validation data,
and expected numerical effect. Add public API documentation and a tutorial
when introducing a new workflow. Do not commit downloaded data, credentials,
local caches, generated documentation, or planning notes.

By contributing, you agree that your contribution is released under the
repository's MIT license.
