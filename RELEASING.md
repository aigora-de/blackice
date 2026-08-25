# Releasing

The project is **blackice**; the distribution is **`kuang`**. Releases are cut by
tagging, and published by `.github/workflows/release.yml` using **PyPI Trusted
Publishing** — there is no API token in this repository, and no secret to leak or
rotate.

Per [`ROADMAP.md`](ROADMAP.md): each epoch bumps a minor on completion, and **the
first real PyPI publish is `v0.3.0`, at the close of Epoch 2**. Being installable
is not the same as being fit to hand someone. Until then, releases go to TestPyPI
only — which the workflow enforces by leaving the PyPI job un-approved.

---

## One-time setup

Both steps are manual and must be done **before the first tag**. Neither can be
done from this repository.

### 1. Register the trusted publishers

The `kuang` project does not exist on either index yet, so register a **pending
publisher** on each — that is the flow for a project's first upload.

- TestPyPI: <https://test.pypi.org/manage/account/publishing/>
- PyPI: <https://pypi.org/manage/account/publishing/>

Enter exactly:

| Field | Value |
|---|---|
| PyPI project name | `kuang` |
| Owner | `aigora-de` |
| Repository name | `blackice` |
| Workflow name | `release.yml` |
| Environment name | `testpypi` on TestPyPI · `pypi` on PyPI |

All five must match or the OIDC claim is rejected. Note the third row: the
publisher binds to the **repository** name, not the distribution name — if this
repository is ever renamed, both publisher configurations must be updated or
publishing stops working.

### 2. Create the two environments

In **Settings → Environments**:

- **`testpypi`** — no protection rules. A tag validates here automatically.
- **`pypi`** — **add required reviewers.** This is the release gate. Publishing
  is irreversible: a version can be yanked but never re-uploaded, and never
  replaced. A human approves it, or it does not happen.

---

## Cutting a release

1. Bump `__version__` in `kuang/__init__.py`. It is the single source — the build
   reads it, and `pyproject.toml` declares the version dynamic from it.
2. Commit, land on `main`.
3. Tag and push:
   ```bash
   git tag -a v0.2.0 -m "Epoch 1 — Foundation"
   git push origin v0.2.0
   ```
4. Watch the run. The `build` job runs the suite against the packaged layout,
   checks both entry points, builds, runs `twine check`, refuses a tag whose
   version disagrees with `kuang.__version__`, and refuses a wheel that ships any
   top-level name other than `kuang`.
5. TestPyPI publishes automatically. Validate it as a user would:
   ```bash
   pipx install --index-url https://test.pypi.org/simple/ kuang
   kuang --help
   ```
6. **PyPI waits for approval** and should stay waiting until Epoch 2 closes. When
   it is time, approve the `pypi` environment on the run.

## Building locally

The same steps the `build` job runs, minus the publish:

```bash
pip install '.[release]'
python -m build
python -m twine check dist/*
```
