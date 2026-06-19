# tools — repo-level drift guards

Machine checks that turn the proposal's boundaries into merge preconditions
(Repo-Design §4⑥). Run in CI (`.github/workflows/ci.yml`).

| guard | enforces | how |
|---|---|---|
| **import-linter** | B1 (benchmark ⊥ model), B3 (halves independent), e2w_core inward | `lint-imports` against `pyproject.toml [tool.importlinter]` |
| **schema validate** | manifest/contract field shape | `bench.py validate-manifest` |
| **spec-test** | the benchmark invariants (two-axis metric, anchors, gating) | `unittest tests.test_cf_vedit_benchmark` |
| **third_party-clean** | B4 (vendored upstream has no in-place diff) | diff `third_party/` against pinned commit |
| **proposal-link** | module READMEs back-link to real proposal sections | (to add) check anchors resolve |

Most guards are scaffolded now and become enforcing as the packages and vendored
upstreams land (the contracts already exist in `pyproject.toml`).
