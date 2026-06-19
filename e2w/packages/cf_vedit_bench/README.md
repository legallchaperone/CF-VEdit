# cf_vedit_bench — the P0 benchmark (pointer)

The benchmark is **built and working**, but it has not been migrated into this
package yet — see [ADR-0002](../../docs/adr/0002-scaffold-alongside-without-migration.md).

➡️ It currently lives at [`../../../physics_iq_for_simple_eval/`](../../../physics_iq_for_simple_eval/).
Run everything from inside that directory (its `bench.py` resolves paths relative
to itself):

```bash
cd ../../../physics_iq_for_simple_eval
python3 bench.py validate-manifest
python3 -m unittest tests.test_cf_vedit_benchmark -v
```

## When it migrates here

A later PR (tracked in ADR-0002) will `git mv` it into this directory preserving
history, give it a `pyproject.toml` depending only on `e2w_core` (schema subset),
and activate the B1 import-linter contract. Until then this package is a pointer
only — do not duplicate the benchmark here.

The contract it shares with the model — the `predictions/<run>/` shape — is
already defined once in [`e2w_core.io_contract`](../e2w_core/e2w_core/io_contract.py);
the migration will make the benchmark import it instead of re-declaring the
constants.
