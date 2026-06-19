# CF-VEdit Baselines

`baselines/` contains scripts only. Baseline videos and metadata are written to
`predictions/<run_name>/`.

## copy_source

Lower-bound anchor. It copies each source video to the prediction slot without
performing the edit:

```bash
python baselines/copy_source.py --run-name copy_source
python bench.py validate copy_source
python bench.py score copy_source --judge vlm
python bench.py report copy_source
```

Expected shape: preservation is high, consequence and edit-success are low.

## free_regen

Upper-bound anchor. The benchmark does not call a generator; this script packages
externally regenerated videos that were produced without conditioning on the
source videos:

```bash
python baselines/free_regen.py --generated-dir /path/to/free_regen_videos --run-name free_regen
```

Expected shape for a complete external run: consequences can be high, source
preservation should be low.
