import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
E2W_ROOT = PACKAGE_ROOT.parents[1]
for path in (PACKAGE_ROOT, E2W_ROOT / "packages" / "e2w_core"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from e2w_generation.void_renderer import VoidRenderer, VoidRendererConfig


class TemporalContractTest(unittest.TestCase):
    def test_quadmask_frame_count_must_match_video_before_backend_load(self):
        class NoBackendRenderer(VoidRenderer):
            def _load_backend(self):
                raise AssertionError("backend should not load for temporal validation")

        renderer = NoBackendRenderer(VoidRendererConfig(base_path="/unused", void_pass1_path="/unused"))
        video = np.zeros((1, 3, 3, 4, 4), dtype=np.float32)
        quadmask = np.zeros((2, 4, 4), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "quadmask.*frames.*video"):
                renderer.render_from_quadmask(video, quadmask, "remove object", Path(tmp) / "out.mp4")


if __name__ == "__main__":
    unittest.main()
