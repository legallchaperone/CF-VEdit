import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gemini_judge.py"
spec = importlib.util.spec_from_file_location("gemini_judge_script", SCRIPT)
gemini_judge = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gemini_judge)


class SourceFrameFallbackTest(unittest.TestCase):
    def test_source_frame_count_falls_back_to_probe_before_args_frames(self):
        class FakeVlmJudge:
            @staticmethod
            def probe_num_frames(path):
                return 21 if str(path).endswith("source.mp4") else None

        got = gemini_judge._source_frame_count(
            {"video_meta": {}},
            Path("/tmp/source.mp4"),
            8,
            FakeVlmJudge,
        )
        self.assertEqual(got, 21)


if __name__ == "__main__":
    unittest.main()
