"""Checks for the public configuration-based cluster execution guard."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime_config import assert_project


class RuntimeConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        (self.root / "configs").mkdir()
        self.config_path = self.root / "configs/osc.json"

    def tearDown(self):
        self.directory.cleanup()

    def test_configured_user_and_checkout_are_accepted(self):
        self.config_path.write_text(json.dumps({"username": "test-user", "project_dir": str(self.root)}))
        with patch("runtime_config.getpass.getuser", return_value="test-user"), patch.dict(
            "os.environ", {"OSC_USERNAME": "test-user", "OSC_PROJECT_DIR": str(self.root)}
        ):
            assert_project(self.root)

    def test_different_user_is_rejected(self):
        self.config_path.write_text(json.dumps({"username": "test-user", "project_dir": str(self.root)}))
        with patch("runtime_config.getpass.getuser", return_value="another-user"), patch.dict(
            "os.environ", {"OSC_USERNAME": "test-user", "OSC_PROJECT_DIR": str(self.root)}
        ):
            with self.assertRaisesRegex(RuntimeError, "configured user"):
                assert_project(self.root)

    def test_public_placeholders_are_rejected_before_generation(self):
        self.config_path.write_text(json.dumps({"username": "YOUR_USERNAME", "project_dir": "/path/to/code-switching"}))
        with patch.dict("os.environ", {"OSC_USERNAME": "YOUR_USERNAME", "OSC_PROJECT_DIR": "/path/to/code-switching"}):
            with self.assertRaisesRegex(RuntimeError, "Configure your own"):
                assert_project(self.root)


if __name__ == "__main__":
    unittest.main()
