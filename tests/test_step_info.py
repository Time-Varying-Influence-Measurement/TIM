# test infl.load_step_data

import unittest
import os
from experiment.infl import load_step_data

class TestStepInfo(unittest.TestCase):
    def test_load_step_data(self):
        save_dir = "test"  # Use only the test directory
        dir_name_base = "experiment"
        step = 1
        seed = 0
        relabel_percentage = None
        dir_name = os.path.join(dir_name_base, save_dir)
        try:
            data = load_step_data(dir_name, step, seed, relabel_percentage)
        except FileNotFoundError as e:
            self.skipTest(f"Test data not found: {e}")
        self.assertIsInstance(data, dict)
        self.assertIn("model_state", data)
        self.assertIn("idx", data)
        self.assertIn("lr", data)
        self.assertIsInstance(data["model_state"], dict)

if __name__ == "__main__":
    unittest.main()
