import unittest
from src.data_processing.preprocess import preprocess_data
import pandas as pd

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_data(self):
        df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
        result = preprocess_data(df)
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)

if __name__ == '__main__':
    unittest.main()
