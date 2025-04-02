import unittest
from src.integration.alert_processor import parse_snort_alert

class TestAlertProcessing(unittest.TestCase):
    def test_parse_snort_alert(self):
        sample_alert = "[**] [1:1000001:0] Test alert [**] [Priority: 1] {TCP} 192.168.1.1:1234 -> 192.168.1.2:80"
        result = parse_snort_alert(sample_alert)
        self.assertIsNotNone(result)
        self.assertEqual(result["source_ip"], "192.168.1.1")

if __name__ == '__main__':
    unittest.main()
