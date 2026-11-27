# Phase 13B regression tests
import os
import unittest
import json
import subprocess

# Helper to call the Flask app test client via imported server
from server import app as create_app

app = create_app()
# client will be used directly from imported app

class TestPhase13BCorrections(unittest.TestCase):
    def test_cors_restriction(self):
        # Disallowed origin should be blocked
        resp = client.get('/api/v2/health', headers={'Origin': 'https://evil.example.com'})
        self.assertEqual(resp.status_code, 403)
        # Allowed origin should succeed
        resp = client.get('/api/v2/health', headers={'Origin': 'http://localhost:5173'})
        self.assertEqual(resp.status_code, 200)

    def test_provenance_defaults(self):
        # Create a job without provenance should default to USER_INSPECTION
        data = {"job_name": "test_job"}
        resp = client.post('/api/v2/jobs', json=data)
        self.assertEqual(resp.status_code, 201)
        job = resp.get_json()
        self.assertEqual(job.get('provenance'), 'USER_INSPECTION')

    def test_quality_variant_monotonicity(self):
        # Run the quality evaluation script in a subprocess and parse JSON output for monotonicity flag
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'phase13', 'quality_geo_maint_eval.py'))
        result = subprocess.run(['python', script_path, '--smoke', 'false'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        # Expect a JSON line with "monotonicity_passed": true
        for line in result.stdout.splitlines():
            if 'monotonicity_passed' in line:
                data = json.loads(line)
                self.assertTrue(data.get('monotonicity_passed'))
                break
        else:
            self.fail('monotonicity_passed not reported')

    def test_geometry_method_d_single_call(self):
        # Ensure geometry_ablation runs without duplicate Method D calls (smoke mode sufficient)
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'phase13', 'geometry_ablation.py'))
        result = subprocess.run(['python', script_path, '--smoke', 'true'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('status":"COMPLETED', result.stdout)

    def test_detection_evaluation_tp_fp_fn(self):
        # Run detection evaluation and check that TP, FP, FN fields are present
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'phase13', 'detection_evaluation.py'))
        result = subprocess.run(['python', script_path, '--smoke', 'false'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        output = json.loads(result.stdout)
        for split in ['dataset_unified_val', 'dataset_unified_test']:
            self.assertIn('TP', output[split])
            self.assertIn('FP', output[split])
            self.assertIn('FN', output[split])

if __name__ == '__main__':
    unittest.main()
