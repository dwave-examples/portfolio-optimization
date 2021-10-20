# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import unittest
import os
import sys

import pandas as pd 

from single_period import SinglePeriod

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestSmoke(unittest.TestCase):
    @unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
    def test_smoke(self):
        """Run portfolio.py and check that nothing crashes"""

        demo_file = os.path.join(project_dir, 'portfolio.py')
        subprocess.check_output([sys.executable, demo_file])

class TestDemo(unittest.TestCase):
    """Verify models are build correctly."""
    def test_build_dqm(self):
        test_portfolio = SinglePeriod(bin_size=5, model_type='DQM')

        data = {'IBM': [93.043, 84.585, 111.453, 99.525, 95.819],
                'WMT': [51.826, 52.823, 56.477, 49.805, 50.287]}

        idx = ['Nov-00', 'Dec-00', 'Jan-01', 'Feb-01', 'Mar-01']

        df = pd.DataFrame(data, index=idx)

        test_portfolio.load_data(df=df)
        test_portfolio.build_dqm()

        self.assertEqual(test_portfolio.model['DQM'].num_variables(), 12)
        self.assertEqual(test_portfolio.model['DQM'].num_cases(), 30)

    def test_build_cqm(self):
        test_portfolio = SinglePeriod(bin_size=5, model_type='DQM')

        data = {'IBM': [93.043, 84.585, 111.453, 99.525, 95.819],
                'WMT': [51.826, 52.823, 56.477, 49.805, 50.287]}

        idx = ['Nov-00', 'Dec-00', 'Jan-01', 'Feb-01', 'Mar-01']

        df = pd.DataFrame(data, index=idx)

        test_portfolio.load_data(df=df)
        test_portfolio.build_cqm()

        self.assertEqual(len(test_portfolio.model['CQM'].variables), 4)
        self.assertEqual(len(test_portfolio.model['CQM'].constraints), 3)

class TestIntegration(unittest.TestCase):
    @unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
    def test_cqm_integration(self):
        """Test integration of portfolio script default cqm run."""

        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        demo_file = os.path.join(project_dir, 'portfolio.py')

        output = subprocess.check_output([sys.executable, demo_file])
        output = output.decode('utf-8') # Bytes to str
        output = output.lower()

        self.assertIn('cqm run', output)
        self.assertIn('cqm formulation', output)
        self.assertIn('best feasible solution', output)
        self.assertIn('estimated returns', output)
        self.assertIn('purchase cost', output)
        self.assertIn('variance', output)
        self.assertNotIn('traceback', output)

    @unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
    def test_dqm_integration(self):
        """Test integration of portfolio script default run."""

        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        demo_file = os.path.join(project_dir, 'portfolio.py')

        output = subprocess.check_output([sys.executable, demo_file] + ["-m", "DQM"])
        output = output.decode('utf-8') # Bytes to str
        output = output.lower()

        self.assertIn('dqm run', output)
        self.assertIn('dqm -- solution', output)
        self.assertIn('shares to buy', output)
        self.assertIn('estimated returns', output)
        self.assertIn('purchase cost', output)
        self.assertIn('variance', output)
        self.assertNotIn('traceback', output)