#!/usr/bin/env python3
"""Execute the device tests' actual environment guards on a CPU, without HIP.

Issue #3063: presence checks selected different counters from production.
Only the guard between architecture validation and tensor setup is extracted.
This is a harness regression test, not GPU dispatch or numerical evidence.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'VT_ROCM_QUANT_WMMA'
KEYS = (PREFIX, PREFIX + '_WIDE', PREFIX + '_BIGTILE', PREFIX + '_SHARE_ACT')

# Master WMMA defaults on and only exact zero disables it. The experimental
# switches default off and require exact one. BigTile precedes Shared.
CASES = (
    ((None, None, None, None), 'none'),
    ((None, '1', None, None), 'none'),
    ((None, None, '1', None), 'none'),
    ((None, None, None, '1'), 'none'),
    ((None, '1', '1', None), 'bigtile'),
    ((None, '1', None, '1'), 'shared'),
    ((None, '1', '1', '1'), 'bigtile'),
    ((None, '1', '0', '1'), 'shared'),
    ((None, '1', '1', '0'), 'bigtile'),
    ((None, '1', '0', '0'), 'none'),
    ((None, '1', '', '1'), 'shared'),
    ((None, '1', '01', '1'), 'shared'),
    ((None, '1', '10', '1'), 'shared'),
    ((None, '1', 'true', '1'), 'shared'),
    ((None, '1', None, ''), 'none'),
    ((None, '1', None, '01'), 'none'),
    ((None, '1', None, '10'), 'none'),
    ((None, '1', None, 'true'), 'none'),
    ((None, '0', '1', '1'), 'none'),
    ((None, '', '1', '1'), 'none'),
    ((None, '01', '1', '1'), 'none'),
    ((None, '10', '1', '1'), 'none'),
    ((None, 'true', '1', '1'), 'none'),
    (('0', '1', '1', '1'), 'none'),
    (('0', '1', None, '1'), 'none'),
    (('1', '1', '1', '1'), 'bigtile'),
    (('', '1', '1', '1'), 'bigtile'),
    (('00', '1', None, '1'), 'shared'),
    (('false', '1', None, '1'), 'shared'),
)


def guard_source(source, quant):
    marker = f'TEST_CASE("keep-quant {quant} WMMA cooperative-tile arms match the CPU oracle")'
    if source.count(marker) != 1:
        raise AssertionError(f'expected one cooperative-tile case for {quant}')
    case = source.split(marker, 1)[1].split('\nTEST_CASE(', 1)[0]
    start = case.index('  const char* wide =')
    end = case.index('  constexpr int64_t M =', start)
    return case[start:end]


class CooperativeToggleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++')
        if compiler is None:
            raise RuntimeError('a host C++ compiler is required for the guard regression')
        cls.temp = tempfile.TemporaryDirectory(prefix='rocm-coop-toggle-')
        cls.addClassCleanup(cls.temp.cleanup)
        directory = Path(cls.temp.name)
        source = (ROOT / 'tests/vt/test_backend_cross_device.cpp').read_text()
        cls.binaries = {}
        for quant in ('Q4_K', 'Q6_K'):
            cpp = directory / (quant + '.cpp')
            binary = directory / quant
            cpp.write_text(
                '#include <cstdlib>\n#include <iostream>\n'
                '#define MESSAGE(...) ((void)0)\n'
                'void RunHarness() {\n' + guard_source(source, quant) +
                '  std::cout << (bigtile ? "bigtile" : "shared");\n}\n'
                'int main() { RunHarness(); }\n')
            result = subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror',
                                     str(cpp), '-o', str(binary)], capture_output=True, text=True)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
            cls.binaries[quant] = binary

    def check_matrix(self, quant):
        for values, expected in CASES:
            with self.subTest(quant=quant, toggles=dict(zip(KEYS, values))):
                env = {key: value for key, value in os.environ.items() if key not in KEYS}
                env.update({key: value for key, value in zip(KEYS, values) if value is not None})
                result = subprocess.run([str(self.binaries[quant])], env=env,
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout or 'none', expected)

    def test_q4k_toggle_matrix(self):
        self.check_matrix('Q4_K')

    def test_q6k_toggle_matrix(self):
        self.check_matrix('Q6_K')


if __name__ == '__main__':
    unittest.main(verbosity=2)
