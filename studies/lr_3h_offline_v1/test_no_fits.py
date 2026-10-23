import ast
import unittest
from pathlib import Path
from . import offline
class OfflineTests(unittest.TestCase):
    def test_zero_fit_sites(self):
        for name in ('offline.py','verify.py'):
            tree=ast.parse(Path(__file__).with_name(name).read_text())
            self.assertFalse([n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='fit'])
    def test_fixed_contract(self):
        self.assertEqual(len(offline.EXTRA),6)
        self.assertEqual(list(offline.MEANS),[offline.EXTRA[0],offline.EXTRA[3]])
        self.assertEqual(offline.MODEL_ID,'offline-lr3h-ca23fe0-v1')
if __name__=='__main__':unittest.main()
