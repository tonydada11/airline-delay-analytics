import ast
import unittest
from pathlib import Path
from studies.lr_v1.metrics import MONTHS, expected_budget, monthly_comparison

class ProtocolTests(unittest.TestCase):
    def test_fractional_duplicate_boundary_and_floor(self):
        result=expected_budget([(0.9,'same',1),(0.9,'same',0)]+[(0.1,str(i),0) for i in range(8)],10)
        self.assertEqual(result['capacity'],1)
        self.assertEqual(result['recall'],0.5)
        self.assertEqual(result['selected_mass'],1)
    def test_all_months_and_both_budget_gates(self):
        def values(pr,rec):return {m:{'pr_auc':pr,'budgets':{'10':{'recall':rec},'20':{'recall':rec}}} for m in MONTHS}
        self.assertTrue(monthly_comparison(values(.4,.3),values(.406,.3))['qualifies'])
        self.assertFalse(monthly_comparison(values(.4,.3),values(.406,.28))['qualifies'])
        self.assertFalse(monthly_comparison(values(.4,.3),values(.404,.3))['qualifies'])
        with self.assertRaises(ValueError):monthly_comparison(values(.4,.3),{})
    def test_single_fit_site_and_frozen_prefix_assembly(self):
        tree=ast.parse(Path(__file__).with_name('experiment.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='fit']
        self.assertEqual(len(calls),1)
        self.assertEqual(ast.unparse(calls[0].func),'estimator.fit')
        self.assertIn("inputCols=['original_features', *coordinates]",ast.unparse(tree))
if __name__=='__main__':unittest.main()
