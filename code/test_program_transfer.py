"""Small semantic tests for program transfer, independent of benchmark splits."""
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
from program_transfer import (extract_program, question_numbers, run_program,
    fit_retriever, save_retriever, load_retriever, suggested_answers)


class ProgramTransferTests(unittest.TestCase):
    def test_counterfactual_numbers(self):
        q = 'A pizza has 16 slices, 4 remain, and each person ate 2. How many people ate?'
        a = '<<16-4=12>> remain eaten; <<12/2=6>> people. #### 6'
        p = extract_program(q, a)
        self.assertIsNotNone(p)
        self.assertEqual(run_program(p, question_numbers(q)), Fraction(6))
        new_q = 'A pizza has 30 slices, 6 remain, and each person ate 3. How many people ate?'
        self.assertEqual(run_program(p, question_numbers(new_q)), Fraction(8))

    def test_constant_answer_rejected(self):
        self.assertIsNone(extract_program('There are 4 apples.', '#### 4'))
        self.assertIsNone(extract_program('There are 4 apples.', '<<123+1=124>> #### 124'))

    def test_arity_and_zero_division(self):
        p = ['arity', 2, ['/', ['v', 0], ['v', 1]]]
        with self.assertRaises(ValueError): run_program(p, [Fraction(2)])
        with self.assertRaises(ZeroDivisionError): run_program(p, [Fraction(2), Fraction(0)])

    def test_checkpoint_roundtrip(self):
        rows = [{'question': 'There are 16 slices and 4 slices left.'},
                {'question': 'There are 30 slices and 6 slices left.'}]
        p = ['arity', 2, ['-', ['v', 0], ['v', 1]]]
        vec, matrix, programs = fit_retriever(rows, [p, p])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'programs.npz'
            save_retriever(path, vec, matrix, programs, .75)
            v2, m2, p2, t = load_retriever(path)
            a, c = suggested_answers(rows, vec, matrix, programs)
            b, d = suggested_answers(rows, v2, m2, p2)
            self.assertEqual(a, b)
            np.testing.assert_allclose(c, d)
            self.assertEqual(t, .75)


if __name__ == '__main__':
    unittest.main()
