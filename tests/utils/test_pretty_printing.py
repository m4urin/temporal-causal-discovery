import unittest

from src.utils.pretty_printing import *


class ShortScientificNotationTestCase(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(short_scientific_notation(0), '0')

    def test_positive_number(self):
        self.assertEqual(short_scientific_notation(123456789, decimals=2), '1.23e8')

    def test_small_positive_number(self):
        self.assertEqual(short_scientific_notation(0.000000000123451111, decimals=4), '1.2345e-10')

    def test_negative_number(self):
        self.assertEqual(short_scientific_notation(-123411111, decimals=3), '-1.234e8')


class ShortNumberTestCase(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(short_number(0), '0')

    def test_small_positive_number(self):
        self.assertEqual(short_number(0.000000000123456789, significant_numbers=4), '1.23e-10')

    def test_small_positive_number_decimal(self):
        self.assertEqual(short_number(0.00123456, significant_numbers=3), '0.00123')

    def test_small_positive_number_decimal_long(self):
        self.assertEqual(short_number(0.000123456, significant_numbers=5), '1.23e-4')

    def test_large_positive_number(self):
        self.assertEqual(short_number(123456789, significant_numbers=3), '1.23e8')

    def test_large_positive_number_decimal(self):
        self.assertEqual(short_number(123.456, significant_numbers=3), '123')

    def test_negative_number(self):
        self.assertEqual(short_number(-123456789, significant_numbers=3), '-1.23e8')

    def test_positive_number_sig_digits(self):
        self.assertEqual(short_number(123456, significant_numbers=5), '123456')

    def test_positive_number_sig_digits_long(self):
        self.assertEqual(short_number(123456, significant_numbers=10), '123456')

    def test_decimal_number_sig_digits(self):
        self.assertEqual(short_number(1234.56, significant_numbers=5), '1234.6')

    def test_default_string_representation(self):
        self.assertEqual(short_number(123456789), '1.23e8')


if __name__ == '__main__':
    unittest.main()
