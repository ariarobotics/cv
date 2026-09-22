"""Representative local checks. Run: python check_hw04.py
These are practice checks, not the entire Gradescope test suite.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
import hw04 as hw

HERE = Path(__file__).resolve().parent


class BasicChecks(unittest.TestCase):
    def check_array(self, actual, expected, atol=2e-5):
        self.assertIsInstance(actual,np.ndarray)
        self.assertEqual(actual.dtype,np.float32)
        self.assertEqual(actual.shape,expected.shape)
        np.testing.assert_allclose(actual,expected,atol=atol,rtol=1e-5)

    def test_constant_spectrum(self):
        image=np.full((6,8),.5,np.float32)
        wanted=np.zeros_like(image); wanted[3,4]=np.log1p(24)
        self.check_array(hw.log_magnitude_spectrum(image),wanted)

    def test_impulse_spectrum(self):
        image=np.zeros((7,9),np.float32); image[2,3]=1
        self.check_array(hw.log_magnitude_spectrum(image),np.full_like(image,np.log(2)))

    def test_mask_properties(self):
        mask=hw.gaussian_frequency_mask((6,8),2.0)
        self.assertEqual(mask.shape,(6,8)); self.assertEqual(mask.dtype,np.float32)
        self.assertAlmostEqual(float(mask[3,4]),1,places=6)
        # One horizontal frequency bin is 1/8 cycle per pixel.
        self.assertAlmostEqual(float(mask[3,5]),0.29121293,places=6)
        self.assertTrue(np.all((mask>=0)&(mask<=1)))

    def test_all_pass_and_zero_pass(self):
        image=np.arange(63,dtype=np.float32).reshape(7,9)/62
        original=image.copy()
        self.check_array(hw.apply_frequency_filter(image,np.ones_like(image)),image)
        self.check_array(hw.apply_frequency_filter(image,np.zeros_like(image)),np.zeros_like(image))
        np.testing.assert_array_equal(image,original)

    def test_constant_hybrid(self):
        a=np.full((8,10),.6,np.float32); b=np.full_like(a,.2)
        low,high,hybrid=hw.make_hybrid_image(a,b,3,2)
        self.check_array(low,a)
        self.check_array(high,np.zeros_like(b))
        self.check_array(hybrid,a/2)

    def test_saved_reference_pixels(self):
        with tempfile.TemporaryDirectory() as output:
            subprocess.run([sys.executable,str(HERE/'hw04.py'),str(HERE/'einstein.png'),
                            str(HERE/'marilyn.png'),'--output-dir',output],check=True)
            actual=cv2.imread(str(Path(output)/'hw04_hybrid.png'),cv2.IMREAD_GRAYSCALE)
            expected=cv2.imread(str(HERE/'reference_hybrid.png'),cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(actual); self.assertIsNotNone(expected)
            self.assertEqual(actual.shape,expected.shape)
            self.assertLessEqual(int(np.abs(actual.astype(int)-expected.astype(int)).max()),1)
            for name in ('hw04_output.png','hw04_distance.png'):
                self.assertIsNotNone(cv2.imread(str(Path(output)/name)))


if __name__=='__main__':
    unittest.main(verbosity=2)
