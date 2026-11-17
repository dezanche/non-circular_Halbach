"""
This program illustrates how conformal mapping can be used to design Halbach magnet arrays on various shapes.

    Copyright (C) 2026  Nicola De Zanche

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np

def rectangle(theta , p):
    # p is parameter that sets aspect ratio of rectangle (0.5 --> square)
    from scipy.special import ellipk, ellipe
    import mpmath as mp
    offset = 0.01  # weird results if exactly on the unit circle (offset = 0)

    # Complete elliptic integrals K' and E'
    Kp = ellipk(1 - p)
    Ep = ellipe(1 - p)
    K = ellipk(p)
    Ex = ellipe(p)

    # dimensions of rectangle (eq. 1.13.13 Ivanov, Trubetskov)
    rect_Y = (Ep-p*Kp)/np.sqrt(p)
    rect_X = (Ex-(1-p)*K)/np.sqrt(p)

    # Dipole locations in model plane
    w = (1 + offset) * np.exp(1j * (theta + np.pi / 1000))

    # Intermediate coordinate
    t = (w + 1 / w) / 2

    phi = np.arcsin(t)
    v_exp = np.vectorize(mp.ellipe)

    # Compute transformation to physical plane (Ivanov, Trubetskov Eq. 1.13.15 p.219)
    E = np.array(v_exp(phi,1/p), dtype = complex)
    Z0 = E + 1j * (Ep - p * Kp) / np.sqrt(p) * np.sign(np.imag(t))
    
    return Z0, rect_X, rect_Y


def Zhukovskii(theta, a, b):
    # transforms unit circle to ellipse
    # a, b are minor and major radii of the ellipse
    d = (b + a) / 2
    c = (b**2 - a**2) / 4
    # Dipole locations (Leifer mapping)
    exp_term = np.exp(1j * theta)
    Z0 = d * exp_term + c / (d * exp_term)
    return Z0

def Laurent(theta, coeffs):
    # Laurent series with constant term = 0
    z = np.exp(1j * theta)
    result = z * coeffs[0]  # linear term
    for k, c in enumerate(coeffs[1:]):
        # print(k, c)
        result = result + c / (z ** (k + 1))   # using += syntax gives wrong results
    return result

def Cassini(theta, c, d):
    # Cassinian oval 
    # c = 1/2 distance between foci, 
    # and d = radius of circle in model plane
    w = d * np.exp(1j * theta)
    Z0 = w*np.sqrt(1+(c**2)/w**2)
    return Z0

def circle(theta, r):
    # just a circle
    Z0 = r * np.exp(1j * theta)
    return Z0

def cardioid(theta, d):
    # cardioid
    w = d * np.exp(1j * theta)
    s = np.acos((w-1)/(w+1))/np.pi
    Z0 = (s**-2-1)
    return Z0
