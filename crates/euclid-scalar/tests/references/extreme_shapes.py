#!/usr/bin/env python3
"""Generate the extreme-shape regression fixtures using MPFR and Decimal."""

import ctypes as C
from ctypes.util import find_library
from decimal import Decimal, localcontext
from math import factorial, inf, nextafter
import struct


class Mpfr(C.Structure):
    _fields_ = [
        ("precision", C.c_long),
        ("sign", C.c_int),
        ("exponent", C.c_long),
        ("limbs", C.c_void_p),
    ]


class References:
    def __init__(self):
        library = find_library("mpfr")
        if library is None:
            raise RuntimeError("MPFR is required to regenerate extreme-shape fixtures")
        self.lib = C.CDLL(library)
        ptr = C.POINTER(Mpfr)
        signatures = {
            "init2": ([ptr, C.c_long], None),
            "clear": ([ptr], None),
            "set_d": ([ptr, C.c_double, C.c_int], C.c_int),
            "get_d": ([ptr, C.c_int], C.c_double),
        }
        signatures["cmp"] = ([ptr, ptr], C.c_int)
        for name in ("gamma", "lngamma", "erfc", "eint", "zeta"):
            signatures[name] = ([ptr, ptr, C.c_int], C.c_int)
        for name in ("gamma_inc", "div", "add", "sub"):
            signatures[name] = ([ptr, ptr, ptr, C.c_int], C.c_int)
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, "mpfr_" + name)
            function.argtypes = arguments
            function.restype = result
        self.values = [Mpfr() for _ in range(5)]
        for value in self.values:
            self.lib.mpfr_init2(C.byref(value), 256)
        self.a, self.b, self.x, self.y, self.z = (C.byref(v) for v in self.values)

    def close(self):
        for value in self.values:
            self.lib.mpfr_clear(C.byref(value))

    def gamma_upper(self, shape, point):
        lib = self.lib
        lib.mpfr_set_d(self.a, shape, 0)
        lib.mpfr_set_d(self.x, point, 0)
        lib.mpfr_gamma_inc(self.y, self.a, self.x, 0)
        lib.mpfr_gamma(self.z, self.a, 0)
        lib.mpfr_div(self.y, self.y, self.z, 0)
        return lib.mpfr_get_d(self.y, 0)

    def complete_gamma(self, point):
        lib = self.lib
        lib.mpfr_set_d(self.x, point, 0)
        lib.mpfr_gamma(self.y, self.x, 0)
        return lib.mpfr_get_d(self.y, 0)

    def zeta(self, point):
        lib = self.lib
        lib.mpfr_set_d(self.x, point, 0)
        lib.mpfr_zeta(self.y, self.x, 0)
        return lib.mpfr_get_d(self.y, 0)

    def exp_integral(self, point):
        lib = self.lib
        lib.mpfr_set_d(self.x, point, 0)
        lib.mpfr_eint(self.y, self.x, 0)
        return lib.mpfr_get_d(self.y, 0)

    def log_beta(self, a, b):
        lib = self.lib
        lib.mpfr_set_d(self.a, a, 0)
        lib.mpfr_set_d(self.b, b, 0)
        lib.mpfr_add(self.x, self.a, self.b, 0)
        lib.mpfr_lngamma(self.a, self.a, 0)
        lib.mpfr_lngamma(self.b, self.b, 0)
        lib.mpfr_lngamma(self.x, self.x, 0)
        lib.mpfr_add(self.y, self.a, self.b, 0)
        lib.mpfr_sub(self.y, self.y, self.x, 0)
        return lib.mpfr_get_d(self.y, 0)

    def log_gamma1p_over_shape(self, shape):
        lib = self.lib
        lib.mpfr_set_d(self.a, shape, 0)
        lib.mpfr_set_d(self.x, 1.0, 0)
        lib.mpfr_add(self.x, self.a, self.x, 0)
        lib.mpfr_lngamma(self.x, self.x, 0)
        lib.mpfr_div(self.x, self.x, self.a, 0)
        return lib.mpfr_get_d(self.x, 0)

    def erfc_inverse(self, probability):
        lib = self.lib
        lib.mpfr_set_d(self.y, probability, 0)
        low, high = 0.0, 32.0
        for _ in range(70):
            mid = low + (high - low) / 2
            lib.mpfr_set_d(self.x, mid, 0)
            lib.mpfr_erfc(self.z, self.x, 0)
            if lib.mpfr_cmp(self.z, self.y) > 0:
                low = mid
            else:
                high = mid
        return low + (high - low) / 2


def integer_beta(a, b, x):
    """Finite sum for the beta complement at a positive integer first shape."""
    with localcontext() as ctx:
        ctx.prec = 450
        b, x = Decimal(b), Decimal(x)
        rising = Decimal(1)
        total = rising
        for k in range(1, a):
            rising *= b + k - 1
            total += rising * x**k / factorial(k)
        return float(1 - (b * (1 - x).ln()).exp() * total)


def log_expm1_references():
    """Rounding checks around the zero of log(exp(x) - 1)."""
    with localcontext() as ctx:
        ctx.prec = 500
        for fmt, bits_fmt, center in [("!d", "!Q", 0x3FE62E42FEFA39EF),
                                      ("!f", "!I", 0x3F317218)]:
            high = struct.unpack(fmt, struct.pack(bits_fmt, center))[0]
            print("ln 2 low", fmt, Decimal(2).ln() - Decimal(high))
            for bits in range(center - 1, center + 2):
                x = struct.unpack(fmt, struct.pack(bits_fmt, bits))[0]
                exact = (Decimal(x).exp() - 1).ln()
                encoded = struct.pack(fmt, float(exact))
                rounded = struct.unpack(fmt, encoded)[0]
                result_bits = struct.unpack(bits_fmt, encoded)[0]
                print("ln(exp(x) - 1)", fmt, hex(bits), rounded, hex(result_bits))


def main():
    refs = References()
    try:
        assert refs.gamma_upper(1.0, 1.0) == float(Decimal(-1).exp())
        assert refs.log_beta(2.0, 3.0) == -float(Decimal(12).ln())
        assert integer_beta(2, 3.0, 0.5) == 0.6875
        for a, x in [(1e-320, 0.5), (1e-320, 1.0), (1e-18, 0.5), (1e-12, 1.0),
                     (1e-3, 0.125), (1e-3, 1.0), (1e-3, 2.0)]:
            print("gamma Q", a, x, refs.gamma_upper(a, x))
        for a, b in [(1e8, 1e8), (1e16, 1e16), (1e306, 1e306),
                     (2e307, 1.7e308), (1e300, 1.7e308)]:
            print("log beta", a, b, refs.log_beta(a, b))
        for a, b, scale in [(2, 1.1e308, 1), (5, 1.1e308, 1), (10, 1e25, 1),
                            (5, 1e18, 2), (5, 1e9, 2), (2, 1e9, 2)]:
            x = scale * a / b
            print("beta CDF", a, b, x, integer_beta(a, b, x))
        reflected_x = 0.99999999
        print("beta CDF", 1e9, 5, reflected_x,
              1.0 - integer_beta(5, 1e9, 1.0 - reflected_x))
        for y in [float.fromhex("0x0.0000000000001p-1022"),
                  float.fromhex("0x0.0000000000003p-1022")]:
            print("erfc inverse", y, refs.erfc_inverse(y))
        print("ln Γ(1+a)/a", 1e-3, refs.log_gamma1p_over_shape(1e-3))
        for x in (700.0, 710.0, 716.0, 716.3554905424517, 716.3554905424518):
            print("Ei", x, refs.exp_integral(x))
        log_expm1_references()
        for zero in (-2.0, -20.0, -168.0):
            for direction in (-inf, inf):
                x = nextafter(zero, direction)
                print("zeta", x, refs.zeta(x))
        for x in (-172.5, -174.5, -176.5, -177.5, -178.5, -1500.5, -1501.5,
                  -2.0000000000000004, -1.9999999999999998, -171.00000000000003):
            value = refs.complete_gamma(x)
            print("gamma f64", x, value, hex(struct.unpack("!Q", struct.pack("!d", value))[0]))
        for x in (-34.25, -37.5, -39.5, -40.5):
            encoded = struct.pack("!f", refs.complete_gamma(x))
            print("gamma f32", x, hex(struct.unpack("!I", encoded)[0]))
        for a in (0.009290680585958758, 0.01):
            shape = struct.unpack("!f", struct.pack("!f", a))[0]
            lo, hi = 0.0, 1e-29
            for _ in range(100):
                mid = lo + (hi - lo) / 2
                if refs.gamma_upper(shape, mid) > 0.5:
                    lo = mid
                else:
                    hi = mid
            print("gamma median at f32 shape", shape, lo + (hi - lo) / 2)
    finally:
        refs.close()


if __name__ == "__main__":
    main()
