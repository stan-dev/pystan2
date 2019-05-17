import unittest
import numpy as np
from numpy.testing import assert_array_equal
import os
import pystan

# NOTE: This test is fragile because the default sampling algorithm used by Stan
# may change in significant ways.

external_cpp_code = """
inline double my_func (double A, double B, std::ostream* pstream) {
  double C = A*B*B+A;

  return C;
}

inline var my_func (const var& A_var, const var& B_var, std::ostream* pstream) {
  // Compute the value
  double A = A_var.val(),
         B = B_var.val(),
         C = my_func(A, B, pstream);

  // Compute the partial derivatives:
  double dC_dA = B*B+1.0,
         dC_dB = 2.0*A*B;

  // Autodiff wrapper:
  return var(new precomp_vv_vari(
    C,             // The _value_ of the output
    A_var.vi_,     // The input gradient wrt A
    B_var.vi_,     // The input gradient wrt B
    dC_dA,         // The partial introduced by this function wrt A
    dC_dB          // The partial introduced by this function wrt B
  ));
}

inline var my_func (double A, const var& B_var, std::ostream* pstream) {
  double B = B_var.val(),
         C = my_func(A, B, pstream),
         dC_dB = 2.0*A*B;
  return var(new precomp_v_vari(C, B_var.vi_, dC_dB));
}

inline var my_func (const var& A_var, double B, std::ostream* pstream) {
  double A = A_var.val(),
         C = my_func(A, B, pstream),
         dC_dA = B*B+1.0;
  return var(new precomp_v_vari(C, A_var.vi_, dC_dA));
}
"""

external_cpp_code_stan_autograd = """
template <typename T1, typename T2>
typename boost::math::tools::promote_args<T1, T2>::type
my_other_func (const T1& A, const T2& B, std::ostream* pstream) {
  typedef typename boost::math::tools::promote_args<T1, T2>::type T;

  T C = A*B*B+A;

  return C;
}
"""

class TestExternalCpp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        include_dir = os.path.join(os.path.dirname(__file__), 'data')
        with open(os.path.join(include_dir, "external_cpp.hpp"), "w") as f:
            f.write(external_cpp_code)
        with open(os.path.join(include_dir, "external_autograd_cpp.hpp"), "w") as f:
            f.write(external_cpp_code_stan_autograd)
        model_code = """
        functions {
            real my_func(real A, real B);
            real my_other_func(real A, real B);
        }
        data {
            real B;
        }
        parameters {
            real A;
            real D;
        }
        transformed parameters {
            real C = my_func(A, B);
            real E = my_other_func(D, B);
        }
        model {
            C ~ std_normal();
            E ~ normal(5,3);
        }
        """
        cls.model = pystan.StanModel(model_code=model_code,
                                     verbose=True,
                                     allow_undefined=True,
                                     includes=["external_cpp.hpp",
                                               "external_autograd_cpp.hpp"],
                                     include_dirs=[include_dir],
                                     )
        cls.B = 4.0
        cls.fit = cls.model.sampling(data={"B" : cls.B}, iter=100, chains=2)
        os.remove(os.path.join(include_dir, "external_cpp.hpp"))
        os.remove(os.path.join(include_dir, "external_autograd_cpp.hpp"))

    def test_external_cpp(self):
        A = self.fit["A"]
        B = self.B
        C = self.fit["C"]
        assert_array_equal(C, A * B * B + A)

    def test_external_autograd_cpp(self):
        B = self.B
        D = self.fit["D"]
        E = self.fit["E"]
        assert_array_equal(E, D * B * B + D)
