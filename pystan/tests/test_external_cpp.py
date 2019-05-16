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

class TestExternalCpp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        with open("data/external_cpp.hpp", "w") as f:
            f.write(external_cpp_code)
        model_code = """
        functions {
            real my_func(real A, real B);
        }
        data {
            real B;
        }
        parameters {
            real A;
        }
        transformed parameters {
            real C = my_func(A, B);
        }
        model {
            C ~ std_normal();
        }
        """
        include_dir = os.path.join(os.path.dirname(__file__), 'data')
        cls.model = pystan.StanModel(model_code=model_code,
                                     verbose=True,
                                     allow_undefined=True,
                                     includes=["external_cpp.hpp"],
                                     include_dirs=[include_dir],
                                     )
        #model = pystan.StanModel(model_code=model_code)
        cls.B = 4.0
        cls.fit = cls.model.sampling(data={"B" : cls.B}, iter=100, chains=2)
        os.remove("data/external_cpp.hpp")

    def test_external_cpp(self):
        A = self.fit["A"]
        B = self.B
        C = self.fit["C"]
        assert_array_equal(C, A * B * B + A)
