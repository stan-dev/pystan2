.. _external_cpp:

.. currentmodule:: pystan

======================================
 External C++ (experimental)
======================================

This feature is experimental and nothing is guaranteed.

Instructions follow loosely https://dfm.io/posts/stan-c++/ which
gives more in-depth example using external C++.

Setting up C++
==============

Your C++ code needs to provide needed gradients and partial derivatives to work
properly. This can be done by manually implementing each needed component
or by using Stan's autodiff (special syntax). Each function also needs to
accept a ``std::ostream`` as a last argument.

In the following instructions it is assumed that the first code is saved to
a file called ``external_manual.hpp`` and ``external_autograd.hpp``. Both files
are saved under ``cwd`` in a directory ``external_cpp``.

The following code shows a minimal example with manually implemented gradients
for a function ``C = A*B*B+A``:

.. code-block:: c++

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

The minimal example using Stan´s autograd for a function ``C = A*B*B+A``:

.. code-block:: c++

        template <typename T1, typename T2>
        typename boost::math::tools::promote_args<T1, T2>::type
        my_other_func (const T1& A, const T2& B, std::ostream* pstream) {
          typedef typename boost::math::tools::promote_args<T1, T2>::type T;

          T C = A*B*B+A;

          return C;
        }

Setting up Stan
===============

User needs to define functions inside a ``functions`` block. Function names
are defined in the C++ code.

.. code-block:: stan

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
            E ~ std_normal();
        }

Setting up Python
=================
In Python user needs information of location for external C++ code. This
information is passed to ``StanModel`` with two ``list``-objects
``include_dir`` and ``includes``. Also ``allow_undefined`` keyword
is set to ``True``.

.. code-block:: stan

    import pystan
    import os

    model_code = """...
                 """

    include_dirs = [os.path.join(".", "external_cpp")]
    include_files = ["external_manual.hpp", "external_autograd.hpp"]
    stan_model = pystan.StanModel(model_code=model_code,
                                  verbose=True,
                                  allow_undefined=True,
                                  includes=include_files,
                                  include_dirs=include_dirs,
                                 )
    stan_data = {"B" : 0.1}
    fit = stan_model.sampling(data=stan_data)
    print(fit)

Compilation with external C++ can take longer than normally. The external C++
code is injected to the ``stanc`` translated C++ code. This injections is done
automatically by PyStan before model compilation.
