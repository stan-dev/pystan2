How to use PyStan in Atom editor?
====================================

This is a short doc explaining how PyStan can be run in `Atom <http://atom.io>`_ on a Windows machine. 

PyStan doesn't automatically run in `Atom <http://atom.io>`_ even if we use script to pick the correct python.exe file from the environment we have installed Stan in. The reason is that Windows is very sensitive to the activation of an environment. If the environment is not explicitly activated before running the PyStan code in Atom compiling errors will occur.

Steps
------
- Install PyStan as explained `here <http://pystan.readthedocs.io/en/latest/windows.html#windows>`_ with Anaconda. (I tried to get it installed without Anaconda first and there were a lot of compiling errors.)
- Install `Atom <http://atom.io>`_.
- Launch `Atom <http://atom.io>`_ in your activated Stan environment as follows:
  
  1. Open a compand prompt by typing ``cmd`` when you click on Start.
      
      a) First time only: Initialise anaconda on the command prompt by typing ``conda init``. This will allow you to run conda commands with command prompt as you would with Anaconda Prompt.

  2. Activate your Stan environment with ``activate myenv``, where ``myenv`` is the name of your environment such as ``stan_env``.
  3. Type ``atom`` to launch the Atom (once launched you can close the cmd window if you like).

Test
------
To check if you can now run PyStan in Atom, copy and run the example code from the `installation doc <http://pystan.readthedocs.io/en/latest/windows.html#windows>`_.

>>> import pystan
>>> model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
>>> model = pystan.StanModel(model_code=model_code)
>>> y = model.sampling().extract()['y']
>>> y.mean()  # with luck the result will be near 0

In my case it looked like it was working but after a while it just repeated itself and ran for quite a while. Have a look at `this comment <https://github.com/stan-dev/pystan/issues/520#issuecomment-426970215>`_ which shows the output I got. 

To remedy this we will need to add ``if __name__ == "__main__":`` to our program which will look like the below. Jupyter does this by default and the line isn't necessary.

.. code-block:: python

  import pystan
  model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
  if __name__ == "__main__":
    model = pystan.StanModel(model_code=model_code)
    y = model.sampling().extract()['y']
    print(y.mean()) # with luck the result will be near 0
	
Run your code with F5 and it should work.
