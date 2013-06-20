# distutils: language = c++
#-----------------------------------------------------------------------------
# Copyright (c) 2013, Allen B. Riddell
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------
from pystan.stanc cimport PyStancResult, stanc as c_stanc

def stanc(bytes model_stancode, bytes model_name):
    cdef PyStancResult result
    c_stanc(model_stancode, model_name, result)
    return {'status': result.status,
            'msg': result.msg.decode('ascii'),
            'model_cppname': result.model_cppname.decode('ascii'),
            'cppcode': result.cppcode.decode('ascii')}
