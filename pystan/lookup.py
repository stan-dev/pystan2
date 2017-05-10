#-----------------------------------------------------------------------------
# Copyright (c) 2017, PyStan developers
#
# This file is licensed under Version 3.0 of the GNU General Public
# License. See LICENSE for a text of the license.
#-----------------------------------------------------------------------------

import numpy as np
import re
import os

lookuptable = None
stancsv = None

def lookup(name):
  if lookuptable is None:
    build()
    lookuptable[name]
  if name not in lookuptable.keys():
    raise ValueError(name + " not avaible in lookup table")
    return
  entries = stancsv[lookuptable[name]]
  if not len(entries):
    return "Found no equivalent Stan function available for " + name

  try:
    import pandas as pd
  except ImportError:
    return entries

  return pd.DataFrame(entries)



def build():
  global lookuptable
  global stancsv
  stanfunctions_file = "lookuptable/stan-functions.txt"
  rfunctions_file = "lookuptable/R.txt"
  lookuptb_file = "lookuptable/python.txt"

  dir = os.path.dirname(__file__)
  stanfunctions_file = os.path.join(dir, stanfunctions_file)
  rfunctions_file = os.path.join(dir, rfunctions_file)
  lookuptb_file = os.path.join(dir, lookuptb_file)

  stancsv = np.genfromtxt(stanfunctions_file, delimiter=';',
                          names=True, skip_header=True,
                          dtype=['<U153','<U153','<U153' ,"int"])

  StanFunction = stancsv["StanFunction"]

  #Auto-extract R functions
  rmatches = [re.findall(r'((?<=RFunction\[StanFunction == \").+?(?=\")'
                          '|(?<=grepl\(").+?(?=", StanFunction\))'
                          '|(?<= \<\- ").+?(?="\)))'
                          '|NA\_character\_', l)
                   for l in open(rfunctions_file)]
  tomatch = list(filter(lambda x: len(x) == 2, rmatches))
  tomatch = np.array(tomatch, dtype=str)
  tomatch[:, 1] = np.vectorize(lambda x: "R." + x)(tomatch[:,1])

  #Get packages lookup table for Python packages
  pymatches = np.genfromtxt(lookuptb_file, delimiter=';', dtype=str)
  tomatch = np.vstack((tomatch, pymatches))

  lookuptb = dict()
  for i in range(tomatch.shape[0]):
    matchedlines = np.vectorize(lambda x: re.match(tomatch[i, 0],
                                                       x))(StanFunction)
    lookuptb[tomatch[i, 1]] = np.where(matchedlines)[0]

  #debug: list of rmatches that got wrong
  print(list(filter(lambda x: len(x) != 2 and len(x) != 0, rmatches)))

  #debug: list of nodes without matches on lookup table
  for k in lookuptb:
    if len(lookuptb[k]) == 0:
      print(k)

  lookuptable = lookuptb
