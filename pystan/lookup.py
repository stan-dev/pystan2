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
stanftable = None

def lookup(name, similarity_ratio=.75):
  """
  Look up for a Stan function with similar functionality to a Python
  function or an R function. If the function is not present on the
  lookup table, then attempts to find similar one and prints the
  results. For better display quality of the results, pandas library
  should be installed (although it will work without it).

  Parameters
  ----------
  name : str
    Function name to look for.
  similarity_ratio : float
    Similarity .

  Examples
  --------
  #Look up for a Stan function similar to scipy.stats.skewnorm
  lookup("scipy.stats.skewnorm")
  #Look up for a Stan function similar to R dnorm
  lookup("R.dnorm")
  #Look up for a Stan function similar to numpy.hstack
  lookup("numpy.hstack")
  #List Stan log probability mass functions
  lookup("lpmfs")
  #List Stan log cumulative density functions
  lookup("lcdfs")
  """
  if lookuptable is None:
    build()
  if name not in lookuptable.keys():
    from difflib import SequenceMatcher
    from operator import itemgetter
    print(name + " not avaible in lookup table")

    lkt_keys = list(lookuptable.keys())
    mapfunction = lambda x: SequenceMatcher(a=name, b=x).ratio()
    similars = list(map(mapfunction, lkt_keys))
    similars = zip(range(len(similars)), similars)
    similars = list(filter(lambda x: x[1] > similarity_ratio, similars))
    similars = sorted(similars, key=itemgetter(1))

    if (len(similars)):
      print("But the following similar entries were found: ")
      for i in range(len(similars)):
        print(lkt_keys[similars[i][0]] + " ===> with similary ratio of "
              "" + str(round(similars[i][1], 3)) + "")
      print("For the most similar one we have:")
      return lookup(lkt_keys[similars[i][0]])
    else:
      print("And no similar entry found. You may try to raise the"
            "parameter similarity_ratio")
    return
  entries = stanftable[lookuptable[name]]
  if not len(entries):
    return "Found no equivalent Stan function available for " + name

  try:
    import pandas as pd
  except ImportError:
    return entries

  return pd.DataFrame(entries)



def build():
  stanfunctions_file = "lookuptable/stan-functions.txt"
  rfunctions_file = "lookuptable/R.txt"
  pythontb_file = "lookuptable/python.txt"
  scipytb_file = "lookuptable/scipy.stats.txt"

  dir = os.path.dirname(__file__)
  stanfunctions_file = os.path.join(dir, stanfunctions_file)
  rfunctions_file = os.path.join(dir, rfunctions_file)
  pythontb_file = os.path.join(dir, pythontb_file)
  scipytb_file = os.path.join(dir, scipytb_file)

  stanftb = np.genfromtxt(stanfunctions_file, delimiter=';',
                          names=True, skip_header=True,
                          dtype=['<U200','<U200','<U200' ,"int"])
  rpl_textbar = np.vectorize(lambda x: x.replace("\\textbar \\", "|"))
  stanftb['Arguments'] = rpl_textbar(stanftb['Arguments'])

  StanFunction = stanftb["StanFunction"]

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
  pymatches = np.genfromtxt(pythontb_file, delimiter='; ', dtype=str)
  tomatch = np.vstack((tomatch, pymatches))

  lookuptb = dict()
  for i in range(tomatch.shape[0]):
    matchedlines = np.vectorize(lambda x: re.match(tomatch[i, 0],
                                                       x))(StanFunction)
    lookuptb[tomatch[i, 1]] = np.where(matchedlines)[0]

  #debug: list of rmatches that got wrong
  #print(list(filter(lambda x: len(x) != 2 and len(x) != 0, rmatches)))

  #debug: list of nodes without matches on lookup table
  #for k in lookuptb:
  #  if len(lookuptb[k]) == 0:
  #    print(k)
  global lookuptable
  global stanftable

  stanftable = stanftb
  lookuptable = lookuptb
