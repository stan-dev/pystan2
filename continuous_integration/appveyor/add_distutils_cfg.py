import distutils
import os

# read distutils location
path = distutils.__path__[0]

cfg_path = os.path.join(path, "distutils.cfg")

cfg_string = "[build] \ncompiler=mingw32 \n"

# create distutils.cfg file
with open(path, mode='w') as f:
    f.write(cfg_string)
