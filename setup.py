import os

from distutils.core import setup


descr = 'Implicit regularization for convex problems'

version = None
with open(os.path.join('iterreg', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

DISTNAME = 'iterreg'
DESCRIPTION = descr
MAINTAINER = 'Mathurin Massias'
MAINTAINER_EMAIL = 'mathurin.massias@gmail.com'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/LCSL/iterreg.git'
VERSION = version

setup(name='iterreg',
      version=VERSION,
      description=DESCRIPTION,
      license=LICENSE,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      install_requires=["celer", "matplotlib", "numpy", "scipy"],
      download_url=DOWNLOAD_URL,
      packages=['iterreg'],
      )
