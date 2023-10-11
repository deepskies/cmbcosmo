from distutils.core import setup

# basics to allow doing a pip install of the code
setup(name='cmbcosmo',
      version='0.1',
      description='cmbcosmo',
      packages=['cmbcosmo'],
      package_dir={'cmbcosmo': 'cmbcosmo'},
     )
