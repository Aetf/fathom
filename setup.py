from setuptools import find_packages, setup

# The hyphens in release candidates (RCs) will automatically be normalized.
# But we normalize below manually anyway.
_VERSION = '1.0-rc0'


setup(name='Fathom-Workloads',  # "fathom" is already taken on PyPI
      description='Reference workloads for modern deep learning',
      url='http://github.com/rdadolf/fathom',

      version=_VERSION.replace('-', ''),
      # TODO: Add version numbers.
      install_requires=[
          'tensorflow>=1.0.0',
          'numpy',
          'scipy',
          'scikit-learn',
          'librosa>=0.6.0',  # audio preprocessing
          'h5py',
          'future',  # python 2 & 3 compatibility
          'requests'  # dataset downloading
          'tqdm'  # TIMIT dataset processing
      ],

      # Authors: Robert Adolf, Saketh Rama, and Brandon Reagen
      # PyPI does not have an easy way to specify multiple authors.
      author="Saketh Rama",
      author_email="rama@seas.harvard.edu",

      # We don't use __file__, but mark False to be safe.
      zip_safe=False,

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 2.7',

          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Image Recognition',
          'Topic :: System :: Hardware',
      ],

      packages=find_packages(),  # find packages in subdirectories

      package_data={'fathom': [
          'fathom.png',

          'Dockerfile',
          'pylintrc',

          'README.md',
          'mkdocs.yml',

          'runtest.sh',

          'setup.cfg',

          'data/*'
      ]},
      include_package_data=True,
      )
