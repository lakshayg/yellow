from setuptools import setup

setup(name = 'yellow',
      version = 0.1,
      description = 'Support routines for efficient kernel methods',
      url = 'https://github.com/lakshayg/yellow',
      author = 'Lakshay Garg',
      author_email = 'lakshayg.iitk@gmail.com',
      license = 'MIT',
      package_dir = {
          'yellow':             'yellow',
          'yellow.feature_map': 'yellow/feature_map',
          'yellow.online':      'yellow/online'
      },
      packages = ['yellow', 'yellow.feature_map', 'yellow.online'],
      zip_safe = False,
      install_requires = [
          'numpy', 'sklearn'
      ]
)

