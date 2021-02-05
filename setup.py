from glob import glob
import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='stackimpute',
      version='0.0.1-beta',
      description='stacking machine learning model for CpG methylation imputation',
      # long_description=read('README.rst'),
      author='Yi Liu',
      author_email='yiliu11@zju.edu.cn',
      license="MIT",
      url='https://github.com/sixone11/stackimpute',
      packages=find_packages(),
      # data_files=[
      #  ('stackimpute', ['data/*.out'])],
      package_data={'stackimpute': ['data/*.tsv']},
      include_package_data=True,
      scripts=glob('./scripts/*.py'),
      install_requires=['shap',
                        'argparse',
                        'scikit-learn',
                        'scipy',
                        'pandas',
                        'numpy',
                        'xgboost'],
      keywords=['machine learning',
                'DNA methylation',
                'imputation',
                'stacking model'],
      classifiers=['Development Status :: 3 - Alpha',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: MIT License',
                   'Natural Language :: English',
                   'Programming Language :: Python :: 3.8',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   ]
      )
