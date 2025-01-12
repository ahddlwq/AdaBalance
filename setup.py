from setuptools import setup

__VERSION__ = '0.0.5'

setup(name='adabalance',
      version=__VERSION__,
      description='AdaBalance optimization algorithm, build on PyTorch.',
      long_description=open("README.md").read(),
      long_description_content_type="text/markdown",
      keywords=['machine learning', 'deep learning'],
      classifiers=[
          'Intended Audience :: Science/Research',
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      url='https://github.com/ahddlwq/AdaBalance',
      author='Wuqi Liang',
      author_email='liangwuqi@ahou.edu.cn',
      license='Apache',
      packages=['adabalance'],
      install_requires=[
          'torch>=0.4.0',
      ],
      zip_safe=False,
      python_requires='>=3.6.0')
