from setuptools import setup, find_packages
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'simreal'))

VERSION = '1.0.0'

# Environment-specific dependencies.
extras = {
}

# Meta dependency groups.
extras['all'] = [item for group in extras.values() for item in group]

setup(name='simreal',
      version=VERSION,
      description='Adapted toolkit for developing and comparing your reinforcement learning agents.',
      url='',
      author='Stijn Woestenborghs',
      author_email='stijn.woestenborghs@live.be',
      license='',
      packages=[],
      zip_safe=False,
      install_requires=[
          'scipy', 
          'numpy>=1.10.4', 
          'pyglet>=1.4.0,<=1.5.0', 
          'cloudpickle>=1.2.0,<1.7.0', 
          'colorama', 
          'six==1.16.0',
          'torch==1.11.0',
          'PyYAML==6.0',
          'pandas==1.3.5',
          'matplotlib==3.5.2',
          'scikit-learn==1.0.2',
          'gym==0.18.0'
      ],
      extras_require=extras,
      package_data={'simreal': [
        # 'envs/mujoco/assets/*.xml',
        'envs/classic_control/assets/*.png']
      },
      tests_require=['pytest', 'mock'],
      python_requires='>=3.5',
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
)