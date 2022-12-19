
from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
   content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='corpy',
      version="0.0.1",
      description="Corpy&Co. AI & ML Engineer Assessment",
      packages=find_packages(),
      install_requires=requirements
        )
