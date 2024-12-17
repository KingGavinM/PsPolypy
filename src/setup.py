from setuptools import setup, find_packages

setup(
    name = 'PsPolypy',
    version = '0.1.0',
    author = 'Creighton M. Lisowski',
    author_email = 'clisowski@missouri.edu',
    packages = find_packages(exclude=['tests*']),
    license = 'GPL-3.0',
    description = 'Python package for atuomated detection of polymers and persistence length analysis in AFM images.',
    extras_require = {
    },
    classifiers=[
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
]
)