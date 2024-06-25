from setuptools import setup
from setuptools.command.install import install
import nltk

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        nltk.download('punkt')

setup(
    setup_requires=['nltk'],
    install_requires=[
        'nltk==3.8.1',
    ]
    cmdclass={
        'install': PostInstallCommand,
    },
)