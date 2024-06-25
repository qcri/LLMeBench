from setuptools import setup
from setuptools.command.install import install
import nltk

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        nltk.download('punkt')

setup(
    cmdclass={
        'install': PostInstallCommand,
    },
)