from setuptools import setup
from setuptools.command.install import install
import nltk
import subprocess

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        subprocess.run(["python", "-m", "nltk.downloader", "punkt"], check=True)

setup(
    setup_requires=['nltk'],
    install_requires=[
        'nltk==3.8.1',
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
)