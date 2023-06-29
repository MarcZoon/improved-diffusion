from setuptools import setup

setup(
    name="ano3ddpm",
    py_modules=["ano3ddpm"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
