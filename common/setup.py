from setuptools import setup, find_packages

requirements_common = [
    "ffmpeg==1.4",
    "gym==0.21.0",
    "Kivy==2.1.0",
    "matplotlib==3.5.2",
    "numpy==1.23.1",
    "opencv-python==4.6.0.66",
    "Pillow==9.2.0",
    "pytest==7.1.2",
    "tensorboard==2.9.1",
    "torch==1.12.0",
    "wandb==0.12.21",
    "ffmpeg==1.4",
    "imageio-ffmpeg==0.4.7",
    "ray[rllib]",
    "pygame==2.1.2",
    "pynput==1.7.5",
    "pydantic==1.10.2",
    "tables==3.7.0",
    "streamlit",
    "httpx-oauth==0.10.2",
    "ray[rllib]",
]

windows_requirements = [
    "pywin32==304",
    "vgamepad==0.0.8"

]
import sys 

requirements = requirements_common
if "linux" not in sys.platform:
    requirements += windows_requirements

setup(
    name="Racing Toolbox",
    version="0.1.0",
    description="Racing Toolbox - module for RL with racing games",
    packages=find_packages(),
    install_requires=requirements,
)
