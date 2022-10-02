from setuptools import setup, find_packages

requirements = [
    "ffmpeg==1.4",
    "gym==0.21.0",
    "Kivy==2.1.0",
    "matplotlib==3.5.2",
    "numpy==1.23.1",
    "opencv-python==4.6.0.66",
    "Pillow==9.2.0",
    "pytest==7.1.2",
    "pywin32==304",
    "stable-baselines3==1.6.0",
    "torch==1.12.0",
    "imageio-ffmpeg==0.4.7",
    "vgamepad==0.0.8",
    "pygame==2.1.2",
    "pynput==1.7.6",
    "pydantic==1.10.2",
    "tables==3.7.0",
]

setup(
    name="Racing Toolbox",
    version="0.1.0",
    description="Racing Toolbox - module for RL with racing games",
    packages=find_packages(),
    install_requires=requirements,
)
