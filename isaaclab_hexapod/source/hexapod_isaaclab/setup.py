"""Installation entry point for the Hexapod Isaac Lab extension."""

from setuptools import find_packages, setup


setup(
    name="hexapod_isaaclab",
    version="0.1.0",
    description="Hexapod MJX contract replay and Isaac Lab DirectRLEnv scaffold",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=["numpy"],
    zip_safe=False,
)

