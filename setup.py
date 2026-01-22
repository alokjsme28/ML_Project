from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'       # Coming from rquirement.txt to trigger setup.py [Need to ignore]

def get_requirements(file_path:str) -> List[str]:
    '''
    This function will return a list of requirements.    
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Remove newline as readlines will append \n after each line
        requirements = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="ML Project",
    version="0.0.1",
    packages=find_packages(),
    install_requires= get_requirements(r"D:\ML\19- ML_Project\requirements.txt"),
    author="Alok",
    description="My sample project",
    author_email="alokjsme28@gmail.com"
)
