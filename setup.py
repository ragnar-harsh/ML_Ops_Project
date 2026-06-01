from setuptools import setup, find_packages
from typing import List


HYPHEN_E_DOT = "-e ."

def get_requirements(file_path : str) -> List[str]:
    '''
    This function fetch the requirements from Requirement.txt
    '''

    requirements = []

    with open(file_path) as file_obj:
        reqs = file_obj.readlines()
        requirements = [r.replace("\n", "") for r in reqs]

        # if HYPHEN_E_DOT in requirements:
        #     requirements.remove(HYPHEN_E_DOT)
    
    return requirements



setup(


    name = "ML_Ops_Project",
    version = "1.0.0",
    author = "ragnar",
    author_email = "ragnar_harsh@yahoo.com",
    packages = find_packages(),
    # install_requires = ['numpy', 'pandas'],
    install_requires = get_requirements('requirements.txt'),
)

