from setuptools import setup,find_packages

setup(
    name='cacti',
    version='0.2',
    packages=find_packages(),
    description="This is a SCI pytorch package.",
    author="Lishun WANG",
    author_email="wanglishun17@mails.ucas.edu.cn",
    url="https://github.com/ucaswangls/cacti",
    include_package_data=True,
    zip_safe=False
)