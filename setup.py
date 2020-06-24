from distutils.core import setup

setup(name='Simon',
    version='1.2.4',
    description='Character-level CNN+LSTM model for text classification',
    packages=['Simon'],
    install_requires=['Faker >= 0.7.7',
        'scikit-learn==0.22.2.post1',
        'python-dateutil==2.8.1',
        'pandas==1.0.3',
        'scipy==1.4.1',
        'tensorflow-gpu==2.2.0',
        'h5py >= 2.7.0'],
    include_package_data=True,
)
