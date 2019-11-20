from distutils.core import setup

setup(name='Simon',
    version='1.2.4',
    description='Character-level CNN+LSTM model for text classification',
    packages=['Simon'],
    install_requires=['Faker >= 0.7.7',
        'Keras >= 2.0.2, <= 2.2.4',
        'scikit-learn>=0.20.3,<=0.21.3',
        'python-dateutil==2.8.1',
        'pandas>=0.23.4,<=0.25.2',
        'scipy>=1.2.1,<=1.3.1',
        'tensorflow-gpu == 2.0.0',
        'h5py >= 2.7.0'],
    include_package_data=True,
)
