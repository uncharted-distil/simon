from distutils.core import setup

setup(name='Simon',
    version='1.1.0',
    description='Character-level CNN+LSTM model for text classification',
    packages=['Simon'],
    install_requires=['Faker >= 0.7.7',
        'Keras >= 2.0.2',
        'scikit-learn >= 0.18.1',
        'python-dateutil >= 2.5.3',
        'pandas >= 0.19.2',
        'scipy >= 0.19.0',
        'tensorflow >= 1.1.0',
        'h5py >= 2.7.0'],
    data_files=['',['Simon/types.json']],
)