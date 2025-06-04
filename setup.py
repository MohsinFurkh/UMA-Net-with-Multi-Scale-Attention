from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='umanet',
    version='1.0.0',
    author='Mohsin Furkh Dar, Avatharam Ganivada',
    author_email='20mcpc02@uohyd.ac.in',
    description='UMA-Net: Adaptive Ensemble Loss and Multi-Scale Attention for Medical Image Segmentation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MohsinFurkh/UMA-Net-with-Multi-Scale-Attention',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.8.0',
        'numpy>=1.19.5',
        'opencv-python>=4.5.5',
        'scikit-image>=0.19.0',
        'scikit-learn>=1.0.2',
        'matplotlib>=3.5.1',
        'seaborn>=0.11.2',
        'pandas>=1.3.5',
        'scipy>=1.7.3',
        'tqdm>=4.62.3',
        'pillow>=9.0.0',
        'pyyaml>=6.0',
        'h5py>=3.6.0',
        'albumentations>=1.1.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
    ],
    python_requires='>=3.8',
    keywords=[
        'deep-learning',
        'medical-imaging',
        'image-segmentation',
        'breast-cancer',
        'ultrasound',
        'attention-mechanism',
        'ensemble-learning'
    ],
)
