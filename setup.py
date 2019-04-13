# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='findfaces',
    version='0.0.02',
    url='https://github.com/tuaplicacionpropia/findfaces',
    download_url='https://github.com/tuaplicacionpropia/findfaces/archive/master.zip',
    author=u'tuaplicacionpropia.com',
    author_email='tuaplicacionpropia@gmail.com',
    description='Python library for find faces.',
    long_description='Python library for find faces.',
    keywords='faces, detect, dnn, opencv',
    classifiers=[
      'Development Status :: 4 - Beta',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python',
      'Programming Language :: Python :: 2.7',
      'Intended Audience :: Developers',
      'Topic :: Multimedia :: Graphics',
    ],
    scripts=[
      'bin/ff_detect.cmd', 'bin/ff_detect',
      'bin/ff_cropAllFaces.cmd', 'bin/ff_cropAllFaces',
      'bin/ff_help.cmd', 'bin/ff_help'
    ],
    packages=find_packages(exclude=['tests']),
    #package_data={},
    #package_data={'': ['license.txt']},
    package_data={'findfaces': ['images/*.jpg', 'models/*.txt', 'models/*.caffemodel']},
    zip_safe=False,
    include_package_data=True,
    platforms='any',
    license='MIT',
    install_requires=[
        'numpy>=1.12.0',
        'opencv-python>=3.1.0.3',
        'screeninfo>=0.3.1'
    ],
)

