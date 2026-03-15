from setuptools import Extension, setup

setup(ext_modules=[Extension("exojit.jitcall", sources=["exojit/jitcall.c"])])
