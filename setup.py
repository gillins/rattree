from distutils.core import setup
import glob

scripts_list = glob.glob("bin/*.py")

setup(name='rattree',
      version='0.1',
      description='Building of Raster Attribute Tables using a Tree data structure',
      author='Sam Gillingham',
      author_email='gillingham.sam@gmail.com',
      scripts=scripts_list,
      packages=['rattree'],
      license='LICENSE.txt', 
      url='https://github.com/gillins/rattree'
     )

