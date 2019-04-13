r"""Command-line tool to findfaces

Usage::

    $ ff_detect /home/miimagen.jpg

"""
import sys
import os
import tempfile
import findfaces
import shutil

def main():
  args = []
  for i in range(2, len(sys.argv)):
    args.append(sys.argv[i])

  getattr(sys.modules[__name__], sys.argv[1])(args)

#args: templatePath, dataPath=None, outputPath=None
def detect (args):
  #print("executing genfiles " + str(args))
  gf = findfaces.FacesTools()
  gf.detect(*args)

#args: templatePath, dataPath=None, outputPath=None
def cropAllFaces (args):
  #print("executing genfiles " + str(args))
  gf = findfaces.FacesTools()
  gf.cropAllFaces(*args)

#args: type
def help (args):
  #print("executing help " + str(args))
  _show_help(*args)

def _show_help (type=None):
  if type == 'detect':
    cwd = os.getcwd()
    tmpDir = tempfile.mkdtemp(prefix='test_', suffix='_findfaces', dir=cwd)
    print("cd " + tmpDir)
    print("Examples of 'test/detect faces':")
    print("$ ff_detect img1.jpg")
  else:
    print("options: detect.")
    print("options: cropAllFaces.")
    print("- detect: Find faces.")
    print("- cropAllFaces: Crop all faces.")
    print("")
    print("Examples of 'find faces':")
    print("$ ff_detect img1.jpg")
    print("$ ff_cropAllFaces img1.jpg")

if __name__ == '__main__':
    main()

