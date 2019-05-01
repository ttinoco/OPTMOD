echo "cleaning OPTMOD..."
find . -name \*~ -delete
find . -name \*.pyc -delete
find . -name \*.so -delete
find . -name __pycache__ -delete
rm -rf build
rm -rf dist
rm -rf OPTMOD.egg-info
rm -f ./optmod/coptmod/coptmod.c
