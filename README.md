---

### Installing PyMesh

This project uses PyMesh for the 3D mesh creation. This is included as a git submodule (due to the installation via pip being poor and the use through docker being unsatisfactory). 

**Initialise the submodule**
```
git submodule update --init
cd PyMesh
git submodule update --init
```

**Export path***
```
export PYMESH_PATH=$(pwd)
```

**Build & install to local pipenv**
```
./setup.py build
cd ..
pipenv shell
cd PyMesh
./setup.py install
```

**Test**
```
python -c "import pymesh; pymesh.test()"
```

If this fails, the full instructions for building PyMesh can be found [here](https://github.com/PyMesh/PyMesh).
