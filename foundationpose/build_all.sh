DIR=$(pwd)

cd $DIR/mycpp/ && mkdir -p build && cd build && cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11
# Kaolin and bundlesdf are not needed because we are not using the model-free setup
#cd /kaolin && rm -rf build *egg* && pip install -e .
#cd $DIR/bundlesdf/mycuda && rm -rf build *egg* && pip install -e .

# Install an older version of pyglet so the 3d visualizer works
pip install "pyglet<2"

cd ${DIR}
