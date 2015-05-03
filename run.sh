python caffe2purine.py ../caffe/models/bvlc_alexnet/train_val.prototxt
#python caffe2purine.py ../caffe/models/bvlc_googlenet/train_val.prototxt
cp NewModel.* ../purine2/examples/
cd ../purine2/
cmake .
make NewModel
test/NewModel
