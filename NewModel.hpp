// Copyright Lin Min 2015
#ifndef PURINE_GOOGLENET
#define PURINE_GOOGLENET

#include "common/common.hpp"
#include "dispatch/runnable.hpp"
#include "composite/composite.hpp"

extern int batch_size;
extern string source;
extern string mean_file;

template <bool test>
class VirtualNetName : public Graph {
 protected:
  Blob* data_;
  Blob* label_;
  Blob* data_diff_;
  vector<Blob*> weights_;
  vector<Blob*> weight_data_;
  vector<Blob*> weight_diff_;
  vector<Blob*> loss_;
 public:
  explicit VirtualNetName(int rank, int device);
  virtual ~VirtualNetName() override {}
  inline const vector<Blob*>& weight_data() { return weight_data_; }
  inline const vector<Blob*>& weight_diff() { return weight_diff_; }
  inline vector<Blob*> data() { return { data_ }; }
  inline vector<Blob*> label() { return { label_ }; }
  inline vector<Blob*> data_diff() { return { data_diff_ }; }
  inline vector<Blob*> loss() { return loss_; }
};

template <bool test>
VirtualNetName<test>::VirtualNetName(int rank, int device) : Graph(rank, device) {

  data_ = create("data", { batch_size, 3, 224, 224 });
  data_diff_ = create("data_diff", { batch_size, 3, 224, 224 });
  label_ = create("label", { batch_size, 1, 1, 1 });
  ConvLayer* conv1 = createGraph<ConvLayer>("conv1",
    ConvLayer::param_tuple(0, 0, 4, 4, 11, 11, 96, "relu"));
  LRNLayer* norm1 = createGraph<LRNLayer>("norm1",
    LRNLayer::param_tuple(0.000100, 0.750000, 5));
  PoolLayer* pool1 = createGraph<PoolLayer>("pool1",
    PoolLayer::param_tuple("max", 3, 3, 2, 2, 0, 0));
  ConvLayer* conv2 = createGraph<ConvLayer>("conv2",
    ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 256, "relu"));
  LRNLayer* norm2 = createGraph<LRNLayer>("norm2",
    LRNLayer::param_tuple(0.000100, 0.750000, 5));
  PoolLayer* pool2 = createGraph<PoolLayer>("pool2",
    PoolLayer::param_tuple("max", 3, 3, 2, 2, 0, 0));
  ConvLayer* conv3 = createGraph<ConvLayer>("conv3",
    ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, 384, "relu"));
  ConvLayer* conv4 = createGraph<ConvLayer>("conv4",
    ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, 384, "relu"));
  ConvLayer* conv5 = createGraph<ConvLayer>("conv5",
    ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, 256, "relu"));
  PoolLayer* pool5 = createGraph<PoolLayer>("pool5",
    PoolLayer::param_tuple("max", 3, 3, 2, 2, 0, 0));
  InnerProdLayer* fc6 = createGraph<InnerProdLayer>("fc6",
    InnerProdLayer::param_tuple(4096, ""));
  DropoutLayer* drop6 = createGraph<DropoutLayer>("drop6",
    DropoutLayer::param_tuple(0.500000, test, false));
  InnerProdLayer* fc7 = createGraph<InnerProdLayer>("fc7",
    InnerProdLayer::param_tuple(4096, ""));
  DropoutLayer* drop7 = createGraph<DropoutLayer>("drop7",
    DropoutLayer::param_tuple(0.500000, test, false));
  InnerProdLayer* fc8 = createGraph<InnerProdLayer>("fc8",
    InnerProdLayer::param_tuple(1000, ""));
  Acc* accuracy = createGraph<Acc>("accuracy", rank_, -1, Acc::param_tuple(1));
  SoftmaxLossLayer* loss = createGraph<SoftmaxLossLayer>("loss",
    SoftmaxLossLayer::param_tuple(1.));


  B{ data_, data_diff_} >> *conv1;
  *conv1 >> *norm1;
  *norm1 >> *pool1;
  *pool1 >> *conv2;
  *conv2 >> *norm2;
  *norm2 >> *pool2;
  *pool2 >> *conv3;
  *conv3 >> *conv4;
  *conv4 >> *conv5;
  *conv5 >> *pool5;
  *pool5 >> *fc6;
  *fc6 >> *drop6;
  *drop6 >> *fc7;
  *fc7 >> *drop7;
  *drop7 >> *fc8;
  accuracy->set_label(label_);
  vector<Blob*>{ fc8->top()[0] } >> *accuracy;
  loss->set_label(label_);
  *fc8 >> *loss;


  loss_ = { loss->loss()[0], accuracy->loss()[0] };


  vector<Layer*> layers = { conv1, conv2, conv3, conv4,
     conv5, fc6, fc7, fc8 };

    

  for (auto layer : layers) {
    const vector<Blob*>& w = layer->weight_data();
    weight_data_.insert(weight_data_.end(), w.begin(), w.end());
  }
  for (auto layer : layers) {
    const vector<Blob*>& w = layer->weight_diff();
    weight_diff_.insert(weight_diff_.end(), w.begin(), w.end());
  }
}

#endif
