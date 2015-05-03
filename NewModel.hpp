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
  ConvLayer* conv1_7x7_s2 = createGraph<ConvLayer>("conv1_7x7_s2",
    ConvLayer::param_tuple(3, 3, 2, 2, 7, 7, 64, "relu"));
  PoolLayer* pool1_3x3_s2 = createGraph<PoolLayer>("pool1_3x3_s2",
    PoolLayer::param_tuple("max", 3, 3, 2, 2, 0, 0));
  LRNLayer* pool1_norm1 = createGraph<LRNLayer>("pool1_norm1",
    LRNLayer::param_tuple(0.000100, 0.750000, 5));
  ConvLayer* conv2_3x3_reduce = createGraph<ConvLayer>("conv2_3x3_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 64, "relu"));
  ConvLayer* conv2_3x3 = createGraph<ConvLayer>("conv2_3x3",
    ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, 192, "relu"));
  LRNLayer* conv2_norm2 = createGraph<LRNLayer>("conv2_norm2",
    LRNLayer::param_tuple(0.000100, 0.750000, 5));
  PoolLayer* pool2_3x3_s2 = createGraph<PoolLayer>("pool2_3x3_s2",
    PoolLayer::param_tuple("max", 3, 3, 2, 2, 0, 0));
  ConvLayer* inception_3a_1x1 = createGraph<ConvLayer>("inception_3a_1x1",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 64, "relu"));
  ConvLayer* inception_3a_3x3_reduce = createGraph<ConvLayer>("inception_3a_3x3_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 96, "relu"));
  ConvLayer* inception_3a_3x3 = createGraph<ConvLayer>("inception_3a_3x3",
    ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, 128, "relu"));
  ConvLayer* inception_3a_5x5_reduce = createGraph<ConvLayer>("inception_3a_5x5_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 16, "relu"));
  ConvLayer* inception_3a_5x5 = createGraph<ConvLayer>("inception_3a_5x5",
    ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 32, "relu"));
  PoolLayer* inception_3a_pool = createGraph<PoolLayer>("inception_3a_pool",
    PoolLayer::param_tuple("max", 3, 3, 1, 1, 1, 1));
  ConvLayer* inception_3a_pool_proj = createGraph<ConvLayer>("inception_3a_pool_proj",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 32, "relu"));
  ConcatLayer* inception_3a_output = createGraph<ConcatLayer>("inception_3a_output",
    ConcatLayer::param_tuple(Split::CHANNELS));
  ActivationLayer* act_of_inception_3a_output = createGraph<ActivationLayer>("act_of_inception_3a_output",
    ActivationLayer::param_tuple("relu", true));
  ConvLayer* inception_3b_1x1 = createGraph<ConvLayer>("inception_3b_1x1",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 128, "relu"));
  ConvLayer* inception_3b_3x3_reduce = createGraph<ConvLayer>("inception_3b_3x3_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 128, "relu"));
  ConvLayer* inception_3b_3x3 = createGraph<ConvLayer>("inception_3b_3x3",
    ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, 192, "relu"));
  ConvLayer* inception_3b_5x5_reduce = createGraph<ConvLayer>("inception_3b_5x5_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 32, "relu"));
  ConvLayer* inception_3b_5x5 = createGraph<ConvLayer>("inception_3b_5x5",
    ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 96, "relu"));
  PoolLayer* inception_3b_pool = createGraph<PoolLayer>("inception_3b_pool",
    PoolLayer::param_tuple("max", 3, 3, 1, 1, 1, 1));
  ConvLayer* inception_3b_pool_proj = createGraph<ConvLayer>("inception_3b_pool_proj",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 64, "relu"));
  ConcatLayer* inception_3b_output = createGraph<ConcatLayer>("inception_3b_output",
    ConcatLayer::param_tuple(Split::CHANNELS));
  ActivationLayer* act_of_inception_3b_output = createGraph<ActivationLayer>("act_of_inception_3b_output",
    ActivationLayer::param_tuple("relu", true));
  PoolLayer* pool3_3x3_s2 = createGraph<PoolLayer>("pool3_3x3_s2",
    PoolLayer::param_tuple("max", 3, 3, 2, 2, 0, 0));
  ConvLayer* inception_4a_1x1 = createGraph<ConvLayer>("inception_4a_1x1",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 192, "relu"));
  ConvLayer* inception_4a_3x3_reduce = createGraph<ConvLayer>("inception_4a_3x3_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 96, "relu"));
  ConvLayer* inception_4a_3x3 = createGraph<ConvLayer>("inception_4a_3x3",
    ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, 208, "relu"));
  ConvLayer* inception_4a_5x5_reduce = createGraph<ConvLayer>("inception_4a_5x5_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 16, "relu"));
  ConvLayer* inception_4a_5x5 = createGraph<ConvLayer>("inception_4a_5x5",
    ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 48, "relu"));
  PoolLayer* inception_4a_pool = createGraph<PoolLayer>("inception_4a_pool",
    PoolLayer::param_tuple("max", 3, 3, 1, 1, 1, 1));
  ConvLayer* inception_4a_pool_proj = createGraph<ConvLayer>("inception_4a_pool_proj",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 64, "relu"));
  ConcatLayer* inception_4a_output = createGraph<ConcatLayer>("inception_4a_output",
    ConcatLayer::param_tuple(Split::CHANNELS));
  ActivationLayer* act_of_inception_4a_output = createGraph<ActivationLayer>("act_of_inception_4a_output",
    ActivationLayer::param_tuple("relu", true));
  PoolLayer* loss1_ave_pool = createGraph<PoolLayer>("loss1_ave_pool",
    PoolLayer::param_tuple("avg", 5, 5, 3, 3, 0, 0));
  ConvLayer* loss1_conv = createGraph<ConvLayer>("loss1_conv",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 128, "relu"));
  InnerProdLayer* loss1_fc = createGraph<InnerProdLayer>("loss1_fc",
    InnerProdLayer::param_tuple(1024, ""));
  DropoutLayer* loss1_drop_fc = createGraph<DropoutLayer>("loss1_drop_fc",
    DropoutLayer::param_tuple(0.700000, test, false));
  InnerProdLayer* loss1_classifier = createGraph<InnerProdLayer>("loss1_classifier",
    InnerProdLayer::param_tuple(1000, ""));
  SoftmaxLossLayer* loss1_loss = createGraph<SoftmaxLossLayer>("loss1_loss",
    SoftmaxLossLayer::param_tuple(1.));
  Acc* loss1_top_1 = createGraph<Acc>("loss1_top_1", rank_, -1, Acc::param_tuple(1));
  Acc* loss1_top_5 = createGraph<Acc>("loss1_top_5", rank_, -1, Acc::param_tuple(1));
  ConvLayer* inception_4b_1x1 = createGraph<ConvLayer>("inception_4b_1x1",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 160, "relu"));
  ConvLayer* inception_4b_3x3_reduce = createGraph<ConvLayer>("inception_4b_3x3_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 112, "relu"));
  ConvLayer* inception_4b_3x3 = createGraph<ConvLayer>("inception_4b_3x3",
    ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, 224, "relu"));
  ConvLayer* inception_4b_5x5_reduce = createGraph<ConvLayer>("inception_4b_5x5_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 24, "relu"));
  ConvLayer* inception_4b_5x5 = createGraph<ConvLayer>("inception_4b_5x5",
    ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 64, "relu"));
  PoolLayer* inception_4b_pool = createGraph<PoolLayer>("inception_4b_pool",
    PoolLayer::param_tuple("max", 3, 3, 1, 1, 1, 1));
  ConvLayer* inception_4b_pool_proj = createGraph<ConvLayer>("inception_4b_pool_proj",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 64, "relu"));
  ConcatLayer* inception_4b_output = createGraph<ConcatLayer>("inception_4b_output",
    ConcatLayer::param_tuple(Split::CHANNELS));
  ActivationLayer* act_of_inception_4b_output = createGraph<ActivationLayer>("act_of_inception_4b_output",
    ActivationLayer::param_tuple("relu", true));
  ConvLayer* inception_4c_1x1 = createGraph<ConvLayer>("inception_4c_1x1",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 128, "relu"));
  ConvLayer* inception_4c_3x3_reduce = createGraph<ConvLayer>("inception_4c_3x3_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 128, "relu"));
  ConvLayer* inception_4c_3x3 = createGraph<ConvLayer>("inception_4c_3x3",
    ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, 256, "relu"));
  ConvLayer* inception_4c_5x5_reduce = createGraph<ConvLayer>("inception_4c_5x5_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 24, "relu"));
  ConvLayer* inception_4c_5x5 = createGraph<ConvLayer>("inception_4c_5x5",
    ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 64, "relu"));
  PoolLayer* inception_4c_pool = createGraph<PoolLayer>("inception_4c_pool",
    PoolLayer::param_tuple("max", 3, 3, 1, 1, 1, 1));
  ConvLayer* inception_4c_pool_proj = createGraph<ConvLayer>("inception_4c_pool_proj",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 64, "relu"));
  ConcatLayer* inception_4c_output = createGraph<ConcatLayer>("inception_4c_output",
    ConcatLayer::param_tuple(Split::CHANNELS));
  ActivationLayer* act_of_inception_4c_output = createGraph<ActivationLayer>("act_of_inception_4c_output",
    ActivationLayer::param_tuple("relu", true));
  ConvLayer* inception_4d_1x1 = createGraph<ConvLayer>("inception_4d_1x1",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 112, "relu"));
  ConvLayer* inception_4d_3x3_reduce = createGraph<ConvLayer>("inception_4d_3x3_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 144, "relu"));
  ConvLayer* inception_4d_3x3 = createGraph<ConvLayer>("inception_4d_3x3",
    ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, 288, "relu"));
  ConvLayer* inception_4d_5x5_reduce = createGraph<ConvLayer>("inception_4d_5x5_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 32, "relu"));
  ConvLayer* inception_4d_5x5 = createGraph<ConvLayer>("inception_4d_5x5",
    ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 64, "relu"));
  PoolLayer* inception_4d_pool = createGraph<PoolLayer>("inception_4d_pool",
    PoolLayer::param_tuple("max", 3, 3, 1, 1, 1, 1));
  ConvLayer* inception_4d_pool_proj = createGraph<ConvLayer>("inception_4d_pool_proj",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 64, "relu"));
  ConcatLayer* inception_4d_output = createGraph<ConcatLayer>("inception_4d_output",
    ConcatLayer::param_tuple(Split::CHANNELS));
  ActivationLayer* act_of_inception_4d_output = createGraph<ActivationLayer>("act_of_inception_4d_output",
    ActivationLayer::param_tuple("relu", true));
  PoolLayer* loss2_ave_pool = createGraph<PoolLayer>("loss2_ave_pool",
    PoolLayer::param_tuple("avg", 5, 5, 3, 3, 0, 0));
  ConvLayer* loss2_conv = createGraph<ConvLayer>("loss2_conv",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 128, "relu"));
  InnerProdLayer* loss2_fc = createGraph<InnerProdLayer>("loss2_fc",
    InnerProdLayer::param_tuple(1024, ""));
  DropoutLayer* loss2_drop_fc = createGraph<DropoutLayer>("loss2_drop_fc",
    DropoutLayer::param_tuple(0.700000, test, false));
  InnerProdLayer* loss2_classifier = createGraph<InnerProdLayer>("loss2_classifier",
    InnerProdLayer::param_tuple(1000, ""));
  SoftmaxLossLayer* loss2_loss = createGraph<SoftmaxLossLayer>("loss2_loss",
    SoftmaxLossLayer::param_tuple(1.));
  Acc* loss2_top_1 = createGraph<Acc>("loss2_top_1", rank_, -1, Acc::param_tuple(1));
  Acc* loss2_top_5 = createGraph<Acc>("loss2_top_5", rank_, -1, Acc::param_tuple(1));
  ConvLayer* inception_4e_1x1 = createGraph<ConvLayer>("inception_4e_1x1",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 256, "relu"));
  ConvLayer* inception_4e_3x3_reduce = createGraph<ConvLayer>("inception_4e_3x3_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 160, "relu"));
  ConvLayer* inception_4e_3x3 = createGraph<ConvLayer>("inception_4e_3x3",
    ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, 320, "relu"));
  ConvLayer* inception_4e_5x5_reduce = createGraph<ConvLayer>("inception_4e_5x5_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 32, "relu"));
  ConvLayer* inception_4e_5x5 = createGraph<ConvLayer>("inception_4e_5x5",
    ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 128, "relu"));
  PoolLayer* inception_4e_pool = createGraph<PoolLayer>("inception_4e_pool",
    PoolLayer::param_tuple("max", 3, 3, 1, 1, 1, 1));
  ConvLayer* inception_4e_pool_proj = createGraph<ConvLayer>("inception_4e_pool_proj",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 128, "relu"));
  ConcatLayer* inception_4e_output = createGraph<ConcatLayer>("inception_4e_output",
    ConcatLayer::param_tuple(Split::CHANNELS));
  ActivationLayer* act_of_inception_4e_output = createGraph<ActivationLayer>("act_of_inception_4e_output",
    ActivationLayer::param_tuple("relu", true));
  PoolLayer* pool4_3x3_s2 = createGraph<PoolLayer>("pool4_3x3_s2",
    PoolLayer::param_tuple("max", 3, 3, 2, 2, 0, 0));
  ConvLayer* inception_5a_1x1 = createGraph<ConvLayer>("inception_5a_1x1",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 256, "relu"));
  ConvLayer* inception_5a_3x3_reduce = createGraph<ConvLayer>("inception_5a_3x3_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 160, "relu"));
  ConvLayer* inception_5a_3x3 = createGraph<ConvLayer>("inception_5a_3x3",
    ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, 320, "relu"));
  ConvLayer* inception_5a_5x5_reduce = createGraph<ConvLayer>("inception_5a_5x5_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 32, "relu"));
  ConvLayer* inception_5a_5x5 = createGraph<ConvLayer>("inception_5a_5x5",
    ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 128, "relu"));
  PoolLayer* inception_5a_pool = createGraph<PoolLayer>("inception_5a_pool",
    PoolLayer::param_tuple("max", 3, 3, 1, 1, 1, 1));
  ConvLayer* inception_5a_pool_proj = createGraph<ConvLayer>("inception_5a_pool_proj",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 128, "relu"));
  ConcatLayer* inception_5a_output = createGraph<ConcatLayer>("inception_5a_output",
    ConcatLayer::param_tuple(Split::CHANNELS));
  ActivationLayer* act_of_inception_5a_output = createGraph<ActivationLayer>("act_of_inception_5a_output",
    ActivationLayer::param_tuple("relu", true));
  ConvLayer* inception_5b_1x1 = createGraph<ConvLayer>("inception_5b_1x1",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 384, "relu"));
  ConvLayer* inception_5b_3x3_reduce = createGraph<ConvLayer>("inception_5b_3x3_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 192, "relu"));
  ConvLayer* inception_5b_3x3 = createGraph<ConvLayer>("inception_5b_3x3",
    ConvLayer::param_tuple(1, 1, 1, 1, 3, 3, 384, "relu"));
  ConvLayer* inception_5b_5x5_reduce = createGraph<ConvLayer>("inception_5b_5x5_reduce",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 48, "relu"));
  ConvLayer* inception_5b_5x5 = createGraph<ConvLayer>("inception_5b_5x5",
    ConvLayer::param_tuple(2, 2, 1, 1, 5, 5, 128, "relu"));
  PoolLayer* inception_5b_pool = createGraph<PoolLayer>("inception_5b_pool",
    PoolLayer::param_tuple("max", 3, 3, 1, 1, 1, 1));
  ConvLayer* inception_5b_pool_proj = createGraph<ConvLayer>("inception_5b_pool_proj",
    ConvLayer::param_tuple(0, 0, 1, 1, 1, 1, 128, "relu"));
  ConcatLayer* inception_5b_output = createGraph<ConcatLayer>("inception_5b_output",
    ConcatLayer::param_tuple(Split::CHANNELS));
  ActivationLayer* act_of_inception_5b_output = createGraph<ActivationLayer>("act_of_inception_5b_output",
    ActivationLayer::param_tuple("relu", true));
  PoolLayer* pool5_7x7_s1 = createGraph<PoolLayer>("pool5_7x7_s1",
    PoolLayer::param_tuple("avg", 7, 7, 1, 1, 0, 0));
  DropoutLayer* pool5_drop_7x7_s1 = createGraph<DropoutLayer>("pool5_drop_7x7_s1",
    DropoutLayer::param_tuple(0.400000, test, false));
  InnerProdLayer* loss3_classifier = createGraph<InnerProdLayer>("loss3_classifier",
    InnerProdLayer::param_tuple(1000, ""));
  SoftmaxLossLayer* loss3_loss3 = createGraph<SoftmaxLossLayer>("loss3_loss3",
    SoftmaxLossLayer::param_tuple(1.));
  Acc* loss3_top_1 = createGraph<Acc>("loss3_top_1", rank_, -1, Acc::param_tuple(1));
  Acc* loss3_top_5 = createGraph<Acc>("loss3_top_5", rank_, -1, Acc::param_tuple(1));


  *B{ data_, data_diff_} >> *conv1_7x7_s2;
  *data >> *conv1_7x7_s2;
  *conv1_7x7_s2 >> *pool1_3x3_s2;
  *pool1_3x3_s2 >> *pool1_norm1;
  *pool1_norm1 >> *conv2_3x3_reduce;
  *conv2_3x3_reduce >> *conv2_3x3;
  *conv2_3x3 >> *conv2_norm2;
  *conv2_norm2 >> *pool2_3x3_s2;
  *pool2_3x3_s2 >> *inception_3a_1x1;
  *pool2_3x3_s2 >> *inception_3a_3x3_reduce;
  *inception_3a_3x3_reduce >> *inception_3a_3x3;
  *pool2_3x3_s2 >> *inception_3a_5x5_reduce;
  *inception_3a_5x5_reduce >> *inception_3a_5x5;
  *pool2_3x3_s2 >> *inception_3a_pool;
  *inception_3a_pool >> *inception_3a_pool_proj;
  vector<Blob*> {inception_3a_1x1->top()[0], inception_3a_3x3->top()[0], inception_3a_5x5->top()[0], inception_3a_pool_proj->top()[0], 
    inception_3a_1x1->top()[1], inception_3a_3x3->top()[1], inception_3a_5x5->top()[1], inception_3a_pool_proj->top()[1]} >> *inception_3a_output >> *act_of_inception_3a_output
  *inception_3a_1x1 >> *inception_3a_output;
  *inception_3a_3x3 >> *inception_3a_output;
  *inception_3a_5x5 >> *inception_3a_output;
  *inception_3a_pool_proj >> *inception_3a_output;
  *inception_3a_output >> *inception_3b_1x1;
  *inception_3a_output >> *inception_3b_3x3_reduce;
  *inception_3b_3x3_reduce >> *inception_3b_3x3;
  *inception_3a_output >> *inception_3b_5x5_reduce;
  *inception_3b_5x5_reduce >> *inception_3b_5x5;
  *inception_3a_output >> *inception_3b_pool;
  *inception_3b_pool >> *inception_3b_pool_proj;
  vector<Blob*> {inception_3b_1x1->top()[0], inception_3b_3x3->top()[0], inception_3b_5x5->top()[0], inception_3b_pool_proj->top()[0], 
    inception_3b_1x1->top()[1], inception_3b_3x3->top()[1], inception_3b_5x5->top()[1], inception_3b_pool_proj->top()[1]} >> *inception_3b_output >> *act_of_inception_3b_output
  *inception_3b_1x1 >> *inception_3b_output;
  *inception_3b_3x3 >> *inception_3b_output;
  *inception_3b_5x5 >> *inception_3b_output;
  *inception_3b_pool_proj >> *inception_3b_output;
  *inception_3b_output >> *pool3_3x3_s2;
  *pool3_3x3_s2 >> *inception_4a_1x1;
  *pool3_3x3_s2 >> *inception_4a_3x3_reduce;
  *inception_4a_3x3_reduce >> *inception_4a_3x3;
  *pool3_3x3_s2 >> *inception_4a_5x5_reduce;
  *inception_4a_5x5_reduce >> *inception_4a_5x5;
  *pool3_3x3_s2 >> *inception_4a_pool;
  *inception_4a_pool >> *inception_4a_pool_proj;
  vector<Blob*> {inception_4a_1x1->top()[0], inception_4a_3x3->top()[0], inception_4a_5x5->top()[0], inception_4a_pool_proj->top()[0], 
    inception_4a_1x1->top()[1], inception_4a_3x3->top()[1], inception_4a_5x5->top()[1], inception_4a_pool_proj->top()[1]} >> *inception_4a_output >> *act_of_inception_4a_output
  *inception_4a_1x1 >> *inception_4a_output;
  *inception_4a_3x3 >> *inception_4a_output;
  *inception_4a_5x5 >> *inception_4a_output;
  *inception_4a_pool_proj >> *inception_4a_output;
  *inception_4a_output >> *loss1_ave_pool;
  *loss1_ave_pool >> *loss1_conv;
  *loss1_conv >> *loss1_fc;
  *loss1_fc >> *loss1_drop_fc;
  *loss1_drop_fc >> *loss1_classifier;
  loss1_loss->set_label(label_);
  *loss1_classifier >> *loss1_loss;
  loss1_top_1->set_label(label_);
  vector<Blob*>{ loss1_classifier->top()[0] } >> *loss1_top_1;
  loss1_top_5->set_label(label_);
  vector<Blob*>{ loss1_classifier->top()[0] } >> *loss1_top_5;
  *inception_4a_output >> *inception_4b_1x1;
  *inception_4a_output >> *inception_4b_3x3_reduce;
  *inception_4b_3x3_reduce >> *inception_4b_3x3;
  *inception_4a_output >> *inception_4b_5x5_reduce;
  *inception_4b_5x5_reduce >> *inception_4b_5x5;
  *inception_4a_output >> *inception_4b_pool;
  *inception_4b_pool >> *inception_4b_pool_proj;
  vector<Blob*> {inception_4b_1x1->top()[0], inception_4b_3x3->top()[0], inception_4b_5x5->top()[0], inception_4b_pool_proj->top()[0], 
    inception_4b_1x1->top()[1], inception_4b_3x3->top()[1], inception_4b_5x5->top()[1], inception_4b_pool_proj->top()[1]} >> *inception_4b_output >> *act_of_inception_4b_output
  *inception_4b_1x1 >> *inception_4b_output;
  *inception_4b_3x3 >> *inception_4b_output;
  *inception_4b_5x5 >> *inception_4b_output;
  *inception_4b_pool_proj >> *inception_4b_output;
  *inception_4b_output >> *inception_4c_1x1;
  *inception_4b_output >> *inception_4c_3x3_reduce;
  *inception_4c_3x3_reduce >> *inception_4c_3x3;
  *inception_4b_output >> *inception_4c_5x5_reduce;
  *inception_4c_5x5_reduce >> *inception_4c_5x5;
  *inception_4b_output >> *inception_4c_pool;
  *inception_4c_pool >> *inception_4c_pool_proj;
  vector<Blob*> {inception_4c_1x1->top()[0], inception_4c_3x3->top()[0], inception_4c_5x5->top()[0], inception_4c_pool_proj->top()[0], 
    inception_4c_1x1->top()[1], inception_4c_3x3->top()[1], inception_4c_5x5->top()[1], inception_4c_pool_proj->top()[1]} >> *inception_4c_output >> *act_of_inception_4c_output
  *inception_4c_1x1 >> *inception_4c_output;
  *inception_4c_3x3 >> *inception_4c_output;
  *inception_4c_5x5 >> *inception_4c_output;
  *inception_4c_pool_proj >> *inception_4c_output;
  *inception_4c_output >> *inception_4d_1x1;
  *inception_4c_output >> *inception_4d_3x3_reduce;
  *inception_4d_3x3_reduce >> *inception_4d_3x3;
  *inception_4c_output >> *inception_4d_5x5_reduce;
  *inception_4d_5x5_reduce >> *inception_4d_5x5;
  *inception_4c_output >> *inception_4d_pool;
  *inception_4d_pool >> *inception_4d_pool_proj;
  vector<Blob*> {inception_4d_1x1->top()[0], inception_4d_3x3->top()[0], inception_4d_5x5->top()[0], inception_4d_pool_proj->top()[0], 
    inception_4d_1x1->top()[1], inception_4d_3x3->top()[1], inception_4d_5x5->top()[1], inception_4d_pool_proj->top()[1]} >> *inception_4d_output >> *act_of_inception_4d_output
  *inception_4d_1x1 >> *inception_4d_output;
  *inception_4d_3x3 >> *inception_4d_output;
  *inception_4d_5x5 >> *inception_4d_output;
  *inception_4d_pool_proj >> *inception_4d_output;
  *inception_4d_output >> *loss2_ave_pool;
  *loss2_ave_pool >> *loss2_conv;
  *loss2_conv >> *loss2_fc;
  *loss2_fc >> *loss2_drop_fc;
  *loss2_drop_fc >> *loss2_classifier;
  loss2_loss->set_label(label_);
  *loss2_classifier >> *loss2_loss;
  loss2_top_1->set_label(label_);
  vector<Blob*>{ loss2_classifier->top()[0] } >> *loss2_top_1;
  loss2_top_5->set_label(label_);
  vector<Blob*>{ loss2_classifier->top()[0] } >> *loss2_top_5;
  *inception_4d_output >> *inception_4e_1x1;
  *inception_4d_output >> *inception_4e_3x3_reduce;
  *inception_4e_3x3_reduce >> *inception_4e_3x3;
  *inception_4d_output >> *inception_4e_5x5_reduce;
  *inception_4e_5x5_reduce >> *inception_4e_5x5;
  *inception_4d_output >> *inception_4e_pool;
  *inception_4e_pool >> *inception_4e_pool_proj;
  vector<Blob*> {inception_4e_1x1->top()[0], inception_4e_3x3->top()[0], inception_4e_5x5->top()[0], inception_4e_pool_proj->top()[0], 
    inception_4e_1x1->top()[1], inception_4e_3x3->top()[1], inception_4e_5x5->top()[1], inception_4e_pool_proj->top()[1]} >> *inception_4e_output >> *act_of_inception_4e_output
  *inception_4e_1x1 >> *inception_4e_output;
  *inception_4e_3x3 >> *inception_4e_output;
  *inception_4e_5x5 >> *inception_4e_output;
  *inception_4e_pool_proj >> *inception_4e_output;
  *inception_4e_output >> *pool4_3x3_s2;
  *pool4_3x3_s2 >> *inception_5a_1x1;
  *pool4_3x3_s2 >> *inception_5a_3x3_reduce;
  *inception_5a_3x3_reduce >> *inception_5a_3x3;
  *pool4_3x3_s2 >> *inception_5a_5x5_reduce;
  *inception_5a_5x5_reduce >> *inception_5a_5x5;
  *pool4_3x3_s2 >> *inception_5a_pool;
  *inception_5a_pool >> *inception_5a_pool_proj;
  vector<Blob*> {inception_5a_1x1->top()[0], inception_5a_3x3->top()[0], inception_5a_5x5->top()[0], inception_5a_pool_proj->top()[0], 
    inception_5a_1x1->top()[1], inception_5a_3x3->top()[1], inception_5a_5x5->top()[1], inception_5a_pool_proj->top()[1]} >> *inception_5a_output >> *act_of_inception_5a_output
  *inception_5a_1x1 >> *inception_5a_output;
  *inception_5a_3x3 >> *inception_5a_output;
  *inception_5a_5x5 >> *inception_5a_output;
  *inception_5a_pool_proj >> *inception_5a_output;
  *inception_5a_output >> *inception_5b_1x1;
  *inception_5a_output >> *inception_5b_3x3_reduce;
  *inception_5b_3x3_reduce >> *inception_5b_3x3;
  *inception_5a_output >> *inception_5b_5x5_reduce;
  *inception_5b_5x5_reduce >> *inception_5b_5x5;
  *inception_5a_output >> *inception_5b_pool;
  *inception_5b_pool >> *inception_5b_pool_proj;
  vector<Blob*> {inception_5b_1x1->top()[0], inception_5b_3x3->top()[0], inception_5b_5x5->top()[0], inception_5b_pool_proj->top()[0], 
    inception_5b_1x1->top()[1], inception_5b_3x3->top()[1], inception_5b_5x5->top()[1], inception_5b_pool_proj->top()[1]} >> *inception_5b_output >> *act_of_inception_5b_output
  *inception_5b_1x1 >> *inception_5b_output;
  *inception_5b_3x3 >> *inception_5b_output;
  *inception_5b_5x5 >> *inception_5b_output;
  *inception_5b_pool_proj >> *inception_5b_output;
  *inception_5b_output >> *pool5_7x7_s1;
  *pool5_7x7_s1 >> *pool5_drop_7x7_s1;
  *pool5_drop_7x7_s1 >> *loss3_classifier;
  loss3_loss3->set_label(label_);
  *loss3_classifier >> *loss3_loss3;
  loss3_top_1->set_label(label_);
  vector<Blob*>{ loss3_classifier->top()[0] } >> *loss3_top_1;
  loss3_top_5->set_label(label_);
  vector<Blob*>{ loss3_classifier->top()[0] } >> *loss3_top_5;


  loss_ = { loss3_loss3->loss()[0], loss3_top_5->loss()[0] };


  vector<Layer*> layers = { conv1_7x7_s2, conv2_3x3_reduce, conv2_3x3, inception_3a_1x1,
     inception_3a_3x3_reduce, inception_3a_3x3, inception_3a_5x5_reduce, inception_3a_5x5,
     inception_3a_pool_proj, inception_3b_1x1, inception_3b_3x3_reduce, inception_3b_3x3,
     inception_3b_5x5_reduce, inception_3b_5x5, inception_3b_pool_proj, inception_4a_1x1,
     inception_4a_3x3_reduce, inception_4a_3x3, inception_4a_5x5_reduce, inception_4a_5x5,
     inception_4a_pool_proj, loss1_conv, loss1_fc, loss1_classifier,
     inception_4b_1x1, inception_4b_3x3_reduce, inception_4b_3x3, inception_4b_5x5_reduce,
     inception_4b_5x5, inception_4b_pool_proj, inception_4c_1x1, inception_4c_3x3_reduce,
     inception_4c_3x3, inception_4c_5x5_reduce, inception_4c_5x5, inception_4c_pool_proj,
     inception_4d_1x1, inception_4d_3x3_reduce, inception_4d_3x3, inception_4d_5x5_reduce,
     inception_4d_5x5, inception_4d_pool_proj, loss2_conv, loss2_fc,
     loss2_classifier, inception_4e_1x1, inception_4e_3x3_reduce, inception_4e_3x3,
     inception_4e_5x5_reduce, inception_4e_5x5, inception_4e_pool_proj, inception_5a_1x1,
     inception_5a_3x3_reduce, inception_5a_3x3, inception_5a_5x5_reduce, inception_5a_5x5,
     inception_5a_pool_proj, inception_5b_1x1, inception_5b_3x3_reduce, inception_5b_3x3,
     inception_5b_5x5_reduce, inception_5b_5x5, inception_5b_pool_proj, loss3_classifier };

    

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
