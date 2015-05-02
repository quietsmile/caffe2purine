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
  VirtualLayerDetails;
  VirtualLayerRelations;
  VirtualLossLayer;
  VirtualLossInfo;
  VirtualWeightInfo;
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
