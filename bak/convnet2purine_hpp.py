import ConfigParser,string,os,sys    
cf = ConfigParser.ConfigParser()    
cf.read("layers-imagenet-1gpu.cfg")
s = cf.sections()


global_space = "  "
class layer:
    def __init__(self, name):
        self.type_name = name

class data_layer(layer):
    def __init__(self, name, var_name, image_size=224, channel=3):
        layer.__init__(self, name)
        self.var_name = var_name
        self.image_size = image_size
        self.channel = channel
    def Print(self):
        print global_space + '%s = create("%s", { %s, %d, %d, %d });' % (self.var_name, self.type_name, "batch_size",
                self.channel, self.image_size, self.image_size)


class conv_layer(layer):
    def __init__(self, name, var_name, pad, stride, kernel, filter, nonlinear_type):
        layer.__init__(self, name)
        self.var_name = var_name
        self.pad = pad
        self.stride = stride
        self.kernel = kernel
        self.filter = filter
        self.nonlinear_type = nonlinear_type
    def Print(self):
        print global_space + 'ConvLayer* %s = createGraph<ConvLayer>("%s",' % (self.var_name, self.type_name)
        print global_space + global_space + 'ConvLayer::param_tuple(%d, %d, %d, %d, %d, %d, %d, "%s"));' % (self.pad, self.pad, 
                self.stride, self.stride, self.kernel, self.kernel, self.filter, self.nonlinear_type)



class pool_layer(layer):
    def __init__(self, name, var_name, pooltype, kernel, stride, pad):
        layer.__init__(self, name)
        self.var_name = var_name
        self.pooltype = pooltype
        self.pad = pad
        self.stride = stride
        self.kernel = kernel
    def Print(self):
        print global_space + 'PoolLayer* %s = createGraph<PoolLayer>("%s",' % (self.var_name, self.type_name)
        print global_space + global_space + 'PoolLayer::param_tuple("%s", %d, %d, %d, %d, %d, %d));' % (self.pooltype, self.kernel, self.kernel, self.stride, self.stride, self.pad, self.pad)



class fc_layer(layer):
    def __init__(self, name, var_name, hidden):
        layer.__init__(self, name)
        self.hidden = hidden
        self.var_name = var_name
    def Print(self):
        print global_space + 'InnerProdLayer* %s = createGraph<InnerProdLayer>("%s",' % (self.var_name, self.type_name)
        print global_space + global_space + 'InnerProdLayer::param_tuple(%d, ""));' % self.hidden


class dropout_layer(layer):
    def __init__(self, name, var_name, ratio):
        layer.__init__(self, name)
        self.ratio = ratio
        self.var_name = var_name
    def Print(self):
        print global_space + 'DropoutLayer* %s = createGraph<DropoutLayer>("%s",' % (self.var_name, self.type_name)
        print global_space + global_space + 'DropoutLayer::param_tuple(%f, test, false));' % self.ratio

class cost_layer(layer):
    def __init__(self, name):
        layer.__init__(self, name)



def Print_header():
    print '''
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
class GoogLeNet : public Graph {
 protected:
  Blob* data_;
  Blob* label_;
  Blob* data_diff_;
  vector<Blob*> weights_;
  vector<Blob*> weight_data_;
  vector<Blob*> weight_diff_;
  vector<Blob*> loss_;
 public:
  explicit GoogLeNet(int rank, int device);
  virtual ~GoogLeNet() override {}
  inline const vector<Blob*>& weight_data() { return weight_data_; }
  inline const vector<Blob*>& weight_diff() { return weight_diff_; }
  inline vector<Blob*> data() { return { data_ }; }
  inline vector<Blob*> label() { return { label_ }; }
  inline vector<Blob*> data_diff() { return { data_diff_ }; }
  inline vector<Blob*> loss() { return loss_; }
};

template <bool test>
GoogLeNet<test>::GoogLeNet(int rank, int device) : Graph(rank, device) {
'''

def Print_tail():
    print '''
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
'''



def Print_last_layers():

    print global_space + 'SoftmaxLossLayer* softmaxloss = createGraph<SoftmaxLossLayer>("softmaxloss",'
    print global_space + global_space + 'SoftmaxLossLayer::param_tuple(1.));'

    print global_space + 'Acc* acc = createGraph<Acc>("acc", rank_, -1, Acc::param_tuple(1));'

def Print_connections(all_layers, main_layers):
    print global_space + 'B{ data_, data_diff_ } >> '
    for id, item in enumerate(all_layers):
        if( id + 1 != len(all_layers)):
            print global_space + '*%s' % item + ' >>'
        else:
            print global_space + '*%s' % item + ';'
            print global_space + 'softmaxloss->set_label(label_);'
            print global_space + '*%s >> *softmaxloss' % item 
            print global_space + 'acc->set_label(label_)'
            print global_space + 'vector<Blob*>{ %s->top()[0]} >> *acc;' % item
            print global_space + 'loss_ = { softmaxloss->loss()[0], acc->loss()[0] };'
            print global_space + 'vector<layer*> layers = { %s };' % ','.join(main_layers)




Print_header()

conv_cnt = 0
pool_cnt = 0
dropout_cnt = 0
fc_cnt = 0

all_layers = []
main_layers = []
for i in s:
    if(cf.get(i, 'type')=='conv'):
        all_layers.append('conv%d' % conv_cnt)
        main_layers.append('conv%d' % conv_cnt)
        conv = conv_layer('conv%d' % conv_cnt, 'conv%d' % conv_cnt, 
                cf.getint(i, 'padding') if cf.has_option(i, 'padding') else 0, 
                cf.getint(i, 'stride') if cf.has_option(i, 'stride') else 1, 
                cf.getint(i, 'filterSize'),
                cf.getint(i, 'filters'),
                cf.get(i, 'neuron') if cf.has_option(i, 'neuron') else 'relu'
                )
        conv.Print()
        conv_cnt += 1

    elif(cf.get(i, 'type')=='pool'):
        all_layers.append('pool%d' % pool_cnt)
        pool_strategy = cf.get(i, 'pool') if cf.has_option(i, 'pool') else 'max'
        pool = pool_layer('%s_pool%d' % (pool_strategy, pool_cnt), 'pool%d' % pool_cnt, 
                pool_strategy,
                cf.getint(i, 'sizeX'),
                cf.getint(i, 'stride') if cf.has_option(i, 'stride') else 1, 
                cf.getint(i, 'pad') if cf.has_option(i, 'pad') else 0
                )
        pool.Print()
        pool_cnt += 1

    elif(cf.get(i, 'type')=='fc'):
        all_layers.append('inner%d' % fc_cnt)
        main_layers.append('inner%d' % fc_cnt)
        fc = fc_layer('inner%d' % fc_cnt, 'inner%d' % fc_cnt, 
                cf.getint(i, 'outputs')
                )
        fc.Print()
        fc_cnt += 1

    elif(cf.get(i, 'type').startswith('dropout')): # todo modify others
        all_layers.append('dropout%d' % dropout_cnt)
        dropout = dropout_layer('dropout%d' % dropout_cnt, 'dropout%d' % dropout_cnt, 
                0.5
                )
        dropout.Print()
        dropout_cnt += 1

Print_last_layers()
Print_connections(all_layers, main_layers)
Print_tail()


'''
data = data_layer("data", "data_")
data.Print()
data_diff = data_layer("data_diff", "data_diff_")
data_diff.Print()
label = data_layer("label", "label_", 1, 1)
label.Print()
conv = conv_layer('conv1', 'conv1', 3, 2, 7, 64, 'relu')
conv.Print()
pool = pool_layer('max_pool1', 'pool1', 'max', 3, 2, 0)
pool.Print()
fc = fc_layer('inner', 'inner', 121)
fc.Print()
dropout = dropout_layer('dropout', 'dropout', 0.4)
dropout.Print()
'''
