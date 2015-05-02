import os, sys
import caffe_pb2                                                                                             
import warnings
from google.protobuf import text_format                                                                      

global_space = "  "
image_crop_size = 224
image_channel = 3
global_init_value = 0.01

conv_cnt = 0
pool_cnt = 0
fc_cnt = 0
dropout_cnt = 0
all_layers = []

class layer:
    def __init__(self, type, name):
        self.type = type
        self.name = name

class data_layer(layer):
    def __init__(self, type, name, image_size=image_crop_size, channel=image_channel):
        layer.__init__(self, type, name)
        self.image_size = image_size
        self.channel = channel
    def Print(self):
        print global_space + '%s = create("%s", { %s, %d, %d, %d });' % (self.name, self.name, "batch_size",
                self.channel, self.image_size, self.image_size)


class conv_layer(layer):
    def __init__(self, type, name, pad, stride, kernel, filter, nonlinear_type, 
            w_init_type, w_init_value, b_init_type, b_init_value):
        layer.__init__(self, type, name)
        self.pad = pad
        self.stride = stride
        self.kernel = kernel
        self.filter = filter
        self.nonlinear_type = nonlinear_type
        self.w_init_type = w_init_type
        self.w_init_value = w_init_value
        self.b_init_type = b_init_type
        self.b_init_value = b_init_value
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
    def __init__(self, name, var_name, hidden,
            w_init_type, w_init_value, b_init_type, b_init_value):
        layer.__init__(self, name)
        self.hidden = hidden
        self.var_name = var_name
        self.w_init_type = w_init_type
        self.w_init_value = w_init_value
        self.b_init_type = b_init_type
        self.b_init_value = b_init_value
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



#
#for i in s:
#    if(cf.get(i, 'type')=='conv'):
#        all_layers.append('conv%d' % conv_cnt)
#        main_layers.append('conv%d' % conv_cnt)
#        conv = conv_layer('conv%d' % conv_cnt, 'conv%d' % conv_cnt, 
#                cf.getint(i, 'padding') if cf.has_option(i, 'padding') else 0, 
#                cf.getint(i, 'stride') if cf.has_option(i, 'stride') else 1, 
#                cf.getint(i, 'filterSize'),
#                cf.getint(i, 'filters'),
#                cf.get(i, 'neuron') if cf.has_option(i, 'neuron') else 'relu'
#                )
#        conv.Print()
#        conv_cnt += 1
#
#    elif(cf.get(i, 'type')=='pool'):
#        all_layers.append('pool%d' % pool_cnt)
#        pool_strategy = cf.get(i, 'pool') if cf.has_option(i, 'pool') else 'max'
#        pool = pool_layer('%s_pool%d' % (pool_strategy, pool_cnt), 'pool%d' % pool_cnt, 
#                pool_strategy,
#                cf.getint(i, 'sizeX'),
#                cf.getint(i, 'stride') if cf.has_option(i, 'stride') else 1, 
#                cf.getint(i, 'pad') if cf.has_option(i, 'pad') else 0
#                )
#        pool.Print()
#        pool_cnt += 1
#
#    elif(cf.get(i, 'type')=='fc'):
#        all_layers.append('inner%d' % fc_cnt)
#        main_layers.append('inner%d' % fc_cnt)
#        fc = fc_layer('inner%d' % fc_cnt, 'inner%d' % fc_cnt, 
#                cf.getint(i, 'outputs')
#                )
#        fc.Print()
#        fc_cnt += 1
#
#    elif(cf.get(i, 'type').startswith('dropout')): # todo modify others
#        all_layers.append('dropout%d' % dropout_cnt)
#        dropout = dropout_layer('dropout%d' % dropout_cnt, 'dropout%d' % dropout_cnt, 
#                0.5
#                )
#        dropout.Print()
#        dropout_cnt += 1
#
#Print_last_layers()
#Print_connections(all_layers, main_layers)
#Print_tail()


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


def deal_with_layer(layer):
    global conv_cnt
    global fc_cnt
    global pool_cnt
    global dropout_cnt
    global all_layers

    if (layer.type == 'Data'):
        if (layer.include[0].phase == 0): # 0 = train, 1 = test
            data_source = layer.data_param.source
            warnings.warn('Data source should be added manually, only the model itself is considered here!')
    elif(layer.type == 'Convolution'):
        if hasattr(layer.convolution_param, 'num_output'):
            filters = layer.convolution_param.num_output
        else:
            raise Exception('Bad_Model_File')

        if hasattr(layer.convolution_param, 'kernel_size'):
            filterSize = layer.convolution_param.kernel_size
        else:
            raise Exception('Bad_Model_File')

        if hasattr(layer.convolution_param, 'stride'):
            stride = layer.convolution_param.stride
        else:
            stride = 1

        if hasattr(layer.convolution_param, 'pad'):
            pad = layer.convolution_param.pad
        else:
            pad = 0

        # when type is xavier, std is useless
        print layer.convolution_param.weight_filler.type
        if hasattr(layer.convolution_param, 'weight_filler'):
            w_init_type = layer.convolution_param.weight_filler.type
            w_init_value = layer.convolution_param.weight_filler.std
        else:
            w_init_type = 'guassian'
            w_init_value = global_init_value

        if hasattr(layer.convolution_param, 'bias_filler'):
            assert(layer.convolution_param.bias_filler.type == 'constant')
            b_init_type = layer.convolution_param.bias_filler.type
            b_init_value = layer.convolution_param.bias_filler.value
        else:
            b_init_type = 'constant'
            b_init_value = 0


        # Simply give an initialization here, might be changed later according to the next layer
        neuron = 'relu'
        conv = conv_layer('conv%d' % conv_cnt, 'conv%d' % conv_cnt, 
                pad,
                stride,
                filterSize,
                filters,
                neuron,
                w_init_type,
                w_init_value,
                b_init_type,
                b_init_value
                )
        conv_cnt += 1
        all_layers.append(conv)

    elif(layer.type == 'LRN'):
        warnings.warn('LRN layer is not supported! It is automatically removed from the model!')
    elif(layer.type == 'Pooling'):
        # 0 == MAX  1 == AVG
        print 'Pooling type', layer.pooling_param.pool
        if layer.pooling_param.pool == 0:
            pool_strategy = 'max'
        elif layer.pooling_param.pool == 1:
            pool_strategy = 'avg'
        else:
            unrecognized_pooling_strategy

        kernel_size =  layer.pooling_param.kernel_size
        if hasattr(layer.pooling_param, 'stride'):
            stride = layer.pooling_param.stride 
        else:
            stride = 1

        if hasattr(layer.pooling_param, 'pad'):
            pad = layer.pooling_param.pad 
        else:
            pad = 0

        pool = pool_layer('%s_pool%d' % (pool_strategy, pool_cnt), 'pool%d' % pool_cnt, 
                pool_strategy,
                kernel_size,
                stride,
                pad
                )
        pool_cnt += 1
        all_layers.append(pool)

    elif(layer.type == 'InnerProduct'):
        print layer.inner_product_param.num_output

        if hasattr(layer.inner_product_param, 'num_output'):
            hidden = layer.inner_product_param.num_output
        else:
            raise Exception('Bad_Model_File')

        # when type is xavier, std is useless
        if hasattr(layer.inner_product_param, 'weight_filler'):
            w_init_type = layer.inner_product_param.weight_filler.type
            w_init_value = layer.inner_product_param.weight_filler.std
        else:
            w_init_type = 'guassian'
            w_init_value = global_init_value

        if hasattr(layer.inner_product_param, 'bias_filler'):
            assert(layer.inner_product_param.bias_filler.type == 'constant')
            b_init_type = layer.inner_product_param.bias_filler.type
            b_init_value = layer.inner_product_param.bias_filler.value
        else:
            b_init_type = 'constant'
            b_init_value = 0

        # Simply give an initialization here, might be changed later according to the next layer
        fc = fc_layer('inner%d' % fc_cnt, 'inner%d' % fc_cnt, 
                hidden,
                w_init_type,
                w_init_value,
                b_init_type,
                b_init_value
                )
        fc_cnt += 1
        all_layers.append(fc)


    elif(layer.type == 'Dropout'):
        if hasattr(layer.dropout_param, 'dropout_ratio'):
            dropout_ratio = layer.dropout_param.dropout_ratio
        else:
            dropout_ratio = 0.5

        dropout = dropout_layer('dropout%d' % dropout_cnt, 'dropout%d' % dropout_cnt, 
                dropout_ratio
                )
        dropout_cnt += 1
        all_layers.append(dropout)

    elif(layer.type == 'Concat'):
        warnings.warn('Concat layer is not supported! This code is not working properly with this warning!')
        raise Exception('Concat Layer is Not Supported')
    elif(layer.type == 'SoftmaxWithLoss'):
        pass 
        #print layer.name
    elif(layer.type == 'Accuracy'):
        pass
        #print layer.name


def calculateNodePairs(nodePair):
    ret = ''
    for pair in nodePair:
        rank, device = pair
        ret += 'parallels.push_back({%d, %d});\n' % (rank, device)
    return ret

def run(argv):

#----------------------------------- all parameters ----------------------------------
    infoVirtualNetName = 'NewNet'

    # How to construct each layer
    infoVirtualLayerDetails = ''
    # How to construct layer interconnections
    infoVirtualLayerRelations = ''
    # How to set loss layer
    infoVirtualLossLayer = ''
    infoVirtualLossInfo = ''
    # Layers with weight (Convolution and InnerProductLayer)
    infoVirtualWeightInfo = []

    infoVirtualBatchSizePerNode = str(64)
    infoVirtualDataPath = '"../data/bowl_train_lmdb"'
    infoVirtualDataMean = '"../data/bowl_mean.binaryproto"';
    infoVirtualWeightNum = ''
    infoVirtualImageSize = str(224)
    infoVirtualWeightInitialization = ''
    infoVirtualAvailabelNodes = calculateNodePairs([[0,3]])
#----------------------------------- resolve model -----------------------------------
    net = caffe_pb2.NetParameter()                                                                                 
    text_format.Parse(open(argv[1]).read(), net)
    for layer in net.layer:
        deal_with_layer(layer)

    get_layer_by_name 
    infoVirtualWeightInfo = [layer.var_name for layer in all_layers if layer.type_name.startswith('conv') or layer.type_name.startswith('inner')]

    print infoVirtualWeightInfo









#-------------------------------------   output   ------------------------------------
    infoVirtualWeightInfo = str(len(infoVirtualWeightInfo))

    infoVirtualNethpp = ''.join(open('VirtualNet.hpp').readlines())
    infoVirtualNethpp = infoVirtualNethpp.replace('VirtualLayerDetails;',infoVirtualLayerDetails)
    infoVirtualNethpp = infoVirtualNethpp.replace('VirtualLayerRelations;',infoVirtualLayerRelations)
    infoVirtualNethpp = infoVirtualNethpp.replace('VirtualLossLayer;',infoVirtualLossLayer)
    infoVirtualNethpp = infoVirtualNethpp.replace('VirtualLossInfo;',infoVirtualLossInfo)
    infoVirtualNethpp = infoVirtualNethpp.replace('VirtualWeightInfo;',infoVirtualWeightInfo)
    infoVirtualNethpp = infoVirtualNethpp.replace('VirtualNetName;',infoVirtualNetName)
    fid = open('NewModel.hpp','w')
    fid.write(infoVirtualNethpp)
    fid.close()

    infoVirtualNetcpp = ''.join(open('VirtualNet.cpp').readlines())
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualBatchSizePerNode;',infoVirtualBatchSizePerNode)
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualDataPath;',infoVirtualDataPath)
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualDataMean;',infoVirtualDataMean)
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualWeightNum;',infoVirtualWeightNum)
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualNetName;',infoVirtualNetName)
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualImageSize;',infoVirtualImageSize)
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualWeightInitialization;',infoVirtualWeightInitialization)
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualAvailableNodes;',infoVirtualAvailabelNodes)
    fid = open('NewModel.cpp','w')
    fid.write(infoVirtualNetcpp)
    fid.close()
    

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print 'Usage: python read_caffe_cfg.py caffe_model_name'
        print 'Example: python read_caffe_cfg.py ../../../models/bvlc_alexnet/train_val.prototxt'
        exit(0)

    run(sys.argv)
    print all_layers

