import os, sys
import caffe_pb2                                                                                             
import warnings
import math
from google.protobuf import text_format                                                                      

global_space = "  "
image_crop_size = 224
image_channel = 3
global_init_value = 0.01

all_layers = []

class layer:
    def __init__(self, type, name, bottom, top):
        self.type = type
        # this is because layer.name will be used as vairable names in purine!
        self.name = name.replace('/','_')
        self.name = self.name.replace('-','_')
        self.bottom = [i.replace('/','_') for i in bottom]
        self.bottom = [i.replace('-','_') for i in self.bottom]
        self.top = [i.replace('/','_') for i in top]
        self.top = [i.replace('-','_') for i in self.top]
        assert(self.type=='Concat' or self.type=='Accuracy' or self.type=='SoftmaxWithLoss' or len(self.bottom) <= 1)
        assert(self.type=='Data' or len(self.top) <= 1)

class data_layer(layer):
    def __init__(self, type, name, bottom, top, image_size=image_crop_size, channel=image_channel):
        layer.__init__(self, type, name, bottom, top)

        self.image_size = image_size
        #self.image_size = image_crop_size
        self.channel = channel
        # these three names are fixed
        self.name = 'data'
        self.diff_name = 'data_diff'
        self.label_name = 'label'
        self.in_size = [self.channel, self.image_size, self.image_size]
        self.out_size = [0,0,0]

    def update_out_size(self):
        self.out_size = self.in_size[:]

    def Print(self, fid):
        fid.write(global_space + '%s_ = create("%s", { %s, %d, %d, %d });\n' % (self.name, self.name, "batch_size", self.channel, self.image_size, self.image_size))
        fid.write(global_space + '%s_ = create("%s", { %s, %d, %d, %d });\n' % (self.diff_name, self.diff_name, "batch_size", self.channel, self.image_size, self.image_size))
        fid.write(global_space + '%s_ = create("%s", { %s, %d, %d, %d });\n' % (self.label_name, self.label_name, "batch_size",1,1,1))



class lrn_layer(layer):
    def __init__(self, type, name, bottom, top, local_size, alpha, beta):
        layer.__init__(self, type, name, bottom, top)
        self.local_size = local_size
        self.alpha = alpha
        self.beta = beta
        self.in_size = [0,0,0]
        self.out_size = [0,0,0]

    def update_out_size(self):
        self.out_size = self.in_size[:]

    def Print(self, fid):
        fid.write(global_space + 'LRNLayer* %s = createGraph<LRNLayer>("%s",' % (self.name, self.name))
        fid.write('\n')
        fid.write(global_space + global_space + 'LRNLayer::param_tuple(%f, %f, %d));' % (self.alpha, self.beta, 
                self.local_size))
        fid.write('\n')


class conv_layer(layer):
    def __init__(self, type, name, bottom, top, pad, stride, kernel, filters, nonlinear_type, 
            w_init_type, w_init_value, b_init_type, b_init_value, with_dropout=0, dropout_layer_name=""):
        layer.__init__(self, type, name, bottom, top)
        self.pad = pad
        self.stride = stride
        self.kernel = kernel
        self.filters = filters
        self.nonlinear_type = nonlinear_type
        self.w_init_type = w_init_type
        self.w_init_value = w_init_value
        self.b_init_type = b_init_type
        self.b_init_value = b_init_value
        self.with_dropout = with_dropout
        self.dropout_layer_name = dropout_layer_name
        self.in_size = [0,0,0]
        self.out_size = [0,0,0]
        
    def update_out_size(self):
        self.out_size[0] = self.filters
        self.out_size[1] = (self.in_size[1] + 2 * self.pad - self.kernel) / self.stride + 1
        self.out_size[2] = self.out_size[1]
        
    def Print(self, fid):
        fid.write(global_space + 'ConvLayer* %s = createGraph<ConvLayer>("%s",' % (self.name, self.name))
        fid.write('\n')
        fid.write(global_space + global_space + 'ConvLayer::param_tuple(%d, %d, %d, %d, %d, %d, %d, "%s"));' % (self.pad, self.pad, 
                self.stride, self.stride, self.kernel, self.kernel, self.filters, self.nonlinear_type))
        fid.write('\n')



class pool_layer(layer):
    def __init__(self, type, name, bottom, top, pooltype, kernel, stride, pad):
        layer.__init__(self, type, name, bottom, top)
        self.pooltype = pooltype
        self.pad = pad
        self.stride = stride
        self.kernel = kernel
        self.in_size = [0,0,0]
        self.out_size = [0,0,0]

    def update_out_size(self):
        self.out_size[0] = self.in_size[0]
        self.out_size[1] = int(math.ceil(float(self.in_size[1] + 2 * self.pad - self.kernel) / self.stride)) + 1
        if((self.out_size[1] - 1) * self.stride >= self.in_size[1] + self.pad):
            self.out_size[1] -= 1
        self.out_size[2] = self.out_size[1]

    def Print(self, fid):
        fid.write(global_space + 'PoolLayer* %s = createGraph<PoolLayer>("%s",' % (self.name, self.name))
        fid.write('\n')
        fid.write(global_space + global_space + 'PoolLayer::param_tuple("%s", %d, %d, %d, %d, %d, %d));' % (self.pooltype, self.kernel, self.kernel, self.stride, self.stride, self.pad, self.pad))
        fid.write('\n')




class fc_layer(layer):
    def __init__(self, type, name, bottom, top, hidden,
            w_init_type, w_init_value, b_init_type, b_init_value, with_dropout=0, dropout_layer_name=""):
        layer.__init__(self, type, name, bottom, top)
        self.hidden = hidden
        self.w_init_type = w_init_type
        self.w_init_value = w_init_value
        self.b_init_type = b_init_type
        self.b_init_value = b_init_value
        self.with_dropout = with_dropout
        self.dropout_layer_name = dropout_layer_name
        self.in_size = [0,0,0]
        self.out_size = [0,0,0]

    def update_out_size(self):
        self.out_size = [self.hidden, 1, 1]

    def Print(self,fid):
        fid.write(global_space + 'InnerProdLayer* %s = createGraph<InnerProdLayer>("%s",' % (self.name, self.name))
        fid.write('\n')
        fid.write(global_space + global_space + 'InnerProdLayer::param_tuple(%d, ""));' % self.hidden)
        fid.write('\n')


class dropout_layer(layer):
    def __init__(self, type, name, bottom, top, ratio):
        layer.__init__(self, type, name, bottom, top)
        self.ratio = ratio
        self.in_size = [0,0,0]
        self.out_size = [0,0,0]

    def update_out_size(self):
        self.out_size = self.in_size[:]

    def Print(self, fid):
        fid.write(global_space + 'DropoutLayer* %s = createGraph<DropoutLayer>("%s",' % (self.name, self.name))
        fid.write('\n')
        fid.write(global_space + global_space + 'DropoutLayer::param_tuple(%f, test, false));' % self.ratio)
        fid.write('\n')

class concat_layer(layer):
    def __init__(self, type, name, bottom, top):
        layer.__init__(self, type, name, bottom, top)
        self.in_size = [0,0,0]
        self.out_size = [0,0,0]
        self.act_name = 'act_of_' + self.name

    def update_out_size(self):
        self.out_size = self.in_size[:]
    def Print(self, fid):
        # maybe the dic should be global
        fid.write(global_space + 'ConcatLayer* %s = createGraph<ConcatLayer>("%s",\n' % (self.name, self.name))
        fid.write(global_space + global_space + 'ConcatLayer::param_tuple(Split::CHANNELS));\n')
        #fid.write(global_space + 'ActivationLayer* %s = createGraph<ActivationLayer>("%s",\n' % (self.act_name, self.act_name))
        #fid.write(global_space + global_space + 'ActivationLayer::param_tuple("relu", true));\n')


class softmaxloss_layer(layer):
    def __init__(self, type, name, bottom, top):
        layer.__init__(self, type, name, bottom, top)

    def Print(self, fid):
        fid.write(global_space + 'SoftmaxLossLayer* %s = createGraph<SoftmaxLossLayer>("%s",' % (self.name, self.name))
        fid.write('\n')
        fid.write(global_space + global_space + 'SoftmaxLossLayer::param_tuple(1.));')
        fid.write('\n')

class accuracy_layer(layer):
    def __init__(self, type, name, bottom, top):
        layer.__init__(self, type, name, bottom, top)

    def Print(self, fid):
        fid.write(global_space + 'Acc* %s = createGraph<Acc>("%s", rank_, -1, Acc::param_tuple(1));' % (self.name, self.name))
        fid.write('\n')


def deal_with_layer(layer):
    global all_layers
    
    if (layer.type == 'Data'):
        if (layer.include[0].phase == 0): # 0 = train, 1 = test
            data_source = layer.data_param.source
            warnings.warn('Data source should be added manually, only this code only construct the model structure!')
            if(hasattr(layer.transform_param, 'crop_size')):
                data = data_layer(layer.type, layer.name, [], layer.top,
                        layer.transform_param.crop_size)
            else:
                data = data_layer(layer.type, layer.name, [], layer.top)
            all_layers.append(data)
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
            if hasattr(layer.convolution_param.weight_filler, 'std'):
                w_init_value = layer.convolution_param.weight_filler.std
            else:
                w_init_value = global_init_value
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

        
        if hasattr(layer.convolution_param,'group'):
            if(layer.convolution_param.group >= 2):
                warnings.warn('group >= 2 is not supported right now! It is automatically changed to 1!')

        #Nowadays, relu is dominant in CNN
        neuron = 'relu'
        conv = conv_layer(layer.type, layer.name, layer.bottom, layer.top,
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
        all_layers.append(conv)

    elif(layer.type == 'LRN'):
        lrn = lrn_layer(layer.type, layer.name, layer.bottom, layer.top,
                layer.lrn_param.local_size,
                layer.lrn_param.alpha,
                layer.lrn_param.beta)
        all_layers.append(lrn)


    elif(layer.type == 'Pooling'):
        # 0 == MAX  1 == AVG
        print 'Pooling type', layer.pooling_param.pool
        if layer.pooling_param.pool == 0:
            pool_strategy = 'max'
        elif layer.pooling_param.pool == 1:
            pool_strategy = 'average'
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

        pool = pool_layer(layer.type, layer.name, layer.bottom, layer.top,
                pool_strategy,
                kernel_size,
                stride,
                pad
                )
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
            if hasattr(layer.inner_product_param.weight_filler, 'std'):
                w_init_value = layer.inner_product_param.weight_filler.std
            else:
                w_init_value = global_init_value
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
        fc = fc_layer(layer.type, layer.name, layer.bottom, layer.top,
                hidden,
                w_init_type,
                w_init_value,
                b_init_type,
                b_init_value
                )
        all_layers.append(fc)


    elif(layer.type == 'Dropout'):
        if hasattr(layer.dropout_param, 'dropout_ratio'):
            # todo: check the ratio is drop ratio or keep ratio
            # this version just assumes that it is the same with caffe
            dropout_ratio = layer.dropout_param.dropout_ratio
        else:
            dropout_ratio = 0.5 

        dropout = dropout_layer(layer.type, layer.name, layer.bottom, layer.top,
                dropout_ratio
                )
        all_layers.append(dropout)

    elif(layer.type == 'Concat'):
        concat = concat_layer(layer.type, layer.name, layer.bottom, layer.top)
        all_layers.append(concat)
    elif(layer.type == 'SoftmaxWithLoss'):
        softmax = softmaxloss_layer(layer.type, layer.name, layer.bottom, layer.top)
        all_layers.append(softmax)

    elif(layer.type == 'Accuracy'):
        acc = accuracy_layer(layer.type, layer.name, layer.bottom, layer.top)
        all_layers.append(acc)


def calculateNodePairs(nodePair):
    ret = ''
    for pair in nodePair:
        rank, device = pair
        ret += 'parallels.push_back({%d, %d});\n' % (rank, device)
    return ret


def calculateInAndOutSize():

    dic = {}
    for layer in all_layers:
        dic[layer.name] = layer

    for i in range(len(all_layers)):
        for layer in all_layers:
            if(not hasattr(layer, 'in_size')):
                continue
            if(reduce(lambda x,y:x*y, layer.out_size) != 0):
                continue
            collected = 1
            for sub in layer.bottom:
                if(reduce(lambda x,y:x*y, dic[sub].out_size) == 0):
                    collected = 0
                    break
            if(collected):
                if len(layer.bottom):
                    common_size = dic[layer.bottom[0]].out_size[:]
                    common_size[0] = 0
                    for sub in layer.bottom:
                        assert(common_size[1] == dic[sub].out_size[1] and common_size[2] == dic[sub].out_size[2])
                        common_size[0] += dic[sub].out_size[0]
                    layer.in_size = common_size[:]
                layer.update_out_size()
                print layer.name, layer.out_size



def run(argv):

#----------------------------------- resolve model -----------------------------------
    net = caffe_pb2.NetParameter()                                                                                 
    text_format.Parse(open(argv[1]).read(), net)
    for layer in net.layer:
        deal_with_layer(layer)
    calculateInAndOutSize()

#----------------------- output information to separate files -----------------------
    folder = 'tmp/'
    fid = open(folder + 'layer_details.txt','w')
    for layer in all_layers:
        layer.Print(fid)
    fid.close()


    fid = open(folder + 'layer_relations.txt','w')
    # first change the bottom of a layer which is after a layer with dropout
    dic = {}
    for layer in all_layers:
        dic[layer.name] = layer

    for layer in all_layers:
        if(layer.type=='SoftmaxWithLoss'):
            fid.write(global_space + '%s->set_label(%s);\n' % (layer.name, 'label_'))
            fid.write(global_space + '*%s >> *%s;\n' % (layer.bottom[0], layer.name))
            continue

        if(layer.type=='Accuracy'):
            fid.write(global_space + '%s->set_label(%s);\n' % (layer.name, 'label_'))
            fid.write(global_space + 'vector<Blob*>{ %s->top()[0] } >> *%s;\n' % (layer.bottom[0], layer.name))
            continue

        if(layer.type=='Concat'):
            tmp = global_space + 'vector<Blob*> {'
            for j in layer.bottom:
                tmp += '%s->top()[0], ' % j
            tmp += '\n' + global_space + global_space
            for j in layer.bottom:
                tmp += '%s->top()[1], ' % j
            tmp = tmp[:-2]
            #tmp += '} >> *%s >> *%s' % (layer.name, layer.act_name)
            tmp += '} >> *%s;' % (layer.name)
            fid.write(tmp+'\n')
            continue

        for j in layer.bottom:
            if(dic[j].type=='Data'):
                fid.write(global_space + '%s >> *%s;\n' % ('B{ data_, data_diff_}', layer.name))
                continue
            if( hasattr(dic[j], 'with_dropout')):
                if(dic[j].with_dropout):
                    fid.write(global_space + '*%s >> *%s;\n' % (dic[j].dropout_layer_name, layer.name))
                else:
                    fid.write(global_space + '*%s >> *%s;\n' % (j, layer.name))
            else:
                fid.write(global_space + '*%s >> *%s;\n' % (j, layer.name))

        if(layer.type=='Dropout'):
            assert(layer.bottom[0] == layer.top[0] and len(layer.top) == 1 and len(layer.bottom) == 1)
            dic[layer.bottom[0]].with_dropout = 1
            dic[layer.bottom[0]].dropout_layer_name = layer.name
    fid.close()


    fid = open(folder + 'loss_info.txt','w')
    softmaxloss_layer_name = ""
    accuracy_layer_name = ""
    for layer in all_layers:
        if(layer.type=='SoftmaxWithLoss'):
            softmaxloss_layer_name = layer.name
        if(layer.type=='Accuracy'):
            accuracy_layer_name = layer.name
    fid.write(global_space + '%s = { %s->loss()[0], %s->loss()[0] };\n' % ('loss_', softmaxloss_layer_name, accuracy_layer_name))
    fid.close()


    layer_with_weights = [layer.name for layer in all_layers if layer.type.startswith('Conv') or layer.type.startswith('Inner')]
    fid = open(folder + 'weight_info.txt','w')
    tmp = global_space + 'vector<Layer*> layers = {'
    cnt = 0
    for layername in layer_with_weights:
        cnt += 1
        tmp += ' '+layername
        if(cnt != len(layer_with_weights)):
            tmp += ','
        else:
            tmp += ' };\n'
        if(cnt % 4 == 0):
            tmp += '\n' + global_space + global_space
    fid.write(tmp)
    fid.close()


    # calculate weights of each layer, support xavier and gaussian


    fid = open(folder + 'weight_initialization.txt', 'w')
    index = 0
    for layername in layer_with_weights:
        if(dic[layername].w_init_type == 'gaussian'):
            tmp = global_space + global_space + 'parallel_net->init<Gaussian>({%d}, Gaussian::param_tuple(0., %f));' % (index, dic[layername].w_init_value)
        elif(dic[layername].w_init_type == 'xavier'):
            scale = math.sqrt(3.0 / reduce(lambda x,y:x*y, dic[layername].in_size))
            tmp = global_space + global_space + 'parallel_net->init<Uniform>({%d}, Uniform::param_tuple(-%f, %f));' % (index, scale, scale)
        fid.write(tmp + '\n')
        index += 1
        tmp = global_space + global_space + 'parallel_net->init<Constant>({%d}, Constant::param_tuple(%f));' % (index, dic[layername].b_init_value)
        fid.write(tmp + '\n')
        index += 1
    fid.close()
        


#----------------------------------- all parameters ----------------------------------
    infoVirtualNetName = 'NewModel'

    # How to construct each layer
    infoVirtualLayerDetails = ''.join(open(folder + 'layer_details.txt').readlines())
    # How to construct layer interconnections
    infoVirtualLayerRelations = ''.join(open(folder + 'layer_relations.txt').readlines())
    # How to set loss layer
    infoVirtualLossInfo = ''.join(open(folder + 'loss_info.txt').readlines())
    # Layers with weight (Convolution and InnerProductLayer)
    infoVirtualWeightInfo = ''.join(open(folder + 'weight_info.txt').readlines())

    infoVirtualBatchSizePerNode = str(64)
    infoVirtualDataPath = '"../data/bowl_train_lmdb"'
    infoVirtualDataMean = '"../data/bowl_mean.binaryproto"';
    infoVirtualWeightNum = str(len(layer_with_weights) * 2)
    infoVirtualImageSize = str(224)
    infoVirtualWeightInitialization = ''.join(open(folder + 'weight_initialization.txt').readlines())
    infoVirtualAvailabelNodes = calculateNodePairs([[0,3]])
#-------------------------------------   combine information   -----------------------

    infoVirtualNethpp = ''.join(open('VirtualNet.hpp').readlines())
    infoVirtualNethpp = infoVirtualNethpp.replace('VirtualLayerDetails;',infoVirtualLayerDetails)
    infoVirtualNethpp = infoVirtualNethpp.replace('VirtualLayerRelations;',infoVirtualLayerRelations)
    infoVirtualNethpp = infoVirtualNethpp.replace('VirtualLossInfo;',infoVirtualLossInfo)
    infoVirtualNethpp = infoVirtualNethpp.replace('VirtualWeightInfo;',infoVirtualWeightInfo)
    infoVirtualNethpp = infoVirtualNethpp.replace('VirtualNetName;',infoVirtualNetName)
    fid = open('%s.hpp' % infoVirtualNetName,'w')
    fid.write(infoVirtualNethpp)
    fid.close()

    infoVirtualNetcpp = ''.join(open('VirtualNet.cpp').readlines())
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualModel',infoVirtualNetName)
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualBatchSizePerNode;',infoVirtualBatchSizePerNode+';')
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualDataPath;',infoVirtualDataPath+';')
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualDataMean;',infoVirtualDataMean+';')
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualWeightNum;',infoVirtualWeightNum+';')
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualNetName;',infoVirtualNetName)
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualImageSize;',infoVirtualImageSize+';')
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualWeightInitialization;',infoVirtualWeightInitialization)
    infoVirtualNetcpp = infoVirtualNetcpp.replace('VirtualAvailableNodes;',infoVirtualAvailabelNodes)
    fid = open('%s.cpp' % infoVirtualNetName,'w')
    fid.write(infoVirtualNetcpp)
    fid.close()
    

if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print 'Usage: python read_caffe_cfg.py caffe_model_name'
        print 'Example: python read_caffe_cfg.py ../../../../models/bvlc_alexnet/train_val.prototxt'
        exit(0)

    run(sys.argv)

