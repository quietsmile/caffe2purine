// Copyright Lin Min 2015

#include <mpi.h>
#include <glog/logging.h>
#include "examples/NewModel.hpp"
#include "composite/graph/all_reduce.hpp"

#include<iostream>
using namespace std;
int batch_size = 64;
string source = "../data/bowl_train_lmdb";
string mean_file = "../data/bowl_mean.binaryproto";
const int weight_num = 16;
const int image_size = 224;

using namespace purine;

void setup_param_server(DataParallel<VirtualNetName<false>, AllReduce>*
    parallel_net, DTYPE global_learning_rate) {

  MPI_LOG(<< "hello_setup");
  // set learning rate etc
  DTYPE global_decay = 0.0001;
  vector<AllReduce::param_tuple> param(weight_num);
  for (int i = 0; i < weight_num; ++i) {
    DTYPE learning_rate = global_learning_rate * (i % 2 ? 2. : 1.);
    param[i] = AllReduce::param_tuple(0.9, learning_rate,
        learning_rate * global_decay * (i % 2 ? 0. : 1.));
  }
  parallel_net->setup_param_server(vector<int>(weight_num, 0),
      vector<int>(weight_num, -1), param);
}

void initialize(DataParallel<VirtualNetName<false>, AllReduce>* parallel_net,
                const string& snapshot) {

  MPI_LOG(<< "hello_init");
  if (snapshot == "") {
    vector<int> indice(weight_num/2);
    iota(indice.begin(), indice.end(), 0);
    vector<int> weight_indice(weight_num/2);
    vector<int> bias_indice(weight_num/2);
    transform(indice.begin(), indice.end(), weight_indice.begin(),
        [](int i)->int {
          return i * 2;
        });
    transform(indice.begin(), indice.end(), bias_indice.begin(),
        [](int i)->int {
          return i * 2 + 1;
        });
    parallel_net->init<Gaussian>({0}, Gaussian::param_tuple(0., 0.010000));
    parallel_net->init<Constant>({1}, Constant::param_tuple(0.000000));
    parallel_net->init<Gaussian>({2}, Gaussian::param_tuple(0., 0.010000));
    parallel_net->init<Constant>({3}, Constant::param_tuple(0.100000));
    parallel_net->init<Gaussian>({4}, Gaussian::param_tuple(0., 0.010000));
    parallel_net->init<Constant>({5}, Constant::param_tuple(0.000000));
    parallel_net->init<Gaussian>({6}, Gaussian::param_tuple(0., 0.010000));
    parallel_net->init<Constant>({7}, Constant::param_tuple(0.100000));
    parallel_net->init<Gaussian>({8}, Gaussian::param_tuple(0., 0.010000));
    parallel_net->init<Constant>({9}, Constant::param_tuple(0.100000));
    parallel_net->init<Gaussian>({10}, Gaussian::param_tuple(0., 0.005000));
    parallel_net->init<Constant>({11}, Constant::param_tuple(0.100000));
    parallel_net->init<Gaussian>({12}, Gaussian::param_tuple(0., 0.005000));
    parallel_net->init<Constant>({13}, Constant::param_tuple(0.100000));
    parallel_net->init<Gaussian>({14}, Gaussian::param_tuple(0., 0.010000));
    parallel_net->init<Constant>({15}, Constant::param_tuple(0.000000));

  } else {
    parallel_net->load(snapshot);
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  // initilize MPI
  int ret;
  MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &ret));

  MPI_LOG(<< "hello");
  // parallels
  vector<pair<int, int> > parallels;
  parallels.push_back({0, 3});

  // parameter server
  pair<int, int> param_server = {0, -1};
  // fetch image
  shared_ptr<FetchImage> fetch = make_shared<FetchImage>(source, mean_file,
      true, true, true, batch_size, image_size, parallels);
  fetch->run();

  // create data parallelism of NewModel
  shared_ptr<DataParallel<VirtualNetName<false>, AllReduce> > parallel_net
      = make_shared<DataParallel<VirtualNetName<false>, AllReduce> >(parallels);
  setup_param_server(parallel_net.get(), 0.1);
  initialize(parallel_net.get(), "");

  auto start = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
          (std::chrono::system_clock::now() - start);

  // iteration
  for (int iter = 1; iter <= 100000; ++iter) {
    // set learning rate
    if (current_rank() == 0) {
      for (int i = 0; i < weight_num; ++i) {
      //for (int i = 0; i < 2; ++i) {
        DTYPE global_learning_rate =
            0.001 * pow(1. - DTYPE(iter) / 100000., 0.5);
        DTYPE global_decay = 0.0001;
        DTYPE learning_rate = global_learning_rate * (i % 2 ? 2. : 1.);
        DTYPE weight_decay  = learning_rate * global_decay * (i % 2 ? 0. : 1.);
        parallel_net->param_server(i)->set_param(
            make_tuple<vector<DTYPE> >({0.9, learning_rate, weight_decay}));
      }
    }
    // feed prefetched data to net
    parallel_net->feed(fetch->images(), fetch->labels());
    // start net and next fetch
    parallel_net->run_async();
    fetch->run_async();
    fetch->sync();
    parallel_net->sync();

    // verbose
    if (iter % 10 == 0 && current_rank() == 0) {
        duration = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::system_clock::now() - start);
        cout << "iteration: " << iter << ", loss: "
        << parallel_net->loss()[0] << ", time: " << duration.count() << endl;
    }
    if (iter % 10000 == 0) {
      parallel_net->save("./net_no_aux_dump_iter_"
          + to_string(iter) + ".snapshot");
    }
  }
  // delete
  fetch.reset();
  parallel_net.reset();
  // Finalize MPI
  MPI_CHECK(MPI_Finalize());
  return 0;
}
