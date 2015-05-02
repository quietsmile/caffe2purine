// Copyright Lin Min 2015

#include <mpi.h>
#include <glog/logging.h>
#include "examples/googlenet.hpp"
#include "composite/graph/all_reduce.hpp"



#include<iostream>
using namespace std;
int batch_size = 128;
string source = "../data/bowl_train_lmdb";
string mean_file = "../data/bowl_mean.binaryproto";

using namespace purine;

void setup_param_server(DataParallel<GoogLeNet<false>, AllReduce>*
    parallel_googlenet, DTYPE global_learning_rate) {

  MPI_LOG(<< "hello_setup");
  // set learning rate etc
  DTYPE global_decay = 0.0001;
  vector<AllReduce::param_tuple> param(116);
  for (int i = 0; i < 116; ++i) {
    DTYPE learning_rate = global_learning_rate * (i % 2 ? 2. : 1.);
    param[i] = AllReduce::param_tuple(0.9, learning_rate,
        learning_rate * global_decay * (i % 2 ? 0. : 1.));
  }
  parallel_googlenet->setup_param_server(vector<int>(116, 0),
      vector<int>(116, -1), param);
}

void initialize(DataParallel<GoogLeNet<false>, AllReduce>* parallel_googlenet,
                const string& snapshot) {

  MPI_LOG(<< "hello_init");
  if (snapshot == "") {
    vector<int> indice(58);
    iota(indice.begin(), indice.end(), 0);
    vector<int> weight_indice(58);
    vector<int> bias_indice(58);
    transform(indice.begin(), indice.end(), weight_indice.begin(),
        [](int i)->int {
          return i * 2;
        });
    transform(indice.begin(), indice.end(), bias_indice.begin(),
        [](int i)->int {
          return i * 2 + 1;
        });
    parallel_googlenet->init<Constant>(bias_indice, Constant::param_tuple(0.2));
    parallel_googlenet->init<Gaussian>(weight_indice,
        Gaussian::param_tuple(0., 0.05));
    parallel_googlenet->init<Gaussian>({0, 2,
            12, 14,
            24, 26,
            36, 38,
            48, 50,
            60, 62,
            72, 74,
            84, 86,
            96, 98,
            108, 110,
            114},
        Gaussian::param_tuple(0., 0.01));
  } else {
    parallel_googlenet->load(snapshot);
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
  //for (int rank : {0, 1, 2, 3}) {
  //  for (int device : {0, 1, 2}) {
  for (int rank : {0}) {
    for (int device : {3}) {
      parallels.push_back({rank, device});
    }
  }
  /*
  for (int rank : {1}) {
    for (int device : {2}) {
      parallels.push_back({rank, device});
    }
  }*/
  // parameter server
  pair<int, int> param_server = {0, -1};
  // fetch image
  shared_ptr<FetchImage> fetch = make_shared<FetchImage>(source, mean_file,
      true, true, true, batch_size, 224, parallels);
  fetch->run();

  // create data parallelism of GoogLeNet;
  shared_ptr<DataParallel<GoogLeNet<false>, AllReduce> > parallel_googlenet
      = make_shared<DataParallel<GoogLeNet<false>, AllReduce> >(parallels);
  setup_param_server(parallel_googlenet.get(), 0.1);
  initialize(parallel_googlenet.get(), "");

  auto start = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
          (std::chrono::system_clock::now() - start);

  // iteration
  for (int iter = 1; iter <= 100000; ++iter) {
    // set learning rate
    if (current_rank() == 0) {
      for (int i = 0; i < 116; ++i) {
      //for (int i = 0; i < 2; ++i) {
        DTYPE global_learning_rate =
            0.001 * pow(1. - DTYPE(iter) / 100000., 0.5);
        DTYPE global_decay = 0.0001;
        DTYPE learning_rate = global_learning_rate * (i % 2 ? 2. : 1.);
        DTYPE weight_decay  = learning_rate * global_decay * (i % 2 ? 0. : 1.);
        parallel_googlenet->param_server(i)->set_param(
            make_tuple<vector<DTYPE> >({0.9, learning_rate, weight_decay}));
      }
    }

    if(0)
    cout << "hello_in_main" << endl;

    // feed prefetched data to googlenet
    parallel_googlenet->feed(fetch->images(), fetch->labels());
    // start googlenet and next fetch
    parallel_googlenet->run_async();
    fetch->run_async();
    fetch->sync();
    parallel_googlenet->sync();

    // verbose
    if(1)
    cout << "iteration: " << iter << ", loss: "
        << parallel_googlenet->loss()[0] << " "
        << parallel_googlenet->loss()[1] << endl;
    MPI_LOG( << "current_rank() " << current_rank() << "iteration: " << iter << ", loss: "
        << parallel_googlenet->loss()[0] << " "
        << parallel_googlenet->loss()[1]);
    if (iter % 100 == 0 && current_rank() == 0) {
        duration = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::system_clock::now() - start);
        cout << duration.count() << endl;
    }
    if (iter % 10000 == 0) {
      parallel_googlenet->save("./googlenet_no_aux_dump_iter_"
          + to_string(iter) + ".snapshot");
    }
  }

  // delete
  fetch.reset();
  parallel_googlenet.reset();
  // Finalize MPI
  MPI_CHECK(MPI_Finalize());
  return 0;
}
