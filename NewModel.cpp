// Copyright Lin Min 2015

#include <mpi.h>
#include <glog/logging.h>
#include "examples/VirtualModel.hpp"
#include "composite/graph/all_reduce.hpp"

#include<iostream>
using namespace std;
int batch_size = 64
string source = "../data/bowl_train_lmdb"
string mean_file = "../data/bowl_mean.binaryproto"
const int weight_num = 128
const int image_size = 224

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
    parallel_net->init<Uniform>({0}, Uniform::param_tuple(-0.004464, 0.004464));
    parallel_net->init<Constant>({1}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({2}, Uniform::param_tuple(-0.003866, 0.003866));
    parallel_net->init<Constant>({3}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({4}, Uniform::param_tuple(-0.003866, 0.003866));
    parallel_net->init<Constant>({5}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({6}, Uniform::param_tuple(-0.004464, 0.004464));
    parallel_net->init<Constant>({7}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({8}, Uniform::param_tuple(-0.004464, 0.004464));
    parallel_net->init<Constant>({9}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({10}, Uniform::param_tuple(-0.006313, 0.006313));
    parallel_net->init<Constant>({11}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({12}, Uniform::param_tuple(-0.004464, 0.004464));
    parallel_net->init<Constant>({13}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({14}, Uniform::param_tuple(-0.015465, 0.015465));
    parallel_net->init<Constant>({15}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({16}, Uniform::param_tuple(-0.004464, 0.004464));
    parallel_net->init<Constant>({17}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({18}, Uniform::param_tuple(-0.003866, 0.003866));
    parallel_net->init<Constant>({19}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({20}, Uniform::param_tuple(-0.003866, 0.003866));
    parallel_net->init<Constant>({21}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({22}, Uniform::param_tuple(-0.005468, 0.005468));
    parallel_net->init<Constant>({23}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({24}, Uniform::param_tuple(-0.003866, 0.003866));
    parallel_net->init<Constant>({25}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({26}, Uniform::param_tuple(-0.010935, 0.010935));
    parallel_net->init<Constant>({27}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({28}, Uniform::param_tuple(-0.003866, 0.003866));
    parallel_net->init<Constant>({29}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({30}, Uniform::param_tuple(-0.005647, 0.005647));
    parallel_net->init<Constant>({31}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({32}, Uniform::param_tuple(-0.005647, 0.005647));
    parallel_net->init<Constant>({33}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({34}, Uniform::param_tuple(-0.012627, 0.012627));
    parallel_net->init<Constant>({35}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({36}, Uniform::param_tuple(-0.005647, 0.005647));
    parallel_net->init<Constant>({37}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({38}, Uniform::param_tuple(-0.030929, 0.030929));
    parallel_net->init<Constant>({39}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({40}, Uniform::param_tuple(-0.005647, 0.005647));
    parallel_net->init<Constant>({41}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({42}, Uniform::param_tuple(-0.019137, 0.019137));
    parallel_net->init<Constant>({43}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({44}, Uniform::param_tuple(-0.038273, 0.038273));
    parallel_net->init<Constant>({45}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({46}, Uniform::param_tuple(-0.054127, 0.054127));
    parallel_net->init<Constant>({47}, Constant::param_tuple(0.000000));
    parallel_net->init<Uniform>({48}, Uniform::param_tuple(-0.005468, 0.005468));
    parallel_net->init<Constant>({49}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({50}, Uniform::param_tuple(-0.005468, 0.005468));
    parallel_net->init<Constant>({51}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({52}, Uniform::param_tuple(-0.011690, 0.011690));
    parallel_net->init<Constant>({53}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({54}, Uniform::param_tuple(-0.005468, 0.005468));
    parallel_net->init<Constant>({55}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({56}, Uniform::param_tuple(-0.025254, 0.025254));
    parallel_net->init<Constant>({57}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({58}, Uniform::param_tuple(-0.005468, 0.005468));
    parallel_net->init<Constant>({59}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({60}, Uniform::param_tuple(-0.005468, 0.005468));
    parallel_net->init<Constant>({61}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({62}, Uniform::param_tuple(-0.005468, 0.005468));
    parallel_net->init<Constant>({63}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({64}, Uniform::param_tuple(-0.010935, 0.010935));
    parallel_net->init<Constant>({65}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({66}, Uniform::param_tuple(-0.005468, 0.005468));
    parallel_net->init<Constant>({67}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({68}, Uniform::param_tuple(-0.025254, 0.025254));
    parallel_net->init<Constant>({69}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({70}, Uniform::param_tuple(-0.005468, 0.005468));
    parallel_net->init<Constant>({71}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({72}, Uniform::param_tuple(-0.005468, 0.005468));
    parallel_net->init<Constant>({73}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({74}, Uniform::param_tuple(-0.005468, 0.005468));
    parallel_net->init<Constant>({75}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({76}, Uniform::param_tuple(-0.010310, 0.010310));
    parallel_net->init<Constant>({77}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({78}, Uniform::param_tuple(-0.005468, 0.005468));
    parallel_net->init<Constant>({79}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({80}, Uniform::param_tuple(-0.021870, 0.021870));
    parallel_net->init<Constant>({81}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({82}, Uniform::param_tuple(-0.005468, 0.005468));
    parallel_net->init<Constant>({83}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({84}, Uniform::param_tuple(-0.018844, 0.018844));
    parallel_net->init<Constant>({85}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({86}, Uniform::param_tuple(-0.038273, 0.038273));
    parallel_net->init<Constant>({87}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({88}, Uniform::param_tuple(-0.054127, 0.054127));
    parallel_net->init<Constant>({89}, Constant::param_tuple(0.000000));
    parallel_net->init<Uniform>({90}, Uniform::param_tuple(-0.005384, 0.005384));
    parallel_net->init<Constant>({91}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({92}, Uniform::param_tuple(-0.005384, 0.005384));
    parallel_net->init<Constant>({93}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({94}, Uniform::param_tuple(-0.009781, 0.009781));
    parallel_net->init<Constant>({95}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({96}, Uniform::param_tuple(-0.005384, 0.005384));
    parallel_net->init<Constant>({97}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({98}, Uniform::param_tuple(-0.021870, 0.021870));
    parallel_net->init<Constant>({99}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({100}, Uniform::param_tuple(-0.005384, 0.005384));
    parallel_net->init<Constant>({101}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({102}, Uniform::param_tuple(-0.008578, 0.008578));
    parallel_net->init<Constant>({103}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({104}, Uniform::param_tuple(-0.008578, 0.008578));
    parallel_net->init<Constant>({105}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({106}, Uniform::param_tuple(-0.019562, 0.019562));
    parallel_net->init<Constant>({107}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({108}, Uniform::param_tuple(-0.008578, 0.008578));
    parallel_net->init<Constant>({109}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({110}, Uniform::param_tuple(-0.043741, 0.043741));
    parallel_net->init<Constant>({111}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({112}, Uniform::param_tuple(-0.008578, 0.008578));
    parallel_net->init<Constant>({113}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({114}, Uniform::param_tuple(-0.008578, 0.008578));
    parallel_net->init<Constant>({115}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({116}, Uniform::param_tuple(-0.008578, 0.008578));
    parallel_net->init<Constant>({117}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({118}, Uniform::param_tuple(-0.017857, 0.017857));
    parallel_net->init<Constant>({119}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({120}, Uniform::param_tuple(-0.008578, 0.008578));
    parallel_net->init<Constant>({121}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({122}, Uniform::param_tuple(-0.035714, 0.035714));
    parallel_net->init<Constant>({123}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({124}, Uniform::param_tuple(-0.008578, 0.008578));
    parallel_net->init<Constant>({125}, Constant::param_tuple(0.200000));
    parallel_net->init<Uniform>({126}, Uniform::param_tuple(-0.054127, 0.054127));
    parallel_net->init<Constant>({127}, Constant::param_tuple(0.000000));

  } else {
    parallel_net->load(snapshot);
  }
}

int main(int argc, char** argv) {
  ::InitLogging(argv[0]);
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

  // create data parallelism of NewNet
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
    if (iter % 100 == 0 && current_rank() == 0) {
        duration = std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::system_clock::now() - start);
        cout << "iteration: " << iter << ", loss: "
        << parallel_net->loss()[0] << ", time: " duration.count() << endl;
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
