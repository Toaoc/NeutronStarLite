#include "core/neutronstar.hpp"

uint64_t nts::op::ForwardCPUfuseOp::forward_compute_count = 0;
uint64_t nts::op::ForwardCPUfuseOp::backward_compute_count = 0;
class GCN_CPU_impl {
public:
  int iterations;
  ValueType learn_rate;
  ValueType weight_decay;
  ValueType drop_rate;
  ValueType alpha;
  ValueType beta1;
  ValueType beta2;
  ValueType epsilon;
  ValueType decay_rate;
  ValueType decay_epoch;
  // graph
  VertexSubset *active;
  // graph with no edge data
  Graph<Empty> *graph;
  //std::vector<CSC_segment_pinned *> subgraphs;
  // NN
  GNNDatum *gnndatum;
  NtsVar L_GT_C;    // 本地的label变量
  NtsVar L_GT_G;
  NtsVar MASK;      // 节点的mask的值
  //GraphOperation *gt;
  PartitionedGraph *partitioned_graph;
  // Variables
  std::vector<Parameter *> P;   // 存储了各层的参数
  std::vector<NtsVar> X;
  nts::ctx::NtsContext* ctx;
  
  NtsVar F;
  NtsVar loss;
  NtsVar tt;
  torch::nn::Dropout drpmodel;
  std::vector<torch::nn::BatchNorm1d> bn1d;
  
  double exec_time = 0;
  double all_sync_time = 0;
  double sync_time = 0;
  double all_graph_sync_time = 0;
  double graph_sync_time = 0;
  double all_compute_time = 0;
  double compute_time = 0;
  double all_copy_time = 0;
  double copy_time = 0;
  double graph_time = 0;
  double all_graph_time = 0;

    double forward_time = 0;
    double backward_time = 0;
    double test_time = 0;
    double loss_time = 0;
    double update_time = 0;
    double forward_graph_time = 0;

  GCN_CPU_impl(Graph<Empty> *graph_, int iterations_,
               bool process_local = false, bool process_overlap = false) {
      // 赋值用户指定的变量
    graph = graph_;
    iterations = iterations_;

    // 分配活跃顶点的位图
    active = graph->alloc_vertex_subset();
    active->fill();

    // 初始化图神经网络运行上下文
    graph->init_gnnctx(graph->config->layer_string);
    // rtminfo initialize
    // 初始化运行时信息
    graph->init_rtminfo();
    graph->rtminfo->process_local = graph->config->process_local;
    graph->rtminfo->reduce_comm = graph->config->process_local;
    graph->rtminfo->copy_data = false;
    graph->rtminfo->process_overlap = graph->config->overlap;
    graph->rtminfo->with_weight = true;
    graph->rtminfo->with_cuda = false;
    graph->rtminfo->lock_free = graph->config->lock_free;
  }

  /**
   * 初始化图基本信息
   */
  void init_graph() {
      // 将graph改造成分块图
    partitioned_graph=new PartitionedGraph(graph, active);
    // 为分块图生成度常量，即根号的系数
    partitioned_graph->GenerateAll([&](VertexId src, VertexId dst) {
      return nts::op::nts_norm_degree(graph,src, dst);
    },CPU_T,(graph->partitions)>1);
    // 初始化图通讯器
    graph->init_communicatior();
    //cp = new nts::autodiff::ComputionPath(gt, subgraphs);
    // 创建图神经网络运行上下文
    ctx=new nts::ctx::NtsContext();
  }

  /**
   * 初始化图神经网络相关参数
   */
  void init_nn() {

    learn_rate = graph->config->learn_rate;
    weight_decay = graph->config->weight_decay;
    drop_rate = graph->config->drop_rate;
    alpha = graph->config->learn_rate;
    decay_rate = graph->config->decay_rate;
    decay_epoch = graph->config->decay_epoch;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-9;
    GNNDatum *gnndatum = new GNNDatum(graph->gnnctx, graph);
    // gnndatum->random_generate();

    // 读取feature或者随机生成feature
    if (0 == graph->config->feature_file.compare("random")) {
      gnndatum->random_generate();
    } else {
        // 从文件中读取feature、label和节点掩码（mask）
      gnndatum->readFeature_Label_Mask(graph->config->feature_file,
                                       graph->config->label_file,
                                       graph->config->mask_file);
      // gnndatum->readFeature_Label_Mask_OGB(graph->config->feature_file,
      //                                  graph->config->label_file,
      //                                  graph->config->mask_file);
    }

    // creating tensor to save Label and Mask
    // 登记label和mask变量
    gnndatum->registLabel(L_GT_C);
    gnndatum->registMask(MASK);

    // initializeing parameter. Creating tensor with shape [layer_size[i],
    // layer_size[i + 1]]
    // 为各层分配空间
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i],
                                graph->gnnctx->layer_size[i + 1], alpha, beta1,
                                beta2, epsilon, weight_decay));
      if(i < graph->gnnctx->layer_size.size() - 2)
        bn1d.push_back(torch::nn::BatchNorm1d(graph->gnnctx->layer_size[i])); 
    }

    // synchronize parameter with other processes
    // because we need to guarantee all of workers are using the same model
    for (int i = 0; i < P.size(); i++) {
      P[i]->init_parameter();
      P[i]->set_decay(decay_rate, decay_epoch);
    }
    drpmodel = torch::nn::Dropout(
        torch::nn::DropoutOptions().p(drop_rate).inplace(true));

    F = graph->Nts->NewLeafTensor(
        gnndatum->local_feature,
        {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
        torch::DeviceType::CPU);

    // X[i] is vertex representation at layer i
    for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
      NtsVar d;
      X.push_back(d);
    }
    // X[0] is the initial vertex representation. We created it from
    // local_feature
    // X[0]就是第一层的feature
    X[0] = F.set_requires_grad(true);
  }

  /**
   * 根据掩码计算训练集、验证集和测试集的准确度
   * @param s 掩码，0 train，1 eval，2 test
   */
  void Test(long s) { // 0 train, //1 eval //2 test
    NtsVar mask_train = MASK.eq(s);
    NtsVar all_train =
        X[graph->gnnctx->layer_size.size() - 1]
            .argmax(1)
            .to(torch::kLong)
            .eq(L_GT_C)
            .to(torch::kLong)
            .masked_select(mask_train.view({mask_train.size(0)}));
    NtsVar all = all_train.sum(0);
    long *p_correct = all.data_ptr<long>();
    long g_correct = 0;
    long p_train = all_train.size(0);
    long g_train = 0;
    MPI_Datatype dt = get_mpi_data_type<long>();
    MPI_Allreduce(p_correct, &g_correct, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&p_train, &g_train, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    float acc_train = 0.0;
    if (g_train > 0)
      acc_train = float(g_correct) / g_train;
    if (graph->partition_id == 0) {
      if (s == 0) {
        LOG_INFO("Train Acc: %f %d %d", acc_train, g_train, g_correct);
      } else if (s == 1) {
        LOG_INFO("Eval Acc: %f %d %d", acc_train, g_train, g_correct);
      } else if (s == 2) {
        LOG_INFO("Test Acc: %f %d %d", acc_train, g_train, g_correct);
      }
    }
  }

  /**
   * 来进行nn计算
   * 该方法已经弃用
   * @param a
   * @param x
   * @return
   */
  NtsVar vertexForward(NtsVar &a, NtsVar &x) {
    NtsVar y;
    int layer = graph->rtminfo->curr_layer;
    // nn operation. Here is just a simple matmul. i.e. y = activate(a * w)
    if (layer == 0) {
      a =this->bn1d[layer](a);
      y = torch::relu(P[layer]->forward(a)).set_requires_grad(true);
    } else if (layer == 1) {
      y = P[layer]->forward(a);
      y = y.log_softmax(1);
    }
    // save the intermediate result for backward propagation
 //   ctx->op_push(a, y, nts::ctx::NNOP);
    return y;
  }

  /**
   * 计算训练的loss
   */
  void Loss() {
    //  return torch::nll_loss(a,L_GT_C);
    // 首先要计算最后一层log softmax
    torch::Tensor a = X[graph->gnnctx->layer_size.size() - 1].log_softmax(1);
    // 每个顶点都有mask，只有部分顶点是训练顶点
    torch::Tensor mask_train = MASK.eq(0);
    // 使用train顶点来计算nll_loss
    loss = torch::nll_loss(
        a.masked_select(mask_train.expand({mask_train.size(0), a.size(1)}))
            .view({-1, a.size(1)}),
        L_GT_C.masked_select(mask_train.view({mask_train.size(0)})));
    // 然后追加算子到上下文中
    ctx->appendNNOp(X[graph->gnnctx->layer_size.size() - 1], loss);
  }

  /**
   * 更新参数的函数
   */
  void Update() {
      // 对于每一层都需要更新参数
    for (int i = 0; i < P.size(); i++) {
      // accumulate the gradient using all_reduce
      // 首先累加使用all_reduce累加各个机器中对应层的梯度
      P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      // update parameters with Adam optimizer
      // 然后使用Adam优化器来更新参数
      P[i]->learnC2C_with_decay_Adam();
      P[i]->next();
    }
  }
  /**
   * 执行前向传播的函数
   */
  void Forward() {
      // 标记当前是在前向阶段
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
        // 记录当前的层数
      graph->rtminfo->curr_layer = i;
//      if (i != 0) {
//        X[i] = drpmodel(X[i]);
//      }

        // 首先运行图传播算子
        forward_graph_time -= get_time();
       NtsVar Y_i= ctx->runGraphOp<nts::op::ForwardCPUfuseOp>(partitioned_graph,active,X[i]);
       forward_graph_time += get_time();
        // 然后运行神经网络算子
        X[i + 1]=ctx->runVertexForward([&](NtsVar n_i,NtsVar v_i){
            if(i<(graph->gnnctx->layer_size.size() - 2)){
                n_i =this->bn1d[i](n_i);
                return drpmodel(torch::relu(P[i]->forward(n_i)));
                
            }else{
                return  P[i]->forward(n_i);
            }
            
            //return vertexForward(n_i, v_i);
        },
        Y_i,
        X[i]);
    }
  }

    /**
     * 整个图神经网络的运行函数
     */
    void run() {
        if (graph->partition_id == 0) {
            LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n",
                     iterations);
        }

        exec_time -= get_time();
        // 迭代规定的次数
        for (int i_i = 0; i_i < iterations; i_i++) {
            // 将当前epoch存起来，方便之后读取
            graph->rtminfo->epoch = i_i;
            if (i_i != 0) {
                for (int i = 0; i < P.size(); i++) {
                    P[i]->zero_grad();
                }
            }

            // 执行前向传播
            forward_time -= get_time();
            Forward();
            forward_time += get_time();
            // 计算训练集准确度
            test_time -= get_time();
            Test(0);
            // 计算验证集准确度
            Test(1);
            // 计算测试集准确度
            Test(2);
            test_time += get_time();
            // 计算loss
            loss_time -= get_time();
            Loss();
            loss_time += get_time();

            // 执行loss反向传播
            backward_time -= get_time();
            ctx->self_backward();
            backward_time += get_time();
            // 利用梯度更新参数
            update_time -= get_time();
            Update();
            update_time += get_time();
//       ctx->debug();
            if (graph->partition_id == 0)
                std::cout << "Nts::Running.Epoch[" << i_i << "]:loss\t" << loss
                          << std::endl;
        }
        exec_time += get_time();
        LOG_INFO("前向传播计算总次数：%lu", nts::op::ForwardCPUfuseOp::forward_compute_count);
        LOG_INFO("反向传播计算总次数：%lu", nts::op::ForwardCPUfuseOp::backward_compute_count);
        LOG_INFO("前向: %lf, test: %lf, loss, %lf, 反向：%lf, update: %lf", forward_time, test_time, loss_time, backward_time, update_time);
        LOG_INFO("前向图计算时间：%lf, 反向图计算时间：%lf", graph->forward_process_time, graph->backward_process_time);
        LOG_INFO("上层前向图计算时间：%lf", forward_graph_time);

//    std::string str="a10";
//    at::ArrayRef<at::Dimname>names({at::Dimname::fromSymbol(at::Symbol::dimname(str)),at::Dimname::fromSymbol(at::Symbol::dimname("b10"))});
//    at::ArrayRef<at::Dimname>names({at::Dimname::fromSymbol(at::Symbol::dimname(str)),at::Dimname::fromSymbol(at::Symbol::dimname("b10"))});
//    NtsVar s=torch::ones({3,3},names,at::TensorOptions().requires_grad(true));
//    std::vector<at::Dimname> dim;

//    dim.push_back(at::Dimname::fromSymbol(at::Symbol::aten("0")));
//    dim.push_back(at::Dimname::fromSymbol(at::Symbol::aten("1")));
//    s.get_named_tensor_meta()->set_names(at::NamedTensorMeta::HasNonWildcard,dim);
//    at::DimnameList::
//    std::cout<<" "<<s.names()[0]<<" " <<s.names()[1]<<std::endl;
        delete active;
    }
};
