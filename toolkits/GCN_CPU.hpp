#include "core/gnnmini.hpp"
#include <c10/cuda/CUDAStream.h>


class GCN_CPU_impl{
public:
        
    int iterations;
    ValueType learn_rate;
    ValueType weight_decay;
    
    //graph
    VertexSubset *active;
    Graph<Empty>* graph;
    std::vector<CSC_segment_pinned *>subgraphs;
    //NN
    GNNDatum *gnndatum;
    NtsVar L_GT_C;
    NtsVar L_GT_G;
    NtsVar MASK;
    std::map<std::string,NtsVar>I_data;
    GTensor<ValueType, long> *gt;
    //Variables
    std::vector<GnnUnit*>P;
    std::vector<NtsVar>X;
    std::vector<NtsVar>Y;
    std::vector<NtsVar>X_grad;
    NtsVar F;
    NtsVar loss;
    NtsVar tt;
    
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

    
    GCN_CPU_impl(Graph<Empty> *graph_, int iterations_, 
                    bool process_local = false, bool process_overlap = false){
        graph=graph_;
        iterations=iterations_;
        learn_rate = 0.01;
        weight_decay=0.05;
        active = graph->alloc_vertex_subset();
        active->fill();
        
        graph->init_gnnctx(graph->config->layer_string);
            //rtminfo initialize
        graph->init_rtminfo();
        graph->rtminfo->process_local= graph->config->process_local;
        graph->rtminfo->reduce_comm = graph->config->process_local;
        graph->rtminfo->copy_data = false;
        graph->rtminfo->process_overlap = graph->config->overlap;
        graph->rtminfo->with_weight=true;
        graph->rtminfo->with_cuda=true;
       
    } 
    void init_graph(){
        //std::vector<CSC_segment_pinned *> csc_segment;
        graph->generate_COO(active);
        graph->reorder_COO_W2W();
        //generate_CSC_Segment_Tensor_pinned(graph, csc_segment, true);
        gt = new GTensor<ValueType, long>(graph, active);
        gt->GenerateGraphSegment(subgraphs,CPU_T,[&](VertexId src,VertexId dst){
            return gt->norm_degree(src,dst);
            });
        graph->init_communicatior();
    }
    void init_nn(){
        GNNDatum *gnndatum = new GNNDatum(graph->gnnctx);
        //gnndatum->random_generate();
        if(0==graph->config->feature_file.compare("random")){
            gnndatum->random_generate();
        }else{
            gnndatum->readFtrFrom1(graph->config->feature_file,graph->config->label_file);
        }
        gnndatum->registLabel(L_GT_C);
        gnndatum->registMask(MASK);
        
        for(int i=0;i<graph->gnnctx->layer_size.size()-1;i++){
            P.push_back(new GnnUnit(graph->gnnctx->layer_size[i], 
                        graph->gnnctx->layer_size[i+1]));
        }
        for(int i=0;i<P.size();i++){
            P[i]->init_parameter();
        }
        

            F=graph->Nts->NewLeafTensor(gnndatum->local_feature,
                {graph->gnnctx->l_v_num,graph->gnnctx->layer_size[0]},
                                     torch::DeviceType::CPU);

                
        for(int i=0;i<graph->gnnctx->layer_size.size()-1;i++){
            Y.push_back(graph->Nts->NewKeyTensor(
                                {graph->gnnctx->l_v_num, 
                                   graph->gnnctx->layer_size[i]},
                                       torch::DeviceType::CPU));
            X_grad.push_back(graph->Nts->NewKeyTensor(
                                {graph->gnnctx->l_v_num, 
                                   graph->gnnctx->layer_size[i]},
                                       torch::DeviceType::CPU));
        }
        for(int i=0;i<graph->gnnctx->layer_size.size();i++){
        NtsVar d;X.push_back(d);
        }
        X[0]=F.set_requires_grad(true);
    }


void Test(NtsVar& y,long s){// 0 train, //1 eval //2 test 
    NtsVar mask_train=MASK.eq(s);
    NtsVar all_train=y.argmax(1).to(torch::kLong)
                    .eq(L_GT_C).to(torch::kLong)
                    .masked_select(mask_train.view({mask_train.size(0)}));
                    std::cout<<"all train1"<<all_train.size(0)<<std::endl;
//                    NtsVar all1=all_train1.sum(0);
                    
    //NtsVar all_train=y.argmax(1).to(torch::kLong).eq(L_GT_C).to(torch::kLong);        
    NtsVar all=all_train.sum(0);
    long * p_correct=all.data_ptr<long>();
    long g_correct=0;
    long p_train=all_train.size(0);
    long g_train=0;
    MPI_Datatype dt = get_mpi_data_type<long>();
    MPI_Allreduce(p_correct, &g_correct, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&p_train, &g_train, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    float acc_train=0.0;
    if(g_train>0)
        acc_train=float(g_correct)/g_train;
   if(graph->partition_id==0){
      if(s==0)
        std::cout<<"Train ACC: "<<acc_train<<" "<<g_train<<" "<<g_correct<<std::endl;
      else if(s==1)
        std::cout<<"Eval  ACC: "<<acc_train<<" "<<g_train<<" "<<g_correct<<" "<<std::endl;
      else if(s==2)
        std::cout<<"Test  ACC: "<<acc_train<<std::endl;
   }
    
}
NtsVar vertexForward(NtsVar &a, NtsVar &x){
    NtsVar y;
    int layer=graph->rtminfo->curr_layer;
    if(layer==0){
        y=torch::relu(P[layer]->forward(a)).set_requires_grad(true);
    }
    else if(layer==1){
        y = P[layer]->forward(a);
        y = y.log_softmax(1); 
    }
    return y;
}
NtsVar Loss(NtsVar &a){
  //  return torch::nll_loss(a,L_GT_C);
    torch::Tensor mask_train=MASK.eq(1);
    return torch::nll_loss(
                a.masked_select(mask_train
                        .expand({mask_train.size(0),a.size(1)}))
                            .view({-1,a.size(1)}),
                L_GT_C.masked_select(mask_train.view({mask_train.size(0)})));
}
void vertexBackward(){
    
    int layer=graph->rtminfo->curr_layer;
    if(layer==0){
        X[1].backward(X_grad[1]); //new
    }
    else if(layer==1){
        loss.backward();
    }
}

void Backward(){
    graph->rtminfo->forward = false;
    for(int i=graph->gnnctx->layer_size.size()-2;i>=0;i--){
    graph->rtminfo->curr_layer = i;
    vertexBackward();
    NtsVar grad_to_Y=Y[i].grad();
    //gt->PropagateBackwardCPU(grad_to_Y, X_grad[i]);
    gt->PropagateBackwardCPU_debug(grad_to_Y, X_grad[i],subgraphs);
    }
    
    //  std::cout<<P[1]->W.grad()<<std::endl;
    for(int i=0;i<P.size()-1;i++){
        P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
        P[i]->learnC2C_with_decay(learn_rate,weight_decay);
    }
    
        
}

void Forward(){
    graph->rtminfo->forward = true;    
    for(int i=0;i<graph->gnnctx->layer_size.size()-1;i++){
    graph->rtminfo->curr_layer = i;
    gt->PropagateForwardCPU_debug(X[i], Y[i],subgraphs);    
    X[i+1]=vertexForward(Y[i],X[i]);
    }
    Test(X[graph->gnnctx->layer_size.size()-1],1);
    loss=Loss(X[graph->gnnctx->layer_size.size()-1]);
}




/*GPU dist*/ void run()
{
    if (graph->partition_id == 0)
        printf("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epochs\n",iterations);
       // graph->print_info();
        
    exec_time -= get_time();
    for (int i_i = 0; i_i < iterations; i_i++){
        graph->rtminfo->epoch = i_i;
        if (i_i != 0){
            for(int i=0;i<P.size();i++){
            P[i]->zero_grad();
            //Y[i].grad().zero_();
            }
        }      
       Forward();
       Backward();         

//    graph->rtminfo->forward = true;
//    graph->rtminfo->curr_layer=0;    
//   gt->PropagateForwardCPU_debug(X[0], Y[0], subgraphs);
//    for(VertexId i=0;i<graph->partitions;i++)
//    if(graph->partition_id==i){
//        int test=graph->gnnctx->p_v_s+1;
//        std::cout<<"DEBUG"<<graph->in_degree_for_backward[test]<<" X: "<<X[0][test-graph->gnnctx->p_v_s][15]<<" Y: "<<Y[0][test-graph->gnnctx->p_v_s][15]<<std::endl;
//        test=graph->gnnctx->p_v_e-1;
//        std::cout<<"DEBUG"<<graph->in_degree_for_backward[test]<<" X: "<<X[0][test-graph->gnnctx->p_v_s][15]<<" Y: "<<Y[0][test-graph->gnnctx->p_v_s][15]<<std::endl;
//        test=(graph->gnnctx->p_v_e+graph->gnnctx->p_v_s)/2+1;
//        std::cout<<"DEBUG"<<graph->in_degree_for_backward[test]<<" X: "<<X[0][test-graph->gnnctx->p_v_s][15]<<" Y: "<<Y[0][test-graph->gnnctx->p_v_s][15]<<std::endl;
//    } 
        
//    graph->rtminfo->forward = false;
//    graph->rtminfo->curr_layer=0;    
//    gt->GraphPropagateBackward(X[0], Y[0], subgraphs);
//    for(VertexId i=0;i<graph->partitions;i++)
//    if(graph->partition_id==i){
//        int test=graph->gnnctx->p_v_s;
//        std::cout<<"DEBUG"<<graph->out_degree_for_backward[test]<<" X: "<<X[0][test-graph->gnnctx->p_v_s][15]<<" Y: "<<Y[0][test-graph->gnnctx->p_v_s][15]<<std::endl;
//        test=graph->gnnctx->p_v_e-1;
//        std::cout<<"DEBUG"<<graph->out_degree_for_backward[test]<<" X: "<<X[0][test-graph->gnnctx->p_v_s][15]<<" Y: "<<Y[0][test-graph->gnnctx->p_v_s][15]<<std::endl;
//        test=(graph->gnnctx->p_v_e+graph->gnnctx->p_v_s)/2;
//        std::cout<<"DEBUG"<<graph->out_degree_for_backward[test]<<" X: "<<X[0][test-graph->gnnctx->p_v_s][15]<<" Y: "<<Y[0][test-graph->gnnctx->p_v_s][15]<<std::endl;
//    }                
     
        if (graph->partition_id == 0)
           std::cout<<"Nts::Running.Epoch["<<i_i<<"]:loss\t"<<loss<< std::endl;       
    }
    exec_time += get_time();
    
//    if(graph->partition_id==0)
//        graph->config->readFromCfgFile("NTS_cora_data.cfg");
    delete active;
}



void DEBUGINFO(){

    if (graph->partition_id == 0)
    {
        printf("\n#Timer Info Start:\n");
        printf("#all_time=%lf(s)\n", exec_time);
        printf("#sync_time=%lf(s)\n", all_sync_time);
        printf("#all_graph_sync_time=%lf(s)\n", all_graph_sync_time);
        printf("#copy_time=%lf(s)\n", all_copy_time);
        printf("#nn_time=%lf(s)\n", all_compute_time);
        printf("#graph_time=%lf(s)\n", all_graph_time);
        printf("#communicate_extract+send=%lf(s)\n", graph->all_compute_time);
        printf("#communicate_processing_received=%lf(s)\n", graph->all_overlap_time);
        printf("#communicate_processing_received.copy=%lf(s)\n", graph->all_recv_copy_time);
        printf("#communicate_processing_received.kernel=%lf(s)\n", graph->all_recv_kernel_time);
        printf("#communicate_processing_received.wait=%lf(s)\n", graph->all_recv_wait_time);
        printf("#communicate_wait=%lf(s)\n", graph->all_wait_time);
        printf("#streamed kernel_time=%lf(s)\n", graph->all_kernel_time);
        printf("#streamed movein_time=%lf(s)\n", graph->all_movein_time);
        printf("#streamed moveout_time=%lf(s)\n", graph->all_moveout_time);
        printf("#cuda wait time=%lf(s)\n", graph->all_cuda_sync_time);
        printf("#graph repliation time=%lf(s)\n", graph->all_replication_time);
        printf("#Timer Info End\n");
    }
    //      NtsVar tt_cpu=tt.cpu();
    //  if(i_i==(iterations-1)&&graph->partition_id==0){
    //     inference(tt_cpu,graph, embedding, pytool,W1,W2);
    //  }
    double max_time = 0;
    double mean_time = 0;
    double another_time = 0;
    MPI_Datatype l_vid_t = get_mpi_data_type<double>();
    MPI_Allreduce(&all_graph_time, &max_time, 1, l_vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&exec_time, &another_time, 1, l_vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&graph->all_replication_time, &mean_time, 1, l_vid_t, MPI_SUM, MPI_COMM_WORLD);
    if (graph->partition_id == 0)
        printf("ALL TIME = %lf(s) GRAPH TIME = %lf(s) MEAN TIME = %lf(s)\n", 
                another_time, max_time / graph->partitions, mean_time / graph->partitions);
}

};