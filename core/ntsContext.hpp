/*
Copyright (c) 2021-2022 Qiange Wang, Northeastern University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef NTSOPS_HPP
#define NTSOPS_HPP
#include <stack>
#include "ntsGraphOp.hpp"
#include<type_traits>
namespace nts {

namespace ctx {

typedef uint32_t OpType;

const OpType NNOP = 0;
const OpType GRAPHOP = 1;
const OpType SELFNNOP = 2;
const OpType BIGRAPHOP = 3;

class IOTensorId{
public:
    IOTensorId(long o_id_,long i_id1_,long i_id2_){
        o_id=o_id_;
        i_id1=i_id1_;
        i_id2=i_id2_;
    }
    IOTensorId(long o_id_,long i_id1_){
        o_id=o_id_;
        i_id1=i_id1_;
    }
    void updateOutput(long o_id_){
       o_id=o_id_; 
    }
    long o_id;
    long i_id1;
    long i_id2;
};
class ntsOperator{
public:
    ntsOperator(){
        
    }
    ntsOperator(nts::op::ntsGraphOp* op_,OpType op_t_){
        assert((GRAPHOP==op_t_)||(BIGRAPHOP==op_t_));
        op=op_;
        op_t=op_t_;
    }
    ntsOperator(nts::op::ntsNNBaseOp* op_,OpType op_t_){
        assert(SELFNNOP==op_t_);
        opn=op_;
        op_t=op_t_;
    }
    ntsOperator(OpType op_t_){
        assert(NNOP==op_t_);
        op_t=op_t_;
    }
//    ntsOperator(OpType op_t_){
//        assert(CATOP==op_t_);
//        op_t=op_t_;
//    }
    nts::op::ntsGraphOp* get_graphop(){
        return op;
    }
    nts::op::ntsNNBaseOp* get_nnop(){
        return opn;
    }
    OpType get_op_T(){
        return op_t;
    }
    nts::op::ntsGraphOp* op;
    nts::op::ntsNNBaseOp* opn;
    OpType op_t;
};
/**
 * @brief
 * since GNN operation is just iteration of graph operation and NN operation.
 * so we can simply use a chain to represent GNN operation, which can reduce
 * system complexity greatly.
 * you can also regard it as the NN operation splited by graph operation.
 * And as the extention of auto diff library, we will provide backward
 * computation for graph operation. And thus, the computation path of GNN is
 * constructed.
 */
class NtsContext {
public:
  NtsContext(){
  std::stack<OpType>().swap(op);
  std::stack<NtsVar>().swap(output);
  std::stack<NtsVar>().swap(input);
  std::stack<ntsOperator>().swap(ntsOp);
  output_grad.clear();
  iot_id.clear();
  count = 0;
  training = true; // default is training mode
}
/**
 * 运行图操作
 * @tparam GOPT 图算子的子类
 * @param partitioned_graph 分区的图
 * @param active 活跃的图顶点
 * @param f_input 图操作的输入，前向时为该层的embedding/feature
 * @return 图操作的输出
 */
  template <typename GOPT>
  NtsVar runGraphOp(PartitionedGraph* partitioned_graph, VertexSubset *active,
        NtsVar &f_input){//graph op
      // GOPT一定要是ntsGraphOp的子类
    static_assert(std::is_base_of<nts::op::ntsGraphOp,GOPT>::value,
                "template must be a type of graph op!");
    
    nts::op::ntsGraphOp * curr=new GOPT(partitioned_graph,active);
    // 调用图操作的前向操作，GCN_CPU中是ForwardCPUfuseOp.forward函数
    NtsVar f_output=curr->forward(f_input);
    // 如果是在训练，那么就记录相应的算子，便于反向时调用进行梯度求导
    if (this->training == true) {
      NtsVar ig;
      op.push(GRAPHOP);         // 记录操作的类型
      output.push(f_output);    // 记录输出的tensor
      input.push(f_input);      // 记录输入的tensor
      ntsOp.push(ntsOperator(curr,GRAPHOP));    // 记录操作的类，便于调用其反向传播函数
      iot_id.push_back(IOTensorId((long)(f_output.data_ptr()),(long)(f_input.data_ptr()))); // 同时记录输入和输出tensor的指针
      // pre-alloc space to save graident
      output_grad.push_back(ig);    // 为反向传播创建梯度空间
      count++;
    }
    return f_output;
}
  template <typename GOPT>
  NtsVar runGraphOp(PartitionedGraph* partitioned_graph, VertexSubset *active,
        NtsVar &f_input1,NtsVar &f_input2){//graph op
      
    static_assert(std::is_base_of<nts::op::ntsGraphOp,GOPT>::value,
                "template must be a type of graph op!");
    
    nts::op::ntsGraphOp * curr=new GOPT(partitioned_graph,active);
    NtsVar f_output=curr->forward(f_input1,f_input2);
    NtsVar ig;
    op.push(BIGRAPHOP);
    output.push(f_output);
    input.push(f_input1);
    ntsOp.push(ntsOperator(curr,BIGRAPHOP));
    iot_id.push_back(IOTensorId((long)(f_output.data_ptr()),(long)(f_input1.data_ptr()),(long)(f_input2.data_ptr())));
    // pre-alloc space to save graident
    output_grad.push_back(ig);
    count++;
    return f_output;
}  
    
template <typename NOPT>
  NtsVar runSelfNNOp(std::function<NtsVar(NtsVar &, int)> vertexforward,NtsVar& f_input,int layer_){//graph op
      
    static_assert(std::is_base_of<nts::op::ntsNNBaseOp,NOPT>::value,
                "template must be a type of graph op!");
    
    nts::op::ntsNNBaseOp * curr=new NOPT([&](NtsVar v_tensor,int layer_){
        return vertexforward(v_tensor,layer_);
    },layer_);
    NtsVar f_output=curr->forward(f_input);
    if (this->training == true) {
      NtsVar ig;
      op.push(SELFNNOP);
      output.push(f_output);
      input.push(f_input);
      ntsOp.push(ntsOperator(curr,SELFNNOP));
      iot_id.push_back(IOTensorId((long)(f_output.data_ptr()),(long)(f_input.data_ptr())));
      // pre-alloc space to save graident
      output_grad.push_back(ig);
      count++;
    }
    return f_output;
}  
  
  template <typename GOPT>
  NtsVar runGraphOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_,
        NtsVar &f_input){//graph op
      
    static_assert(std::is_base_of<nts::op::ntsGraphOp,GOPT>::value,
                "template must be a type of graph op!");
    
    nts::op::ntsGraphOp * curr=new GOPT(subgraphs_,graph_,layer_);
    NtsVar f_output=curr->forward(f_input);
    if (this->training == true) {
      NtsVar ig;
      op.push(GRAPHOP);
      output.push(f_output);
      input.push(f_input);
      ntsOp.push(ntsOperator(curr,GRAPHOP));
      iot_id.push_back(IOTensorId((long)(f_output.data_ptr()),(long)(f_input.data_ptr())));
      // pre-alloc space to save graident
      output_grad.push_back(ig);
      count++;
    }
    return f_output;
}  
  
 NtsVar runVertexForward(std::function<NtsVar(NtsVar &, NtsVar &)> vertexforward,
            NtsVar &nbr_input,NtsVar &vtx_input){//NNOP
//     LOG_INFO("call run vertex forward");
    NtsVar f_output=vertexforward(nbr_input,vtx_input); 
    if (this->training == true) {
        // 如果是训练模式，则记录nn算子便于反向传播
      appendNNOp(nbr_input, f_output);
    }
//    printf("tese %ld\n",(long)(&f_output));
    return f_output;
}
 NtsVar runVertexForward(std::function<NtsVar(NtsVar &)> vertexforward,
            NtsVar &nbr_input){//NNOP
//     LOG_INFO("call run vertex forward");
    NtsVar f_output=vertexforward(nbr_input); 
    if (this->training == true) {
      appendNNOp(nbr_input, f_output);
    }
    return f_output;
}
 
 NtsVar runEdgeForward(std::function<NtsVar(NtsVar &)> edgeforward,
            NtsVar &edge_input){//NNOP
//     LOG_INFO("call run vertex forward");
    NtsVar f_output=edgeforward(edge_input); 
    if (this->training == true) {
      appendNNOp(edge_input, f_output);
    }
    return f_output;
} 
  /**
   * 记录前向时的nn算子和loss的算子
   * @param input_t nn计算中的输入
   * @param output_t nn计算中的输出
   */
  void appendNNOp(NtsVar &input_t, NtsVar &output_t){
    assert(this->training);
    // if (!this->training) return;
    NtsVar ig;

    // we will chain the NNOP together, because torch lib will handle the backward propagation
    // when there is no graph operation
    // 因为nn的反向交给pytorch自己完成了，所以反向时只需要处理graph操作的反向
    // 我们将 NNOP 链接在一起，因为当没有图形操作时，torch lib 将处理反向传播
    // 因为最后一个nn操作调用backward就会进行之前的backward，所以当一个nn操作前还是nn操作的话，只调用最后一个就行，
    // 不过，如果nn操作前是一个图操作的话，由于图也会处理梯度，所以图操作完成之后，还需要再次调用该nn操作进行二次backward
    if ((count > 0 && op.top() == NNOP)&&((long)input_t.data_ptr())==iot_id[iot_id.size()-1].o_id) {
        // 将上一个nn操作记录在output中的tensor弹掉
        output.pop();
        // 将新的nn操作记录入栈中
        output.push(output_t);
     //    LOG_INFO("update DATA_PTR %ld",(long)output_t.data_ptr());
        // 更改最后的输出向量为该nn操作的输出向量
        iot_id[iot_id.size()-1].updateOutput((long)(output_t.data_ptr()));
    } else {
        // 如果前一个操作不是nn操作，直接记录该操作，不用进行pop
        op.push(NNOP);          // 记录操作的类型
        output.push(output_t);  // 记录nn操作的输出向量
        input.push(input_t);    // 记录nn操作的输入向量
        ntsOp.push(ntsOperator(NNOP));      // 记录nn操作
     //    LOG_INFO("inster DATA_PTR %ld",(long)output_t.data_ptr());
        // 记录nn操作的输入和输出数据部分的指针，分开两个记录，图操作中是合成一个记录
        iot_id.push_back(IOTensorId((long)(output_t.data_ptr()),(long)(input_t.data_ptr())));
        // pre-alloc space to save graident
        // 提前为梯度分配空间
        output_grad.push_back(ig);
        count++;
    }
  }
 
  void reset(){
    assert(count<=1);
    if(count==1&&ntsOp.top().op_t==GRAPHOP){
        delete ntsOp.top().op;
    }
    count = 0;
    std::stack<OpType>().swap(op);
    std::stack<NtsVar>().swap(output);
    std::stack<NtsVar>().swap(input);
    std::stack<ntsOperator>().swap(ntsOp);
    output_grad.clear();
    iot_id.clear();
}
  void pop_one_op(){
      // 依次弹出各个栈的内容，除了grad和iot_id
    if(ntsOp.top().op_t==GRAPHOP){
        delete ntsOp.top().op;
    }
    op.pop();
    output.pop();
    input.pop();
    ntsOp.pop();
    count--;
  }
  /**
   * 进行反向传播的函数
   * @param retain_graph 一次反向传播后是否保留计算图
   */
  void self_backward(bool retain_graph = true){
    assert(this->training);
    // if (!this->training) return;
    // 因为最后一个操作肯定是loss操作，所以直接调一个backward
    output.top().backward(torch::ones_like(output.top()), retain_graph);    // loss.backward
    output_grad[top_idx()-1]= input.top().grad();// grad of loss
    // 将该操作弹出，除了iot_id和output_grad之外的都进行了弹出
    pop_one_op();
//    LOG_INFO("FINISH LOSS");
    // 当反向栈中只有一个图操作时，就不需要再进行反向，图操作计算的是根号系数部分，第一层的根号系数部分不用再往回传
    while (count > 1 || (count == 1 && NNOP == op.top())) {
    // NNOP means we are using torch lib to do the forward computation
    // thus we can use auto diff framework in libtorch
//     LOG_INFO("FINISH %d",op.size());
    // 对于图操作的处理
    if (GRAPHOP == op.top()) {
//        LOG_INFO("FINISH Graph %d",op.size());  
      int preop_id=top_idx();
      // 如果grad的维度小于2，就是最后层计算得到的梯度
      if(output_grad[top_idx()].dim()<2){
          output_grad[top_idx()]=output.top().grad();
      }//determine o grad
      // 下面是获取要写入的梯度的位置
      for(preop_id=top_idx();preop_id>=0;preop_id--){
          // 一直递减直到找到前一个的输出为栈顶的输入的位置，然后将梯度写入该位置
          if(iot_id[preop_id].o_id==iot_id[top_idx()].i_id1)
              break;
      }// where to write i grad
      // 调用图操作类的反向操作函数
      output_grad[preop_id]=ntsOp.top().get_graphop()->backward(output_grad[top_idx()]);
   //   LOG_INFO("input id %ld %d %d",preop_id,top_idx(),output_grad[preop_id].dim());
//stable      
//      output_grad[top_idx()-1]=ntsOp.top().op->backward(output_grad[top_idx()]);
      // 弹出图操作
      pop_one_op();
    }else if (BIGRAPHOP == op.top()) { // test
//          LOG_INFO("FINISH BIGRAPHOP %d",op.size());  
      int preop_id=top_idx(); 
      if(output_grad[top_idx()].dim()<2){
          output_grad[top_idx()]=output.top().grad();
      }//determine o grad
      for(preop_id=top_idx();preop_id>=0;preop_id--){
          if(iot_id[preop_id].o_id==iot_id[top_idx()].i_id1)
              break;
      }// where to write i grad
  //    LOG_INFO("15 bug %d",preop_id);
        // 调用图操作类的反向操作函数
      output_grad[preop_id]=ntsOp.top().get_graphop()->backward(output_grad[top_idx()]);
      
      preop_id=top_idx();
      for(preop_id=top_idx();preop_id>=0;preop_id--){
          if(iot_id[preop_id].o_id==iot_id[top_idx()].i_id2)
              break;
      }
      output_grad[preop_id]=ntsOp.top().get_graphop()->get_additional_grad();
      
      
//stable      
//      output_grad[top_idx()-1]=ntsOp.top().op->backward(output_grad[top_idx()]);
      pop_one_op();
    }else if (SELFNNOP == op.top()) { // test
//          LOG_INFO("FINISH SELF_NN %d",op.size());  
      int preop_id=top_idx(); 
      if(output_grad[top_idx()].dim()<2){
          output_grad[top_idx()]=output.top().grad();
      }//determine o grad
      for(preop_id=top_idx();preop_id>=0;preop_id--){
          if(iot_id[preop_id].o_id==iot_id[top_idx()].i_id1)
              break;
      }// where to write i grad
      output_grad[preop_id]=ntsOp.top().get_nnop()->backward(output_grad[top_idx()]);
//stable      
//      output_grad[top_idx()-1]=ntsOp.top().op->backward(output_grad[top_idx()]);
      pop_one_op();
    }  else if (NNOP == op.top()) {// directly use pytorch
//        LOG_INFO("FINISH nn %d",op.size());
        // 如果是最后一层的nn操作，直接计算梯度并保存，因为一般是nn-->loss，反向时倒过来，这时的nn就不用再进行反向了
      if(output_grad[top_idx()].dim()<2){
          output_grad[top_idx()]=output.top().grad();
      }//determine o grad
      // 如果不是最后一层的梯度，调用其backward对栈顶梯度进行反向传播
      if(output_grad[top_idx()].dim()>1){
        assert(output_grad[top_idx()].size(1)==output.top().size(1));
        assert(output_grad[top_idx()].size(0)==output.top().size(0)); 
        output.top().backward(output_grad[top_idx()], retain_graph);
      }
      
      pop_one_op();
  //  LOG_INFO("FINISH NN OP");
    } else {
      LOG_INFO("NOT SUPPORT OP");
      assert(true);
    }
  }
    reset();  
  }
  void debug(){
    printf("ADDEBUG input.size()%d\n", input.size());
    // for(int i=0;i<count;i++){
    int i=0;
     for(int k=0;k<iot_id.size();k++){
        LOG_INFO("IOT %ld %ld",iot_id[k].i_id1,iot_id[k].o_id);
    }
    while (!input.empty()) {
        if(i==0){
          LOG_INFO("input dim %d %d\t output dim %d \t OP type %d", input.top().size(0),input.top().size(1),output.top().dim(),op.top());
        }else{
          LOG_INFO("input dim %d %d \t output dim %d %d\t OP type %d", input.top().size(0),
                  input.top().size(1),output.top().size(0),output.top().size(1),op.top());  
        }
        input.pop();
        output.pop();
        op.pop();
        ntsOp.pop();
        i++;
    }
    this->output_grad.clear();
    count=0;
  }
  
  
  int top_idx(){
    return count - 1;
  }

  void train() {
    this->training = true;
  }

  void eval() {
    this->training = false;
  }

//private:
  std::stack<OpType> op;        // 记录操作的类型
  std::stack<NtsVar> output;    // 记录操作的输出tensor
  std::stack<NtsVar> input;     // 记录操作的输入tensor
  std::stack<ntsOperator> ntsOp;// 记录进行操作的类的实例或者是否是nn操作
  std::vector<NtsVar> output_grad;  // 存储反向的梯度
  std::vector<IOTensorId> iot_id;   // 存储指向输入和输出tensor的数据部分的指针
  int count;    // 记录前向操作的数量
  bool training; // specify training or evaluation mode.
//  GraphOperation *gt;
//  std::vector<CSC_segment_pinned *> subgraphs;
//  bool bi_direction;
};

} // namespace autodiff
} // namespace nts

#endif
