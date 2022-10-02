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
#ifndef NTSSAMPLER_HPP
#define NTSSAMPLER_HPP
#include <mutex>
#include <random>
#include <cmath>
#include <stdlib.h>
#include "FullyRepGraph.hpp"
class Sampler{
public:
    std::vector<SampledSubgraph*> work_queue;// excepted to be single write multi read
    std::mutex queue_start_lock;
    int queue_start;
    std::mutex queue_end_lock;
    int queue_end;
    FullyRepGraph* whole_graph;
    VertexId start_vid,end_vid;
    VertexId work_range[2];
    VertexId work_offset;
    std::vector<VertexId> sample_nids;
    Sampler(FullyRepGraph* whole_graph_, VertexId work_start,VertexId work_end){
        whole_graph=whole_graph_;
        queue_start=-1;
        queue_end=0;
        work_range[0]=work_start;
        work_range[1]=work_end;
        work_offset=work_start;
    }
    Sampler(FullyRepGraph* whole_graph_, std::vector<VertexId>& index){
        assert(index.size() > 0);
        sample_nids.assign(index.begin(), index.end());
        assert(sample_nids.size() == index.size());
        whole_graph=whole_graph_;
        queue_start=-1;
        queue_end=0;
        work_range[0]=0;
        work_range[1]=sample_nids.size();
        work_offset=0;
    }
    ~Sampler(){
        clear_queue();
    }
    bool has_rest(){
        bool condition=false;
        int cond_start=0;
        queue_start_lock.lock();
        cond_start=queue_start;
        queue_start_lock.unlock();
        
        int cond_end=0;
        queue_end_lock.lock();
        cond_end=queue_end;
        queue_end_lock.unlock();
       
        condition=cond_start<cond_end&&cond_start>=0;
        return condition;
    }
//    bool has_rest(){
//        bool condition=false;
//        condition=queue_start<queue_end&&queue_start>=0;
//        return condition;
//    }
    SampledSubgraph* get_one(){
//        while(true){
//            bool condition=queue_start<queue_end;
//            if(condition){
//                break;
//            }
//         __asm volatile("pause" ::: "memory");  
//        }
        queue_start_lock.lock();
        VertexId id=queue_start++;
        queue_start_lock.unlock();
        assert(id<work_queue.size());
        return work_queue[id];
    }
    void clear_queue(){
        for(VertexId i=0;i<work_queue.size();i++){
            delete work_queue[i];
        }
        work_queue.clear();
    } 
    bool sample_not_finished(){
        return work_offset<work_range[1];
    }
    void restart(){
        work_offset=work_range[0];
        queue_start=-1;
        queue_end=0;
    }
    void reservoir_sample(int layers_, int batch_size_,std::vector<int> fanout_){
        assert(work_offset<work_range[1]);
        int actual_batch_size=std::min((VertexId)batch_size_,work_range[1]-work_offset);
        SampledSubgraph* ssg=new SampledSubgraph(layers_,fanout_);  
        
        for(int i=0;i<layers_;i++){
            ssg->sample_preprocessing(i);
            //whole_graph->SyncAndLog("preprocessing");
            if(i==0){
              ssg->sample_load_destination([&](std::vector<VertexId>& destination){
                  for(int j=0;j<actual_batch_size;j++){
                    //   destination.push_back(work_offset++);
                    destination.push_back(sample_nids[work_offset++]);
                  }
              },i);
              //whole_graph->SyncAndLog("sample_load_destination");
            }else{
               ssg->sample_load_destination(i); 
              //whole_graph->SyncAndLog("sample_load_destination2");
            }
            ssg->init_co([&](VertexId dst){
                VertexId nbrs=whole_graph->column_offset[dst+1]
                                 -whole_graph->column_offset[dst];
            return (nbrs>fanout_[i]) ? fanout_[i] : nbrs;
            },i);
            ssg->sample_processing([&](VertexId fanout_i,
                    VertexId dst,
                    std::vector<VertexId> &column_offset,
                        std::vector<VertexId> &row_indices,VertexId id){
                for(VertexId src_idx=whole_graph->column_offset[dst];
                        src_idx<whole_graph->column_offset[dst+1];src_idx++){
                    //ReservoirSampling
                    VertexId write_pos=(src_idx-whole_graph->column_offset[dst]);
                    if(write_pos<fanout_i){
                        write_pos+=column_offset[id];
                        row_indices[write_pos]=whole_graph->row_indices[src_idx];
                    }else{
                        VertexId random=rand()%write_pos;
                        if(random<fanout_i){
                          row_indices[random+column_offset[id]]=  
                                  whole_graph->row_indices[src_idx];
                        }
                    }
                }
            });
            //whole_graph->SyncAndLog("sample_processing");
            ssg->sample_postprocessing();
            //whole_graph->SyncAndLog("sample_postprocessing");
        }
        work_queue.push_back(ssg);
        queue_end_lock.lock();
        queue_end++;
        queue_end_lock.unlock();
        if(work_queue.size()==1){
            queue_start_lock.lock();
            queue_start=0;
            queue_start_lock.unlock();
        }
    }
};


class FullBatchSampler{
private:
    int sample_limit;
    std::atomic_flag sample_flag = ATOMIC_FLAG_INIT;
    inline void deleteSampleGraph() {
       if(using_sample_graph == nullptr) {
           return;
       }
       delete []using_sample_graph->row_offset;
       delete []using_sample_graph->column_indices;
       delete []using_sample_graph->column_offset;
       delete []using_sample_graph->row_indices;
       delete using_sample_graph;
       using_sample_graph = nullptr;
    }
public:
    int sample_total;
    int sample_count;
    float sample_rate;
    std::queue<std::vector<CSC_segment_pinned*>> sample_graphs;
//    std::vector<CSC_segment_pinned> sample_graphs;
    std::mutex queue_mutex;
    PartitionedGraph* partitionedGraph;
    std::vector<CSC_segment_pinned*> total_graph;
    CSC_segment_pinned* using_sample_graph;


    FullBatchSampler(PartitionedGraph* partitionedGraph1, int epoch_num, float sample_rate) {
        this->partitionedGraph = partitionedGraph1;
        this->total_graph = partitionedGraph1->graph_chunks;
        this->using_sample_graph = nullptr;
        this->sample_total = epoch_num;
        this->sample_count = 0;
        this->sample_limit = 20;
        this->sample_rate = sample_rate;
        std::thread(&FullBatchSampler::sampling, this);
    }
    ~FullBatchSampler(){
//        auto partition_id = partitionedGraph->partition_id;
        deleteSampleGraph();

    }



    std::vector<CSC_segment_pinned*> get_one() {
        deleteSampleGraph();
//        if(sample_count == sample_total) {
//            return std::vector<CSC_segment_pinned*>{};
//        }
        while(sample_graphs.size() < sample_limit/2+1 && sample_count < sample_total) {
            if(!sample_flag.test_and_set()) {
                std::thread(&FullBatchSampler::sampling, this);
                break;
            }
        }
        std::vector<CSC_segment_pinned*> return_graph = std::vector<CSC_segment_pinned*>{};
        std::unique_lock<std::mutex> uniqueLock(queue_mutex);
        uniqueLock.lock();
        if(sample_graphs.size() != 0) {
            return_graph = sample_graphs.front();
            using_sample_graph = return_graph[partitionedGraph->partition_id];
            sample_graphs.pop();
        }
        uniqueLock.unlock();
        return return_graph;
    }

    bool sample_not_finished(){
        return sample_count != sample_total;
    }

    void sampling() {
        if(sample_flag.test_and_set()) {
            return;
        }
        while(sample_count < sample_total && sample_graphs.size() < sample_limit) {
            reservoir_sample(sample_rate);
            sample_count++;
        }
        sample_flag.clear();
    }

    void reservoir_sample(float sample_rate){
        CSC_segment_pinned* subgraph = total_graph[partitionedGraph->partition_id];
        auto partition_id = partitionedGraph->partition_id;

        std::vector<CSC_segment_pinned*> sample_graph = total_graph;
        auto partition_verts = partitionedGraph->partition_offset[partition_id + 1] -
                partitionedGraph->partition_offset[partition_id];
        auto partition_start = partitionedGraph->partition_offset[partition_id];
        std::vector<std::vector<VertexId>> adjList, backwardAdjList;
        adjList.resize(partition_verts);
        backwardAdjList.resize(partition_verts);
        std::mt19937 randGen(std::random_device{}());
        EdgeId edgeNum = 0;

//        subgraph->allocEdgeAssociateData();

//        LOG_INFO("遍历边之前");
        for(int i = 0; i < partition_verts; i++) {
            for(int j = subgraph->row_offset[i]; j < subgraph->row_offset[i+1]; j++) {
                assert(subgraph->column_indices[j] <= partitionedGraph->partition_offset[partition_id + 1] &&
                        subgraph->column_indices[j] >= partitionedGraph->partition_offset[partition_id]);
                if(randGen() % 100 < sample_rate * 100) {
                    adjList[i].push_back(subgraph->column_indices[j]);
                    backwardAdjList[subgraph->column_indices[j] - partition_start].push_back(i + partition_start);
                    edgeNum++;
                }
            }
        }

//        for(int i = 0; i < partition_verts; i++) {
//            assert(subgraph->row_offset[i+1] == subgraph->column_offset[i+1]);
//            for(auto j = subgraph->row_offset[i]; j < subgraph->row_offset[i+1];j++) {
//                auto csc_start = subgraph->row_offset[i];
//                auto csc_end = subgraph->column_indices[j];
//
//                bool isFind = false;
//                for(auto k = subgraph->column_offset[csc_end]; k < subgraph->column_offset[csc_end+1];k++) {
//                    if(subgraph->row_indices[k] == csc_start) {
//                        isFind = true;
//                        break;
//                    }
//                }
//                assert(isFind);
//            }
//        }
//        LOG_INFO("遍历边之后");
        VertexId *column_offset = new VertexId[partition_verts + 1];
        VertexId *row_indices = new VertexId[edgeNum];
        VertexId *row_offset = new VertexId[partition_verts + 1];
        VertexId *column_indices = new VertexId[edgeNum];
        LOG_INFO("边的总数为：%lu, 采样的边的数量: %lu", subgraph->edge_size, edgeNum);
        if(partition_verts > 0) {
            column_offset[0] = 0;
            row_offset[0] = 0;
        }
        for(int i = 0; i < partition_verts; i++) {
            row_offset[i+1] = row_offset[i] + backwardAdjList[i].size();
            column_offset[i+1] = column_offset[i] + adjList[i].size();
//            assert(row_offset[i+1] == subgraph->row_offset[i+1]);
//            assert(column_offset[i+1] == subgraph->column_offset[i+1]);
//            LOG_INFO("第%d次复制开始, back size: %lu, size: %lu", i, backwardAdjList[i].size(), adjList[i].size());
//            LOG_INFO("row_offset[%d]: %u, column_offset[%d]: %u",i, row_offset[i], i, column_offset[i]);
            memcpy(&column_indices[row_offset[i]], backwardAdjList[i].data(),
                   sizeof(VertexId) * (backwardAdjList[i].size()));
            memcpy(&row_indices[column_offset[i]], adjList[i].data(),
                   sizeof(VertexId) * (adjList[i].size()));
//            LOG_INFO("第%d次复制完成", i);
        }
//        LOG_INFO("复制之后");


        CSC_segment_pinned* cscSegmentPinned = new CSC_segment_pinned{};
        cscSegmentPinned->batch_size_forward = subgraph->batch_size_forward;
        cscSegmentPinned->batch_size_backward = subgraph->batch_size_backward;
        cscSegmentPinned->source_active = subgraph->source_active;
        cscSegmentPinned->destination_active = subgraph->destination_active;
        cscSegmentPinned->row_offset = row_offset;
        cscSegmentPinned->column_indices = column_indices;
        cscSegmentPinned->column_offset = column_offset;
        cscSegmentPinned->row_indices = row_indices;

//        cscSegmentPinned->row_offset = subgraph->row_offset;
//        cscSegmentPinned->column_indices = subgraph->column_indices;
//        cscSegmentPinned->column_offset = subgraph->column_offset;
//        cscSegmentPinned->row_indices = subgraph->row_indices;
//        cscSegmentPinned->source_active = subgraph->source_active;
//        cscSegmentPinned->destination_active = subgraph->destination_active;

        sample_graph[partition_id] = cscSegmentPinned;
//        LOG_INFO("赋值之后");

        std::unique_lock<std::mutex> uniqueLock(queue_mutex);
        uniqueLock.lock();
        sample_graphs.push(sample_graph);
        uniqueLock.unlock();
//        LOG_INFO("存储子图");

    }
};



#endif