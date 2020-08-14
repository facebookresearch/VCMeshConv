//
// Created by zhouyi on 6/14/19.
//

#ifndef POINTSAMPLING_MESHCNN_H
#define POINTSAMPLING_MESHCNN_H

#endif //POINTSAMPLING_MESHCNN_H

#include "meshPooler.h"
#include "cnpy/cnpy.h"


//gives you an fully convolutional auto-encoder. The pooling and unpooling strides are symmetric.
class MeshCNN
{
public:
    Mesh _mesh;
    vector<MeshPooler> _meshPoolers;

    MeshCNN(Mesh mesh) {_mesh = mesh;}

    //stride==1 means no pool
    void add_pool_layer(int stride, int radius_pool, int radius_unpool)
    {
        cout<<"## Add Pooling "<<_meshPoolers.size()<<"\n";//<<". stride: "<<stride<<" radius_pool: "<<radius_pool<<" radius_unpool: "<<radius_unpool<<"\n";
        if(_meshPoolers.size()==0)
        {
            MeshPooler meshPooler;
            meshPooler.set_connection_map_from_mesh(_mesh);
            meshPooler.set_must_include_center_lst_from_mesh(_mesh);
            meshPooler.compute_pool_and_unpool_map(stride, radius_pool, radius_unpool);
            _meshPoolers.push_back(meshPooler);
            return;
        }

        MeshPooler meshPooler;
        meshPooler._connection_map = _meshPoolers.back()._center_center_map; //set the last pooler's center_center_map as the connection map of the current one.
        meshPooler._must_include_center_lst=_meshPoolers.back()._must_include_center_lst_new_index;
        //cout<<"must include centers num: "<<meshPooler._must_include_center_lst.size()<<"\n";
        meshPooler.compute_pool_and_unpool_map(stride, radius_pool,radius_unpool);
        _meshPoolers.push_back(meshPooler);
        return;

    }


    //neighbour_lst_lst  current_size*(1+max_neighbor_num*2).
    //               neighbor_num, (neighbor_id0,dist0), (neighbor_id1,dist1) ..., (neighbor_idx,distx), (previous_size,-1), ..., (previous_size,-1)
    vector<int> get_neighborID_lst_lst(const vector<vector<Int2>> &pool_map, int previous_size)
    {
        int current_point_num = pool_map.size();

        //first count the maximum number of neighbors
        int max_neighbor_num = 0;
        for (int i=0;i<pool_map.size();i++)
        {
            int neighbor_num = pool_map[i].size();
            if(neighbor_num > max_neighbor_num)
                max_neighbor_num = neighbor_num;
        }

        vector<int> neighborID_lst_lst_flat;

        for (int i=0;i<pool_map.size();i++)
        {
            vector<int> neighborID_lst = vector<int>(max_neighbor_num*2);
            for (int j =0; j <pool_map[i].size(); j++)
            {
                neighborID_lst[j*2] = pool_map[i][j][0]; // id
                neighborID_lst[j*2+1] =pool_map[i][j][1];// dist

            }
            if(pool_map[i].size()<max_neighbor_num)
            {
                for (int j=pool_map[i].size(); j<max_neighbor_num;j++) {
                    neighborID_lst[j * 2] = previous_size;
                    neighborID_lst[j * 2+1] = -1;
                }
            }
            neighborID_lst_lst_flat.push_back(pool_map[i].size());

            neighborID_lst_lst_flat.insert(neighborID_lst_lst_flat.end(), neighborID_lst.begin(), neighborID_lst.end());

        }

        return neighborID_lst_lst_flat;
    }




    //neighbour_lst_lst  current_size*max_neighbor_num.
    //               neighbor_num, neighbor_id0, neighbor_id1, ..., neighbor_idx, previous_size, ..., previous_size
    void save_pool_and_unpool_neighbor_info_to_npz(const string& save_path)
    {
        cout<<"Save pool and unpool neighbor info to npz.\n";
        for(int i=0;i<_meshPoolers.size(); i++)
        {
            int after_pool_size = (_meshPoolers[i]._center_center_map.size());
            int before_pool_size = (_meshPoolers[i]._connection_map.size());

            vector<int> pool_neighborID_lst_lst = get_neighborID_lst_lst(_meshPoolers[i]._pool_map, before_pool_size);

            vector<int> unpool_neighborID_lst_lst = get_neighborID_lst_lst(_meshPoolers[i]._unpool_map, after_pool_size);


            cout<<"save pool "<<to_string(i)<<".\n";
            int neighbor_num_pool_2 = pool_neighborID_lst_lst.size()/after_pool_size-1;
            
            std::vector<size_t > shape_info = {(size_t)after_pool_size, (size_t)(1+neighbor_num_pool_2) };
            cout<<shape_info[0]<<" "<<shape_info[1]<<"\n";
            //cout<< pool_neighborID_lst_lst.size()<<"\n";

            //cout<<save_path+"_pool"+to_string(i)+".npy"<<"\n";

            cnpy::npy_save(save_path+"_pool"+to_string(i)+".npy", &pool_neighborID_lst_lst [0], shape_info, "w");//"a" appends to the file we created above


            cout<<"save unpool "<<to_string(i)<<".\n";
            int neighbor_num_unpool_2 = unpool_neighborID_lst_lst.size()/before_pool_size-1;
            shape_info = {(size_t)before_pool_size, (size_t)(1+neighbor_num_unpool_2) };
            cout<<shape_info[0]<<" "<<shape_info[1]<<"\n";

            cnpy::npy_save(save_path+"_unpool"+to_string(i)+".npy", &unpool_neighborID_lst_lst [0], shape_info, "w");//"a" appends to the file we created above

            cout<<"save old to new index list"<<to_string(i)<<".\n";
            shape_info = {(size_t)after_pool_size};
            cout<<shape_info[0]<<" "<<shape_info[1]<<"\n";
            cnpy::npy_save(save_path+"_center_lst"+to_string(i)+".npy", &_meshPoolers[i]._center_lst, shape_info, "w");

        }


    }


    /*
     *
    //map, per concerned center, [p_id_in_previous_layer, radius]
    //current_size map.size()
    //matrix_flat vector<float> to_size*previous_size
    vector<float> get_flatten_connection_matrix(const vector<vector<Int2>> &map, const int previous_size)
    {
        vector<float> matrix_flat; //current_size*previous_size
        for(int i=0;i<map.size();i++)
        {
            vector<float> line(previous_size);

            const vector<Int2> *raw_line = &map[i];

            for (int i=0;i<raw_line->size();i++)
            {
                int p_id = (*raw_line)[i][0];
                int radius = (*raw_line)[i][1];
                float value = 1.0/float(pow(2, double(radius)));
                line[p_id] = value;
            }
            matrix_flat.insert(matrix_flat.end(), line.begin(), line.end());

        }

        return matrix_flat;
    }

    //save to npz
    //layer_num
    //pool1 float current_size*previous_size
    //unpool1 float current_size*previous_size
    //...
    //value is 1/(radius^2)
    void save_pool_and_unpool_matrices_to_npz(const string& save_path)
    {
        cout<<"Save pool and unpool matrices to npz.\n";
        int layer_num = _meshPoolers.size();

        for(int i=0;i<_meshPoolers.size(); i++)
        {
            int current_size = (_meshPoolers[i]._center_center_map.size());
            int previous_size = (_meshPoolers[i]._connection_map.size());
            std::vector<size_t > shape_info = {(size_t)current_size, (size_t)previous_size};
            cout<<shape_info[0]<<" "<<shape_info[1]<<"\n";

            cout<<"save pool "<<to_string(i)<<".\n";

            vector<float> flatten_pool_matrix = get_flatten_connection_matrix(_meshPoolers[i]._pool_map, previous_size);
            cnpy::npy_save(save_path+"_pool"+to_string(i)+".npy", &flatten_pool_matrix[0], shape_info, "w");//"a" appends to the file we created above

            cout<<"save unpool "<<to_string(i)<<".\n";

            vector<float> flatten_unpool_matrix = get_flatten_connection_matrix(_meshPoolers[i]._unpool_map, previous_size);

            cnpy::npy_save(save_path+"_unpool"+to_string(i)+".npy", &flatten_unpool_matrix[0], shape_info, "w");//"a" appends to the file we created above

        }
    }
     */


};
