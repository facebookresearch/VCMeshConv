//
// Created by zhouyi on 6/13/19.
//

#ifndef POINTSAMPLING_MESHPOOLER_VISUALIZER_H
#define POINTSAMPLING_MESHPOOLER_VISUALIZER_H

#endif //POINTSAMPLING_MESHPOOLER_VISUALIZER_H


//#include "meshPooler.h"
#include "meshCNN.h"

class MeshPooler_Visualizer
{
public:

    void save_colored_obj_pool_receptive_field(const string &filename, Mesh mesh, MeshPooler meshPooler)
    {
        cout<<"save colored obj for visualizing pooling receptive field\n";
        vector<vector< Vec3<float> >> points_colors;
        points_colors.resize(mesh.points.size());

        //set each receptive field a random color
        vector<vector<Int2>> pool_map = meshPooler._pool_map;

        for (int i=0;i<pool_map.size();i++)
        {
            vector<Int2> connection_lst = pool_map[i];
            float r = rand() / (float)RAND_MAX  ;
            float g = rand() / (float)RAND_MAX  ;
            float b = rand() / (float)RAND_MAX  ;
            Vec3<float> color = Vec3<float>(r,g,b);
            //cout<<connection_lst.size()<<"\n";
            for(int j=0;j<connection_lst.size();j++)
            {
                int q = connection_lst[j][0];
                points_colors[q].push_back(color);
            }
        }

        vector< Vec3<float> > colors;
        colors.resize(mesh.points.size());

        for(int i=0;i<points_colors.size();i++)
        {
            vector<Vec3<float>> point_colors = points_colors[i];
            Vec3<float> color=Vec3<float>(0,0,0);
            for (int j=0;j<point_colors.size();j++)
            {
                color = color + point_colors[j];
            }
            color = color / float(point_colors.size());
            colors[i] = color;
        }


        mesh.SaveOBJ(filename, mesh.points, colors, mesh.triangles);

    }

    void save_colored_obj_pool_receptive_field(const string &filename, Mesh mesh, vector<vector<int>> receptive_map)
    {
        cout<<"save colored obj for visualizing pooling receptive field\n";
        vector<vector< Vec3<float> >> points_colors;
        points_colors.resize(mesh.points.size());


        for (int i=0;i<receptive_map.size();i++)
        {
            vector<int> connection_lst = receptive_map[i];
            float r = rand() / (float)RAND_MAX  ;
            float g = rand() / (float)RAND_MAX  ;
            float b = rand() / (float)RAND_MAX  ;
            Vec3<float> color = Vec3<float>(r,g,b);
            //cout<<connection_lst.size()<<"\n";
            for(int j=0;j<connection_lst.size();j++)
            {
                int q = connection_lst[j];
                points_colors[q].push_back(color);
            }
        }

        vector< Vec3<float> > colors;
        colors.resize(mesh.points.size());

        for(int i=0;i<points_colors.size();i++)
        {
            vector<Vec3<float>> point_colors = points_colors[i];
            Vec3<float> color=Vec3<float>(0,0,0);
            for (int j=0;j<point_colors.size();j++)
            {
                color = color + point_colors[j];
            }
            color = color / float(point_colors.size());
            colors[i] = color;
        }

        //set center to red

        /*
        for(int i =0;i<receptive_map.size();i++)
        {
            int center_id = receptive_map[i][0];
            colors[center_id] = Vec3<float>(1,0,0);
        }*/




        mesh.SaveOBJ(filename, mesh.points, colors, mesh.triangles);

    }


    vector<int> get_connected_points_in_original_mesh( int index, int layer, MeshCNN &meshCNN)
    {
        vector<int> total_receptive_lst;

        vector<vector<Int2>> *pool_map = &meshCNN._meshPoolers[layer]._pool_map;
        //vector<int> center2old_index_lst = meshCNN._meshPoolers[layer]._center_lst;

        vector<Int2> receptive2_lst = (*pool_map)[index];

        vector<int> receptive_lst;
        for(int j=0;j<receptive2_lst.size();j++)
            receptive_lst.push_back(receptive2_lst[j][0]);

        if(layer==0)
        {

            total_receptive_lst = receptive_lst;
            return total_receptive_lst;
        }

        for(int j=0;j<receptive_lst.size();j++)
        {
            int index = receptive_lst[j];
            vector<int> sub_receptive_lst = get_connected_points_in_original_mesh(index, layer-1, meshCNN);
            total_receptive_lst.insert(total_receptive_lst.end(), sub_receptive_lst.begin(), sub_receptive_lst.end());
        }

        return total_receptive_lst;


    }

    void save_colored_obj_pool_receptive_field_for_whole_cnn(const string &file_dir, Mesh mesh, MeshCNN meshCNN)
    {

        for (int n=0;n<meshCNN._meshPoolers.size();n++)
        {
            MeshPooler meshPooler = meshCNN._meshPoolers[n];

            vector<vector<int>> receptive_map;

            for(int i=0;i<meshPooler._pool_map.size(); i++)
            {
                vector<int> i_receptive_lst=get_connected_points_in_original_mesh(i, n, meshCNN);
                receptive_map.push_back(i_receptive_lst);
            }
            cout<<"save colored pool\n";

            save_colored_obj_pool_receptive_field(file_dir + to_string(n+1) +".obj", mesh, receptive_map);

        }

    }

    /*
    void save_colored_obj_pool_receptive_field_for_whole_cnn_old(const string &file_dir, Mesh mesh, MeshCNN meshCNN)
    {
        MeshPooler meshPooler = meshCNN._meshPoolers[0];
        vector<int> new2old_index_lst =  meshPooler._center_lst;
        save_colored_obj_pool_receptive_field(file_dir + to_string(1) +".obj", mesh, meshPooler);

        for (int n=1;n<meshCNN._meshPoolers.size();n++)
        {
            MeshPooler meshPooler = meshCNN._meshPoolers[n];
            vector<int> current_new2old_index_lst;
            for (int i=0;i<meshPooler._center_lst.size();i++)
            {
                int original_index = new2old_index_lst[meshPooler._center_lst[i]];
                current_new2old_index_lst.push_back(original_index);
            }
            meshPooler._center_lst = current_new2old_index_lst;


            //reindex _pool_map and __unpool_map
            vector<vector<Int2>> pool_map, unpool_map;
            for (int i=0; i<meshPooler._pool_map.size();i++)
            {
                vector<Int2> connection_lst = meshPooler._pool_map[i];
                for(int j=0;j<connection_lst.size();j++)
                {
                    int index = connection_lst[j][0];
                    connection_lst[j][0] = new2old_index_lst[index];
                }
                pool_map.push_back(connection_lst);
            }

            for (int i=0; i<meshPooler._unpool_map.size();i++)
            {
                vector<Int2> connection_lst = meshPooler._unpool_map[i];
                for(int j=0;j<connection_lst.size();j++)
                {
                    int index = connection_lst[j][0];
                    connection_lst[j][0] = new2old_index_lst[index];
                }
                unpool_map.push_back(connection_lst);
            }

            new2old_index_lst = current_new2old_index_lst;

            save_colored_obj_pool_receptive_field(file_dir + to_string(n+1) +".obj", mesh, meshPooler);


        }

    }
    */


    void save_obj_with_colored_sample_points(const string &filename, Mesh mesh, vector<int> center_lst)
    {

        cout<<"save colored result\n";
        vector< Vec3<float> > colors;
        for (int i=0;i<mesh.points.size();i++)
        {
            Vec3<float> color = Vec3<float>(0.7,0.7,0.7);
            colors.push_back(color);
        }
        for (int i=0;i<center_lst.size();i++)
        {
            Vec3<float> color = Vec3<float>(1,0,0);
            colors[center_lst[i]] =color;
        }

        mesh.SaveOBJ(filename, mesh.points, colors, mesh.triangles);
    }

    void save_obj_with_colored_sample_points_all_layers(const string &path, Mesh mesh, MeshCNN meshCNN)
    {
        vector<int> last_center_lst;
        for (int i=0;i<mesh.points.size();i++)
            last_center_lst.push_back(i);
        for(int i=0;i<meshCNN._meshPoolers.size();i++)
        {
            vector<int> current_center_lst = meshCNN._meshPoolers[i]._center_lst;
            for(int j=0;j<current_center_lst.size();j++)
            {
                int last_layer_index= current_center_lst[j];
                current_center_lst[j] = last_center_lst[last_layer_index];
            }

            last_center_lst=current_center_lst;

            save_obj_with_colored_sample_points(path+to_string(i)+".obj", mesh, current_center_lst);

            //if(i == (meshCNN._meshPoolers.size() -1) )
            {
                std::string txtfilename = path+to_string(i)+".txt";
                std::cout<<txtfilename<<std::endl;
                FILE *pfile = fopen(txtfilename.c_str(),"w");
                for (int i=0;i<current_center_lst.size();i++)
                {
                    int tmpind = current_center_lst[i];
                    //std::cout<<tmpind<<" ";
                    fprintf(pfile,"%d\n",tmpind);
           
                }
                fclose(pfile);
                //std::cout<<std::endl;
            }  

        }

    }


};