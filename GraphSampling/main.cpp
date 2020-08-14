
#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>
//#include "meshPooler.h"
#include "meshPooler_visualizer.h"


void set_7k_mesh_layers(MeshCNN &meshCNN)
{
    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(2,2,2);//1
    meshCNN.add_pool_layer(1,1,1);//2
    meshCNN.add_pool_layer(2,2,2);//3
    meshCNN.add_pool_layer(1,1,1);//4
    meshCNN.add_pool_layer(2,2,2);//5
    meshCNN.add_pool_layer(1,1,1);//6
    meshCNN.add_pool_layer(2,2,2);//7
    meshCNN.add_pool_layer(1,1,1);//8
    meshCNN.add_pool_layer(2,2,2);//9
    meshCNN.add_pool_layer(1,1,1);//10
    meshCNN.add_pool_layer(2,2,2);//11
    meshCNN.add_pool_layer(1,1,1);//12
}


void set_20k_mesh_layers(MeshCNN &meshCNN)
{
    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(2,2,2);//1
    meshCNN.add_pool_layer(1,1,1);//2
    meshCNN.add_pool_layer(2,2,2);//3
    meshCNN.add_pool_layer(1,1,1);//4
    meshCNN.add_pool_layer(2,2,2);//5
    meshCNN.add_pool_layer(1,1,1);//6
    meshCNN.add_pool_layer(2,2,2);//7
    meshCNN.add_pool_layer(1,1,1);//8
    meshCNN.add_pool_layer(2,2,2);//9
    meshCNN.add_pool_layer(1,1,1);//10
    meshCNN.add_pool_layer(2,2,2);//11
    meshCNN.add_pool_layer(1,1,1);//12
    meshCNN.add_pool_layer(2,2,2);//13
    meshCNN.add_pool_layer(1,1,1);//14

}




void set_70k_mesh_layers(MeshCNN &meshCNN)
{
    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(2,2,2);//1
    meshCNN.add_pool_layer(1,1,1);//2
    meshCNN.add_pool_layer(2,2,2);//3
    meshCNN.add_pool_layer(1,1,1);//4
    meshCNN.add_pool_layer(2,2,2);//5
    meshCNN.add_pool_layer(1,1,1);//6
    meshCNN.add_pool_layer(2,2,2);//7
    meshCNN.add_pool_layer(1,1,1);//8
    meshCNN.add_pool_layer(2,2,2);//9
    meshCNN.add_pool_layer(1,1,1);//10
    meshCNN.add_pool_layer(2,2,2);//11
    meshCNN.add_pool_layer(1,1,1);//12
    meshCNN.add_pool_layer(2,2,2);//13
    meshCNN.add_pool_layer(1,1,1);//14
    meshCNN.add_pool_layer(2,2,2);//15
    meshCNN.add_pool_layer(1,1,1);//16
    meshCNN.add_pool_layer(2,2,2);//17
    meshCNN.add_pool_layer(1,1,1);//18
    

}


void set_500k_mesh_layers(MeshCNN &meshCNN)
{

    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(3,2,3);//1
    meshCNN.add_pool_layer(2,2,2);//2
    meshCNN.add_pool_layer(2,2,2);//3
    meshCNN.add_pool_layer(2,2,2);//4
    meshCNN.add_pool_layer(2,2,2);//5
    meshCNN.add_pool_layer(2,2,2);//6
    meshCNN.add_pool_layer(2,2,2);//7
    meshCNN.add_pool_layer(2,2,2);//8
    

}

void set_943k_mesh_layers(MeshCNN &meshCNN)
{

    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(3,2,3);//1
    meshCNN.add_pool_layer(3,2,3);//2
    meshCNN.add_pool_layer(3,2,3);//3
    meshCNN.add_pool_layer(2,2,2);//4
    meshCNN.add_pool_layer(2,2,2);//5
    meshCNN.add_pool_layer(2,2,2);//6
    meshCNN.add_pool_layer(2,2,2);//7
    meshCNN.add_pool_layer(2,2,2);//8
    meshCNN.add_pool_layer(2,2,2);//9
    

}


void set_150k_mesh_layers_option1(MeshCNN &meshCNN)
{
    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(2,2,2);//1
    meshCNN.add_pool_layer(2,2,2);//2
    meshCNN.add_pool_layer(2,2,2);//3
    meshCNN.add_pool_layer(2,2,2);//4
    meshCNN.add_pool_layer(2,2,2);//5
    meshCNN.add_pool_layer(2,2,2);//6
    meshCNN.add_pool_layer(2,2,2);//7
    meshCNN.add_pool_layer(2,2,2);//8
    meshCNN.add_pool_layer(2,2,2);//9
}


void set_150k_mesh_layers_option2(MeshCNN &meshCNN)
{
    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(3,2,3);//1
    meshCNN.add_pool_layer(2,2,2);//2
    meshCNN.add_pool_layer(2,2,2);//3
    meshCNN.add_pool_layer(2,2,2);//4
    meshCNN.add_pool_layer(2,2,2);//5
    meshCNN.add_pool_layer(2,2,2);//6
    meshCNN.add_pool_layer(2,2,2);//7
    meshCNN.add_pool_layer(2,2,2);//8
}


void set_150k_mesh_layers_option3(MeshCNN &meshCNN)
{
    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(2,2,2);//1
    meshCNN.add_pool_layer(2,2,2);//2
    meshCNN.add_pool_layer(2,2,2);//3
    meshCNN.add_pool_layer(2,2,2);//4
    meshCNN.add_pool_layer(2,2,2);//5
    meshCNN.add_pool_layer(2,2,2);//6
    meshCNN.add_pool_layer(2,2,2);//7
    meshCNN.add_pool_layer(2,2,2);//8
    //meshCNN.add_pool_layer(2,2,2);//9
}


void set_150k_mesh_layers_option4(MeshCNN &meshCNN)
{
    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(2,2,2);//1
    meshCNN.add_pool_layer(1,1,1);//2
    meshCNN.add_pool_layer(2,2,2);//3
    meshCNN.add_pool_layer(1,1,1);//4
    meshCNN.add_pool_layer(2,2,2);//5
    meshCNN.add_pool_layer(1,1,1);//6
    meshCNN.add_pool_layer(2,2,2);//7
    meshCNN.add_pool_layer(1,1,1);//8
    meshCNN.add_pool_layer(2,2,2);//9
    meshCNN.add_pool_layer(1,1,1);//10
    meshCNN.add_pool_layer(2,2,2);//11
    meshCNN.add_pool_layer(1,1,1);//12
    meshCNN.add_pool_layer(2,2,2);//13
    meshCNN.add_pool_layer(1,1,1);//14
    meshCNN.add_pool_layer(2,2,2);//15
    meshCNN.add_pool_layer(1,1,1);//16
    //meshCNN.add_pool_layer(2,2,2);//9
}


void set_150k_mesh_layers_option5(MeshCNN &meshCNN)
{
    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(2,2,2);//1
    meshCNN.add_pool_layer(2,2,2);//2
    meshCNN.add_pool_layer(2,2,2);//3
    meshCNN.add_pool_layer(2,2,2);//4
    meshCNN.add_pool_layer(2,2,2);//5    
    //meshCNN.add_pool_layer(2,2,2);//9
}

void set_150k_mesh_layers_option6(MeshCNN &meshCNN)
{
    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(2,1,2);//1
    meshCNN.add_pool_layer(1,1,1);//2
    meshCNN.add_pool_layer(2,1,2);//3
    meshCNN.add_pool_layer(1,1,1);//4
    meshCNN.add_pool_layer(2,1,2);//5
    meshCNN.add_pool_layer(1,1,1);//6
    meshCNN.add_pool_layer(2,1,2);//7
    meshCNN.add_pool_layer(1,1,1);//8
}


void set_150k_mesh_layers_option7(MeshCNN &meshCNN)
{
    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(2,2,2);//1
    meshCNN.add_pool_layer(1,1,1);//2
    meshCNN.add_pool_layer(2,2,2);//3
    meshCNN.add_pool_layer(1,1,1);//4
    meshCNN.add_pool_layer(2,2,2);//5
    meshCNN.add_pool_layer(1,1,1);//6
    meshCNN.add_pool_layer(2,2,2);//7
    meshCNN.add_pool_layer(1,1,1);//8
    meshCNN.add_pool_layer(2,2,2);//9
    meshCNN.add_pool_layer(1,1,1);//10
    meshCNN.add_pool_layer(2,2,2);//11
    meshCNN.add_pool_layer(1,1,1);//12
}


void set_150k_mesh_layers_option8(MeshCNN &meshCNN)
{
    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(2,2,2);//1
    meshCNN.add_pool_layer(1,1,1);//2
    meshCNN.add_pool_layer(2,2,2);//3
    meshCNN.add_pool_layer(1,1,1);//4
    meshCNN.add_pool_layer(2,2,2);//5
    meshCNN.add_pool_layer(1,1,1);//6
    meshCNN.add_pool_layer(2,2,2);//7
    meshCNN.add_pool_layer(1,1,1);//8
    meshCNN.add_pool_layer(2,2,2);//9
    meshCNN.add_pool_layer(1,1,1);//10
    meshCNN.add_pool_layer(2,2,2);//11
    meshCNN.add_pool_layer(1,1,1);//12
}


void set_150k_mesh_layers_option9(MeshCNN &meshCNN)
{
    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(2,1,2);//1
    meshCNN.add_pool_layer(1,1,1);//2
    meshCNN.add_pool_layer(2,1,2);//3
    meshCNN.add_pool_layer(1,1,1);//4
    meshCNN.add_pool_layer(2,1,2);//5
    meshCNN.add_pool_layer(1,1,1);//6
    meshCNN.add_pool_layer(2,1,2);//7
    meshCNN.add_pool_layer(1,1,1);//8
}


void set_150k_mesh_layers_option10(MeshCNN &meshCNN)
{
    meshCNN.add_pool_layer(1,1,1);//0
    meshCNN.add_pool_layer(2,1,2);//1
    meshCNN.add_pool_layer(1,1,1);//2
    meshCNN.add_pool_layer(2,1,2);//3
    meshCNN.add_pool_layer(1,1,1);//4
    meshCNN.add_pool_layer(2,1,2);//5
    meshCNN.add_pool_layer(1,1,1);//6
    meshCNN.add_pool_layer(2,1,2);//7
    meshCNN.add_pool_layer(1,1,1);//8
    meshCNN.add_pool_layer(2,1,2);//9
    meshCNN.add_pool_layer(1,1,1);//10
    meshCNN.add_pool_layer(2,1,2);//11
    meshCNN.add_pool_layer(1,1,1);//12
}



void set_1_mesh_layers(MeshCNN &meshCNN)
{
    meshCNN.add_pool_layer(1,1,1);//0
}



int main(int argc, char **argv) {

    std::cout << "EXE mesh_in_obj outputPath" << std::endl;
    std::cout << "Input Obj Mesh need to have verts, faces and color!" << std::endl;
    std::cout << "The color (in full GREEN) indicates which point you would like to keep in the latent code!" << std::endl;

    if (argc < 3){
        std::cout<<"Parameter Too Few!"<<std::endl;
        return -1;
        }
    
    Mesh mesh;

    std::string meshname = argv[1];
    std::string basepath = argv[2];
    mkdir(basepath.c_str(), 0755);
   
   
    mesh.loadmesh_obj(meshname);


    MeshCNN meshCNN=MeshCNN(mesh);

    //TIPS, you should always have layer 0 being stride=1, pool_radius=1, unpool_radius=1 which is the laplace matrix because it will be used in training for computing the laplace loss
    //you need to modify the following function to design the right network structure for your mesh resolution or application    
    set_150k_mesh_layers_option8(meshCNN);

    
    string folder= basepath + "/connections";
    mkdir(folder.c_str(), 0755);
       
    std::string outfolderpath = folder+"/body/";
    mkdir(outfolderpath.c_str(), 0755);    
    
    meshCNN.save_pool_and_unpool_neighbor_info_to_npz(outfolderpath);

    MeshPooler_Visualizer mpv;    
    mpv.save_obj_with_colored_sample_points_all_layers(outfolderpath+"center_", mesh,meshCNN);


    return 0;

}
