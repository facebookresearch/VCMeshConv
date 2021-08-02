/*
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/

#ifndef POINTSAMPLING_MESHLOADER_H
#define POINTSAMPLING_MESHLOADER_H



#endif //POINTSAMPLING_MESHLOADER_H



#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>

#include "third-party/mdSArray.h"
#include "third-party/mdVector.h"
#include "third-party/obj_io.h"

using namespace std;

using namespace PointSampling;



class Mesh
{
public:
    vector<Vec3<float> > points;
    vector<Vec3<float> > colors;
    vector<Vec3<int> > triangles;

    bool loadmesh_obj(const string &fileName) {
        cout<<"Load mesh "<<fileName<<"\n";
        using ObjFaceType =
            thinks::ObjTriangleFace<thinks::ObjIndex<std::uint16_t>>;

        auto add_position =
            thinks::MakeObjAddFunc<thinks::ObjPosition<float, 3>>(
                [this](const auto &pos) {
                    this->points.push_back(
                        Vec3<float>{pos.values[0], pos.values[1], pos.values[2]});
                });
        auto add_face =
            thinks::MakeObjAddFunc<ObjFaceType>([this](const auto &face) {
              this->triangles.push_back({
                  face.values[0].value,
                  face.values[1].value,
                  face.values[2].value,
              });
            });

        {
          auto ifs = std::ifstream(fileName);
          const auto result = thinks::ReadObj(ifs, add_position, add_face);
        }

        return 0;
    }

    bool LoadOFF(const string & fileName,
                 vector< Vec3<float> > & points,
                 vector< Vec3<int> > & triangles)
    {
        FILE * fid = fopen(fileName.c_str(), "r");
        if (fid)
        {
            const string strOFF("OFF");
            char temp[1024];
            fscanf(fid, "%s", temp);
            if (string(temp) != strOFF)
            {
                printf( "Loading error: format not recognized \n");
                fclose(fid);
                return false;
            }
            else
            {
                int nv = 0;
                int nf = 0;
                int ne = 0;
                fscanf(fid, "%i", &nv);
                fscanf(fid, "%i", &nf);
                fscanf(fid, "%i", &ne);
                points.resize(nv);
                triangles.resize(nf);
                Vec3<float> coord;
                float x = 0;
                float y = 0;
                float z = 0;
                for (int p = 0; p < nv ; p++)
                {
                    fscanf(fid, "%f", &x);
                    fscanf(fid, "%f", &y);
                    fscanf(fid, "%f", &z);
                    points[p].X() = x;
                    points[p].Y() = y;
                    points[p].Z() = z;
                }
                int i = 0;
                int j = 0;
                int k = 0;
                int s = 0;
                for (int t = 0; t < nf ; ++t) {
                    fscanf(fid, "%i", &s);
                    if (s == 3)
                    {
                        fscanf(fid, "%i", &i);
                        fscanf(fid, "%i", &j);
                        fscanf(fid, "%i", &k);
                        triangles[t].X() = i;
                        triangles[t].Y() = j;
                        triangles[t].Z() = k;
                    }
                    else            // Fix me: support only triangular meshes
                    {
                        for(int h = 0; h < s; ++h) fscanf(fid, "%i", &s);
                    }
                }
                fclose(fid);
            }
        }
        else
        {
            printf( "Loading error: file not found \n");
            return false;
        }    return true;
    }


    bool SaveOBJ(const std::string          & fileName,
                 const vector< Vec3<float> > & points,
                 const vector< Vec3<int> > & triangles)
    {
        std::cout << "Saving " <<  fileName << std::endl;
        std::ofstream fout(fileName.c_str());
        if (fout.is_open())
        {
            const size_t nV = points.size();
            const size_t nT = triangles.size();
            for(size_t v = 0; v < nV; v++)
            {
                fout << "v " << points[v][0] << " "
                     << points[v][1] << " "
                     << points[v][2] << std::endl;
            }
            for(size_t f = 0; f < nT; f++)
            {
                fout <<"f " << triangles[f][0]+1 << " "
                     << triangles[f][1]+1 << " "
                     << triangles[f][2]+1 << std::endl;
            }
            fout.close();
            return true;
        }
        return false;
    }

    bool SaveOBJ(const std::string          & fileName,
                 const vector< Vec3<float> > & points,
                 const vector< Vec3<float> > & colors,
                 const vector< Vec3<int> > & triangles)
    {
        std::cout << "Saving " <<  fileName << std::endl;
        std::ofstream fout(fileName.c_str());
        if (fout.is_open())
        {
            const size_t nV = points.size();
            const size_t nT = triangles.size();
            for(size_t v = 0; v < nV; v++)
            {
                fout << "v " << points[v][0] << " "
                     << points[v][1] << " "
                     << points[v][2] << " "

                     << colors[v][0] << " "
                     << colors[v][1] << " "
                     << colors[v][2] << " "
                     <<std::endl;
            }
            for(size_t f = 0; f < nT; f++)
            {
                fout <<"f " << triangles[f][0]+1 << " "
                     << triangles[f][1]+1 << " "
                     << triangles[f][2]+1 << std::endl;
            }
            fout.close();
            return true;
        }
        return false;
    }


};
