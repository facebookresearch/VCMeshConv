//
// Created by zhouyi on 6/12/19.
//

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


#include "mdVector.h"
#include "mdSArray.h"

using namespace std;

using namespace PointSampling;



class Mesh
{
public:
    vector<Vec3<float> > points;
    vector<Vec3<float> > colors;
    vector<Vec3<int> > triangles;

    bool loadmesh_obj(const string &fileName) {
        //bool succeed = LoadOBJ(fileName, points, triangles);
        cout<<"Load mesh "<<fileName<<"\n";
        bool succeed = LoadOBJ_withcolor(fileName, points, colors,triangles);
        if (succeed == false) {
            cout << "Failed to load obj.\n";
            return 0;
        }

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

    bool LoadOBJ(const string & fileName,
                 vector< Vec3<float> > & points,
                 vector< Vec3<int> > & triangles)
    {
        const char ObjDelimiters[]=" /";
        const unsigned int BufferSize = 1024;
        FILE * fid = fopen(fileName.c_str(), "r");

        if (fid)
        {
            char buffer[BufferSize];
            Vec3<int> ip;
            Vec3<int> in;
            Vec3<int> it;
            char * pch;
            char * str;
            size_t nn = 0;
            size_t nt = 0;
            Vec3<float> x;
            while (!feof(fid))
            {
                if (!fgets(buffer, BufferSize, fid))
                {
                    break;
                }
                else if (buffer[0] == 'v')
                {
                    if (buffer[1] == ' ')
                    {
                        str = buffer+2;
                        for(int k = 0; k < 3; ++k)
                        {
                            pch = strtok (str, " ");
                            if (pch) x[k] = static_cast<float>(atof(pch));
                            else
                            {
                                return false;
                            }
                            str = NULL;
                        }
                        points.push_back(x);
                    }
                    else if (buffer[1] == 'n')
                    {
                        ++nn;
                    }
                    else if (buffer[1] == 't')
                    {
                        ++nt;
                    }
                }
                else if (buffer[0] == 'f')
                {

                    str = buffer+2;
                    for(int k = 0; k < 3; ++k)
                    {
                        pch = strtok (str, ObjDelimiters);
                        if (pch) ip[k] = atoi(pch) - 1;
                        else
                        {
                            return false;
                        }
                        str = NULL;
                        if (nt > 0)
                        {
                            pch = strtok (NULL, ObjDelimiters);
                            if (pch)  it[k] = atoi(pch) - 1;
                            else
                            {
                                return false;
                            }
                        }
                        if (nn > 0)
                        {
                            pch = strtok (NULL, ObjDelimiters);
                            if (pch)  in[k] = atoi(pch) - 1;
                            else
                            {
                                return false;
                            }
                        }
                    }
                    triangles.push_back(ip);
                }
            }
            fclose(fid);
        }
        else
        {
            cout << "File not found" << endl;
            return false;
        }
        return true;
    }

    bool LoadOBJ_withcolor(const string & fileName,
                 vector< Vec3<float> > & points, vector< Vec3<float> > & colors,
                 vector< Vec3<int> > & triangles)
    {
        const char ObjDelimiters[]=" /";
        const unsigned int BufferSize = 1024;
        FILE * fid = fopen(fileName.c_str(), "r");

        if (fid)
        {
            char buffer[BufferSize];
            Vec3<int> ip;
            Vec3<int> in;
            Vec3<int> it;
            char * pch;
            char * str;
            size_t nn = 0;
            size_t nt = 0;
            Vec3<float> x;
            Vec3<float> c;
            while (!feof(fid))
            {
                if (!fgets(buffer, BufferSize, fid))
                {
                    break;
                }
                else if (buffer[0] == 'v')
                {
                    if (buffer[1] == ' ')
                    {
                        str = buffer+2;
                        for(int k = 0; k < 6; ++k)
                        {
                            pch = strtok (str, " ");
                            if (pch) 
                            {
                                if(k<3)
                                {
                                    x[k] = static_cast<float>(atof(pch));
                                }
                                else
                                {
                                    c[k-3] = static_cast<float>(atof(pch));
                                }
                                
                            }
                            else
                            {
                                return false;
                            }
                            str = NULL;
                        }

                        points.push_back(x);
                        colors.push_back(c);
                        
                    }
                    else if (buffer[1] == 'n')
                    {
                        ++nn;
                    }
                    else if (buffer[1] == 't')
                    {
                        ++nt;
                    }
                }
                else if (buffer[0] == 'f')
                {

                    str = buffer+2;
                    for(int k = 0; k < 3; ++k)
                    {
                        pch = strtok (str, ObjDelimiters);
                        if (pch) ip[k] = atoi(pch) - 1;
                        else
                        {
                            return false;
                        }
                        str = NULL;
                        if (nt > 0)
                        {
                            pch = strtok (NULL, ObjDelimiters);
                            if (pch)  it[k] = atoi(pch) - 1;
                            else
                            {
                                return false;
                            }
                        }
                        if (nn > 0)
                        {
                            pch = strtok (NULL, ObjDelimiters);
                            if (pch)  in[k] = atoi(pch) - 1;
                            else
                            {
                                return false;
                            }
                        }
                    }
                    triangles.push_back(ip);
                }
            }
            fclose(fid);
        }
        else
        {
            cout << "File not found" << endl;
            return false;
        }
        return true;
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