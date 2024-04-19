#define _CRT_SECURE_NO_WARNINGS 1
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

static inline double sqr(double x) { return x * x; }

#define M_PI 3.14159
#include <cmath>
#include <string>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <list>

#include <random>
static std::default_random_engine engine(10);
static std::uniform_real_distribution<double>uniform(0, 1);


class Vector {
public:
    explicit Vector(double x = 0., double y = 0., double z = 0.) {
        coord[0] = x;
        coord[1] = y;
        coord[2] = z;
    }
    double& operator[](int i) { return coord[i]; }
    double operator[](int i) const { return coord[i]; }

    Vector& operator+=(const Vector& v) {
        coord[0] += v[0];
        coord[1] += v[1];
        coord[2] += v[2];
        return *this;
    }

    double norm2() const {
        return sqr(coord[0]) + sqr(coord[1]) + sqr(coord[2]);
    }

    Vector normalize()  {
        Vector v(coord[0], coord[1], coord[2]);
        double norm = sqrt(v.norm2());
        coord[0] /= norm;
        coord[1] /= norm;
        coord[2] /= norm;
        return *this;
    }

    double coord[3];
};

Vector operator+(const Vector& a, const Vector& b) {
    return Vector(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
Vector operator-(const Vector& a, const Vector& b) {
    return Vector(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
Vector operator*(const Vector& a, double b) {
    return Vector(a[0] * b, a[1] * b, a[2] * b);
}
Vector operator*(double a, const Vector& b) {
    return Vector(a * b[0], a * b[1], a * b[2]);
}
Vector operator*(const Vector& a, const Vector& b) {
    return Vector(a[0] * b[0], a[1] * b[0], a[2] * b[2]);
}
Vector operator/(const Vector& a, double b) {
    return Vector(a[0] / b, a[1] / b, a[2] / b);
}
Vector cross(const Vector& a, const Vector& b) {
    return Vector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

double dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}


Vector random_cos(const Vector& N) {
    double r1 = uniform(engine);
    double r2 = uniform(engine);
    double x = cos(2 * M_PI * r1) * sqrt(1 - r2);
    double y = sin(2 * M_PI * r1) * sqrt(1 - r2);
    double z = sqrt(r2);
    Vector T1;
    if (std::abs(N[0]) < std::abs(N[1])) {
        if (std::abs(N[0]) < std::abs(N[2])) T1 = Vector(0, -N[2], N[1]);
        else T1 = Vector(-N[1], N[0], 0);
    }
    else {
        if (std::abs(N[1]) < std::abs(N[2])) T1 = Vector(-N[2], 0, N[0]);
        else T1 = Vector(-N[1], N[0], 0);
    }
    T1.normalize();
    Vector T2 = cross(N, T1);

    return (z * N + x * T1 + y * T2);
}

class Ray {
public:
    Vector O, u;
    Ray(const Vector& O0, const Vector& u0)
    {
        O = O0;
        u = u0;
    }
};

class Geometry {
public:
    Vector alb;
    bool mir;
    bool transp;
    bool inv;

    Geometry(const Vector& albedo = Vector(1.0, 1.0, 1.0), bool mirroir = false, bool transparent = false, bool inverse = false) {
        alb = albedo;
        mir = mirroir;
        transp = transparent;
        inv = inverse;
    };

    virtual bool intersect(const Ray& r, Vector& P, Vector& N, double& t, Vector& text_alb) const = 0;
};

class Sphere : public Geometry{
public:

    Vector C;
    double R;


    Sphere(const Vector& centre, float rayon, const Vector& albedo, bool mirroir = false, bool transparent = false, bool inverse = false) : C(centre), R(rayon), Geometry(albedo, mirroir, transparent, inverse){}


    bool intersect(const Ray& r, Vector& P, Vector& N, double& t, Vector& text_alb) const {
        double a = 1;
        double b = 2 * dot(r.u, r.O - C);
        double c = (r.O - C).norm2() - R * R;
        double delta = b * b - 4 * a * c;
        if (delta < 0) {
            return false;
        }
        double sqrtdelta = sqrt(delta);
        double t1 = (-b - sqrtdelta) / (2 * a);
        double t2 = (-b + sqrtdelta) / (2 * a);
        if (t2 < 0) {
            return false;
        }
        t = t1;
        if (t1 < 0) t = t2;
        P = r.O + t * r.u;
        N = (P - C);
        N.normalize();
        text_alb = alb;
        return true;
    }
};

class BBox {
public:
    BBox() {}

    BBox(const Vector& min, const Vector& max) : m(min), M(max) {}

    bool intersect(const Ray& r) const {
        Vector invU(1 / r.u[0], 1 / r.u[1], 1 / r.u[2]);
        bool has_intersect = false;
        double tmx = (m[0] - r.O[0]) * invU[0];
        double tMx = (M[0] - r.O[0]) * invU[0];
        double t1x = std::min(tmx, tMx);
        double t2x = std::max(tmx, tMx);
        double tmy = (m[1] - r.O[1]) * invU[1];
        double tMy = (M[1] - r.O[1]) * invU[1];
        double t1y = std::min(tmy, tMy);
        double t2y = std::max(tmy, tMy);
        double tmz = (m[2] - r.O[2]) * invU[2];
        double tMz = (M[2] - r.O[2]) * invU[2];
        double t1z = std::min(tmz, tMz);
        double t2z = std::max(tmz, tMz);

        double tentry = std::max(t1x, std::max(t1y, t1z));
        double texit = std::min(t2x, std::min(t2y, t2z));

        if (texit < 0) return false;
        if (tentry < texit) has_intersect = true;
   
        return has_intersect;

    }

    Vector m, M;
};

class BVH{
public:
    BVH(const BBox& b, int start=-1, int end=-1) : bbox(b), idx_start(start), idx_end(end) {
        left = NULL;
        right = NULL;
    }
    BVH* left;
    BVH* right;
    BBox bbox;
    int idx_start, idx_end;
};

class TriangleIndices {
public:
    TriangleIndices(int vtxi = -1, int vtxj = -1, int vtxk = -1, int ni = -1, int nj = -1, int nk = -1, int uvi = -1, int uvj = -1, int uvk = -1, int group = -1, bool added = false) : vtxi(vtxi), vtxj(vtxj), vtxk(vtxk), uvi(uvi), uvj(uvj), uvk(uvk), ni(ni), nj(nj), nk(nk), group(group) {
    };
    int vtxi, vtxj, vtxk; // indices within the vertex coordinates array
    int uvi, uvj, uvk;  // indices within the uv coordinates array
    int ni, nj, nk;  // indices within the normals array
    int group;       // face group
};
  
class TriangleMesh : public Geometry{
public:
     TriangleMesh(const Vector& alb = Vector(1.0, 1.0, 1.0), bool mir = false, bool transp = false, bool inv = false)
         : Geometry(alb, mir, transp, inv) {}
    ~TriangleMesh() {}
    

    void readOBJ(const char* obj) {
 
        char matfile[255];
        char grp[255];
 
        FILE* f;
        f = fopen(obj, "r");
        int curGroup = -1;
        while (!feof(f)) {
            char line[255];
            if (!fgets(line, 255, f)) break;
 
            std::string linetrim(line);
            linetrim.erase(linetrim.find_last_not_of(" \r\t") + 1);
            strcpy(line, linetrim.c_str());
 
            if (line[0] == 'u' && line[1] == 's') {
                sscanf(line, "usemtl %[^\n]\n", grp);
                curGroup++;
            }
 
            if (line[0] == 'v' && line[1] == ' ') {
                Vector vec;
 
                Vector col;
                if (sscanf(line, "v %lf %lf %lf %lf %lf %lf\n", &vec[0], &vec[1], &vec[2], &col[0], &col[1], &col[2]) == 6) {
                    col[0] = std::min(1., std::max(0., col[0]));
                    col[1] = std::min(1., std::max(0., col[1]));
                    col[2] = std::min(1., std::max(0., col[2]));
 
                    vertices.push_back(vec);
                    vertexcolors.push_back(col);
 
                } else {
                    sscanf(line, "v %lf %lf %lf\n", &vec[0], &vec[1], &vec[2]);
                    vertices.push_back(vec);
                }
            }
            if (line[0] == 'v' && line[1] == 'n') {
                Vector vec;
                sscanf(line, "vn %lf %lf %lf\n", &vec[0], &vec[1], &vec[2]);
                normals.push_back(vec);
            }
            if (line[0] == 'v' && line[1] == 't') {
                Vector vec;
                sscanf(line, "vt %lf %lf\n", &vec[0], &vec[1]);
                uvs.push_back(vec);
            }
            if (line[0] == 'f') {
                TriangleIndices t;
                int i0, i1, i2, i3;
                int j0, j1, j2, j3;
                int k0, k1, k2, k3;
                int nn;
                t.group = curGroup;
 
                char* consumedline = line + 1;
                int offset;
 
                nn = sscanf(consumedline, "%u/%u/%u %u/%u/%u %u/%u/%u%n", &i0, &j0, &k0, &i1, &j1, &k1, &i2, &j2, &k2, &offset);
                if (nn == 9) {
                    if (i0 < 0) t.vtxi = vertices.size() + i0; else t.vtxi = i0 - 1;
                    if (i1 < 0) t.vtxj = vertices.size() + i1; else t.vtxj = i1 - 1;
                    if (i2 < 0) t.vtxk = vertices.size() + i2; else t.vtxk = i2 - 1;
                    if (j0 < 0) t.uvi = uvs.size() + j0; else   t.uvi = j0 - 1;
                    if (j1 < 0) t.uvj = uvs.size() + j1; else   t.uvj = j1 - 1;
                    if (j2 < 0) t.uvk = uvs.size() + j2; else   t.uvk = j2 - 1;
                    if (k0 < 0) t.ni = normals.size() + k0; else    t.ni = k0 - 1;
                    if (k1 < 0) t.nj = normals.size() + k1; else    t.nj = k1 - 1;
                    if (k2 < 0) t.nk = normals.size() + k2; else    t.nk = k2 - 1;
                    indices.push_back(t);
                } else {
                    nn = sscanf(consumedline, "%u/%u %u/%u %u/%u%n", &i0, &j0, &i1, &j1, &i2, &j2, &offset);
                    if (nn == 6) {
                        if (i0 < 0) t.vtxi = vertices.size() + i0; else t.vtxi = i0 - 1;
                        if (i1 < 0) t.vtxj = vertices.size() + i1; else t.vtxj = i1 - 1;
                        if (i2 < 0) t.vtxk = vertices.size() + i2; else t.vtxk = i2 - 1;
                        if (j0 < 0) t.uvi = uvs.size() + j0; else   t.uvi = j0 - 1;
                        if (j1 < 0) t.uvj = uvs.size() + j1; else   t.uvj = j1 - 1;
                        if (j2 < 0) t.uvk = uvs.size() + j2; else   t.uvk = j2 - 1;
                        indices.push_back(t);
                    } else {
                        nn = sscanf(consumedline, "%u %u %u%n", &i0, &i1, &i2, &offset);
                        if (nn == 3) {
                            if (i0 < 0) t.vtxi = vertices.size() + i0; else t.vtxi = i0 - 1;
                            if (i1 < 0) t.vtxj = vertices.size() + i1; else t.vtxj = i1 - 1;
                            if (i2 < 0) t.vtxk = vertices.size() + i2; else t.vtxk = i2 - 1;
                            indices.push_back(t);
                        } else {
                            nn = sscanf(consumedline, "%u//%u %u//%u %u//%u%n", &i0, &k0, &i1, &k1, &i2, &k2, &offset);
                            if (i0 < 0) t.vtxi = vertices.size() + i0; else t.vtxi = i0 - 1;
                            if (i1 < 0) t.vtxj = vertices.size() + i1; else t.vtxj = i1 - 1;
                            if (i2 < 0) t.vtxk = vertices.size() + i2; else t.vtxk = i2 - 1;
                            if (k0 < 0) t.ni = normals.size() + k0; else    t.ni = k0 - 1;
                            if (k1 < 0) t.nj = normals.size() + k1; else    t.nj = k1 - 1;
                            if (k2 < 0) t.nk = normals.size() + k2; else    t.nk = k2 - 1;
                            indices.push_back(t);
                        }
                    }
                }
 
                consumedline = consumedline + offset;
 
                while (true) {
                    if (consumedline[0] == '\n') break;
                    if (consumedline[0] == '\0') break;
                    nn = sscanf(consumedline, "%u/%u/%u%n", &i3, &j3, &k3, &offset);
                    TriangleIndices t2;
                    t2.group = curGroup;
                    if (nn == 3) {
                        if (i0 < 0) t2.vtxi = vertices.size() + i0; else    t2.vtxi = i0 - 1;
                        if (i2 < 0) t2.vtxj = vertices.size() + i2; else    t2.vtxj = i2 - 1;
                        if (i3 < 0) t2.vtxk = vertices.size() + i3; else    t2.vtxk = i3 - 1;
                        if (j0 < 0) t2.uvi = uvs.size() + j0; else  t2.uvi = j0 - 1;
                        if (j2 < 0) t2.uvj = uvs.size() + j2; else  t2.uvj = j2 - 1;
                        if (j3 < 0) t2.uvk = uvs.size() + j3; else  t2.uvk = j3 - 1;
                        if (k0 < 0) t2.ni = normals.size() + k0; else   t2.ni = k0 - 1;
                        if (k2 < 0) t2.nj = normals.size() + k2; else   t2.nj = k2 - 1;
                        if (k3 < 0) t2.nk = normals.size() + k3; else   t2.nk = k3 - 1;
                        indices.push_back(t2);
                        consumedline = consumedline + offset;
                        i2 = i3;
                        j2 = j3;
                        k2 = k3;
                    } else {
                        nn = sscanf(consumedline, "%u/%u%n", &i3, &j3, &offset);
                        if (nn == 2) {
                            if (i0 < 0) t2.vtxi = vertices.size() + i0; else    t2.vtxi = i0 - 1;
                            if (i2 < 0) t2.vtxj = vertices.size() + i2; else    t2.vtxj = i2 - 1;
                            if (i3 < 0) t2.vtxk = vertices.size() + i3; else    t2.vtxk = i3 - 1;
                            if (j0 < 0) t2.uvi = uvs.size() + j0; else  t2.uvi = j0 - 1;
                            if (j2 < 0) t2.uvj = uvs.size() + j2; else  t2.uvj = j2 - 1;
                            if (j3 < 0) t2.uvk = uvs.size() + j3; else  t2.uvk = j3 - 1;
                            consumedline = consumedline + offset;
                            i2 = i3;
                            j2 = j3;
                            indices.push_back(t2);
                        } else {
                            nn = sscanf(consumedline, "%u//%u%n", &i3, &k3, &offset);
                            if (nn == 2) {
                                if (i0 < 0) t2.vtxi = vertices.size() + i0; else    t2.vtxi = i0 - 1;
                                if (i2 < 0) t2.vtxj = vertices.size() + i2; else    t2.vtxj = i2 - 1;
                                if (i3 < 0) t2.vtxk = vertices.size() + i3; else    t2.vtxk = i3 - 1;
                                if (k0 < 0) t2.ni = normals.size() + k0; else   t2.ni = k0 - 1;
                                if (k2 < 0) t2.nj = normals.size() + k2; else   t2.nj = k2 - 1;
                                if (k3 < 0) t2.nk = normals.size() + k3; else   t2.nk = k3 - 1;                             
                                consumedline = consumedline + offset;
                                i2 = i3;
                                k2 = k3;
                                indices.push_back(t2);
                            } else {
                                nn = sscanf(consumedline, "%u%n", &i3, &offset);
                                if (nn == 1) {
                                    if (i0 < 0) t2.vtxi = vertices.size() + i0; else    t2.vtxi = i0 - 1;
                                    if (i2 < 0) t2.vtxj = vertices.size() + i2; else    t2.vtxj = i2 - 1;
                                    if (i3 < 0) t2.vtxk = vertices.size() + i3; else    t2.vtxk = i3 - 1;
                                    consumedline = consumedline + offset;
                                    i2 = i3;
                                    indices.push_back(t2);
                                } else {
                                    consumedline = consumedline + 1;
                                }
                            }
                        }
                    }
                }
 
            }
 
        }
        fclose(f);
 
    }
 
    void loadTEXTURE(const char * file) {
		int x,y,c;
		texture.push_back(stbi_loadf(file,&x,&y,&c,3));
		texture_width.push_back(x);
		texture_height.push_back(y);
	}

    //BVH* bvh;
    BVH* bvh = new BVH(setBBox(0, indices.size()), 0, indices.size());
    std::vector<TriangleIndices> indices;
    std::vector<Vector> vertices;
    std::vector<Vector> normals;
    std::vector<Vector> uvs;
    std::vector<Vector> vertexcolors;
    std::vector<float*> texture;
	std::vector<int> texture_width, texture_height;

   // Code for BVH
   bool intersect(const Ray& r, Vector& P, Vector& N, double& t, Vector& text_alb) const {

       if (!bvh->bbox.intersect(r)) return false;

       bool has_inter = false;
       double best_alpha, best_beta;
       int best_i;
       std::list<BVH*> stack;
       stack.push_back(bvh);

       while (!stack.empty()) {
           const BVH* node = stack.back();
           stack.pop_back();
           if (node->left) {
               if (node->left->bbox.intersect(r)) {
                   stack.push_back(node->left);
               }
               if (node->right->bbox.intersect(r)) {
                   stack.push_back(node->right);
               }
           }
           else {
               for (int i = node->idx_start; i < node->idx_end; i++) {
                   Vector A = vertices[indices[i].vtxi];
                   Vector B = vertices[indices[i].vtxj];
                   Vector C = vertices[indices[i].vtxk];
                   Vector e1 = B - A;
                   Vector e2 = C - A;
                   Vector N_local = cross(e1, e2);
                   double invUdotN = 1 / dot(r.u, N_local);
                   Vector OA = A - r.O;
                   Vector OAcrossU = cross(OA, r.u);
                   double beta = dot(e2, OAcrossU) * invUdotN;
                   double gamma = -dot(e1, OAcrossU) * invUdotN;
                   double alpha = 1 - beta - gamma;
                   double localt = dot(OA, N_local) * invUdotN;

                   if (beta >= 0 && beta <= 1 && gamma >= 0 && gamma <= 1 && localt >= 0 && alpha >= 0) { 
                       if (localt < t) {
                           has_inter = true;
                           t = localt;
                           P = r.O + t * r.u;
                           //N = N_local;
                           N = alpha * normals[indices[i].ni] + beta * normals[indices[i].nj] + gamma * normals[indices[i].nk];
                           N.normalize();
                           best_alpha = alpha;
                           best_beta = beta;
                           best_i = i;
                       }
                   }
               }
           }
       }
       if (has_inter) {
           if (indices[best_i].group > texture.size() || texture.size() == 0) {
				text_alb = alb;
			}
           else {
				Vector UV = uvs[indices[best_i].uvi] * best_alpha + uvs[indices[best_i].uvj] * best_beta + uvs[indices[best_i].uvk] * (1 - best_alpha - best_beta);
               UV[0] = fabs(UV[0]);
				UV[0] = UV[0] - floor(UV[0]);
				UV[1] = fabs(UV[1]);
				UV[1] = UV[1] - floor(UV[1]);
				UV[1] = 1 - UV[1];
				UV = UV * Vector(texture_width[indices[best_i].group], texture_height[indices[best_i].group], 0);

				int u = std::min((int)UV[0],texture_width[indices[best_i].group] - 1);
				int v = std::min((int)UV[1],texture_height[indices[best_i].group] - 1);
				int pos = v * texture_width[indices[best_i].group] + u;
               text_alb = Vector(texture[indices[best_i].group][3 * pos], texture[indices[best_i].group][3 * pos + 1], texture[indices[best_i].group][3 * pos + 2]);
			}
       }
       return has_inter;
   }


    //Code pour construire un BVH
    BBox setBBox(int triangle_start, int triangle_end) {
        BBox bbox;
        bbox.m = Vector(1E10, 1E10, 1E10);
        bbox.M = Vector(0, 0, 0);

        for (int i = triangle_start;i < triangle_end;i++) {
            for (int j = 0; j < 3; j++) {
                bbox.m[j] = std::min(bbox.m[j], vertices[indices[i].vtxi][j]);
                bbox.m[j] = std::min(bbox.m[j], vertices[indices[i].vtxj][j]);
                bbox.m[j] = std::min(bbox.m[j], vertices[indices[i].vtxk][j]);
                bbox.M[j] = std::max(bbox.M[j], vertices[indices[i].vtxi][j]);
                bbox.M[j] = std::max(bbox.M[j], vertices[indices[i].vtxj][j]);
                bbox.M[j] = std::max(bbox.M[j], vertices[indices[i].vtxk][j]);
            }
        }
        return bbox;
    }

    void setBVH(BVH* node, int idx_debut, int idx_fin){
        node->idx_start = idx_debut;
        node->idx_end = idx_fin;
        BBox bbox = setBBox(idx_debut, idx_fin);
        node->bbox = bbox;

        Vector size = bbox.M - bbox.m;
        Vector box_middle = (bbox.m + bbox.M) / 2;
        int dimension = 0;
        if (size[1] > size[dimension]) dimension = 1;
        if (size[2] > size[dimension]) dimension = 2;


        int idx_pivot = idx_debut;

        for (int i=idx_debut; i<idx_fin; i++){
            Vector middle_triangle = (vertices[indices[i].vtxi] + vertices[indices[i].vtxj] + vertices[indices[i].vtxk])/3;
            if (middle_triangle[dimension] < box_middle[dimension]){
                std::swap(indices[i], indices[idx_pivot]);
                idx_pivot++;
            }
        }
        if (idx_fin - idx_debut < 5 || idx_pivot - idx_debut < 2 || idx_fin - idx_pivot < 2) {
            return;
        }
        node->left = new BVH(BBox(), idx_debut, idx_pivot);
        node->right = new BVH(BBox(), idx_pivot, idx_fin);
        setBVH(node->left, idx_debut, idx_pivot);
        setBVH(node->right, idx_pivot, idx_fin);
    }

	void translate(const Vector& t) {
		for (int i = 0; i < vertices.size(); i++) {
			vertices[i] = vertices[i] + t;
		}
	}

    void scale(double s) {
		for (int i = 0; i < vertices.size(); i++) {
			vertices[i] = vertices[i] * s;
		}
	}

    void rotate_z(double theta) {//}, const Vector& axis) {
		double s = sin(theta);
		double c = cos(theta);
		for (int i = 0; i < vertices.size(); i++) {
			double x = vertices[i][0];
			double y = vertices[i][1];
			double z = vertices[i][2];
			vertices[i][0] = x * c + z * s;
			vertices[i][2] = -x * s + y * s;
		}
	}

};

class Scene {
public:
    Scene() {};
    std::vector<Geometry*> objects;

    void addgeometry(Geometry* obj){
        objects.push_back(obj);
    }

    bool intersect(Ray& r, int& idx, Vector& P, Vector& N, double& t, Vector& text_alb){
        bool has_intersect = false;
        double tmin;
        t = 1E10;
        Vector P_idx, N_idx, text_alb_idx;
        for (int i = 0; i < objects.size(); i++) {
            if (objects[i]->intersect(r, P_idx, N_idx, tmin, text_alb_idx)) {
                has_intersect = true;
                if (tmin < t) {
                    t = tmin;
                    idx = i;
                    P = P_idx;
                    if (objects[idx]->inv) { N = -1 * N_idx; }
                    else { N = N_idx; }
                    text_alb = text_alb_idx;
                    
                }    
            }
        }
        return has_intersect;
    }

    Vector getColor(Ray r, int bounce, bool was_diffuse_interaction=false) {
        Vector P, N, alb;
        int idx = -1;
        double t;
        Vector L(-10, 20, 40);
        double intensity = 1E10;

        Vector lumC = dynamic_cast<Sphere*>(objects[0])->C;
        double lumR = dynamic_cast<Sphere*>(objects[0])->R;

        if (bounce == 0) return Vector(0, 0, 0);

        if (intersect(r, idx, P, N, t, alb)){
            if (idx == 0) {
                if (was_diffuse_interaction) return Vector(0., 0., 0.);
                else return (intensity / (4 * M_PI * M_PI * lumR * lumR)) * Vector(1, 1, 1);
            }


            if (objects[idx]->mir) {
                Ray reflect(P + 0.1 * N, r.u - 2 * dot(r.u, N) * N);
                return getColor(reflect, bounce - 1);
            }

            else if (objects[idx]->transp) {
                double n1 = 1;
                double n2 = 1.5;
                if (dot(r.u, N) > 0) {
                    std::swap(n1, n2);
                    N = -1 * N;
                }
                Vector tx_tran, tx_refr;
                tx_refr = n1 / n2 * (r.u - dot(r.u, N) * N);
                double rad = 1 - sqr(n1 / n2) * (1 - sqr(dot(r.u, N)));
                if (rad < 0) {
                    Vector R = r.u - 2 * dot(r.u, N) * N;
                    return getColor(Ray(P + 0.1 * N, R), bounce - 1);
                }
                tx_tran = -sqrt(rad) * N;
                Vector T = tx_refr + tx_tran;
                Ray refract = Ray(P - 0.1 * N, T);
                Vector color_1 = getColor(refract, bounce - 1);
                
                // Sans Fresnel
                //return color_1; 

                //// Avec Fresnel
                double k0 = sqr((n1 - n2) / (n1 + n2));
                double R = k0 + (1 - k0) * std::pow(1 - std::abs(dot(r.u, N)), 5);
                double T0 = 1 - R;
                Ray reflect = Ray(P + 0.1 * N, r.u - 2 * dot(r.u, N) * N);
                Vector color_2 = getColor(reflect, bounce - 1);

                Vector color = T0 * color_1 + R * color_2;
                return color;
            }

            else {
                Vector vecLum = lumC - P;
                vecLum.normalize();
                Vector Nprime = random_cos(-1 * vecLum);
                Vector Pprime = Nprime * lumR + lumC;
                Vector wi = Pprime - P;
                double d2 = wi.norm2();
                wi.normalize();

                Ray w_i(P + 0.01 * N, random_cos(N));
                Vector color_indirect = alb * getColor(w_i, bounce - 1, true);


                Vector vecLum_ind = L - P;
                vecLum_ind.normalize();
                int idx_1;
                Vector P_1, N_1, alb_1;
                Ray r_1 = Ray(P + 0.001 * N, wi);

                if (intersect(r_1, idx_1, P_1, N_1, t, alb_1) && (sqr(t + 0.01) < d2))
                    {
                        return color_indirect;
                    }
                else {
                        Vector color_direct;


                        double L_w_i = intensity / (4 * sqr(M_PI * lumR));
                        color_direct = alb * L_w_i * std::max(0., dot(N, wi)) * std::max(0., dot(Nprime, -1 * wi)) * sqr(lumR) / (d2 * std::max(0., dot(Nprime, -1 * vecLum)));

                        return color_direct + color_indirect;

                }
            }
        }
        return Vector(0, 0, 0);
    }
};




int main() {
    int W = 512;
    int H = 512;

    Vector camera(0., 0., 55.);
    double fov = 60 * M_PI / 180;
    double d = W / (2 * tan(fov / 2));
    double gamma = 2.2;

    bool blur = false;
    double lens = 1.;
    double focus_distance = 55;

    int bounce = 7;
    int chemins = 50;

    Scene S;

    // Lumiere
    S.addgeometry(new Sphere(Vector(-10., 20., 40.), 10., Vector(1,1,1)));

    //S.addgeometry(new Sphere(Vector(0., 0., 0.), 10., Vector(0.5, 0.3, 0.8)));
    //S.addgeometry(new Sphere(Vector(-20., 10., 0.), 10., Vector(0.5, 0.3, 0.8), true));
    //S.addgeometry(new Sphere(Vector(0., 0., 0.), 10., Vector(0.5, 0.3, 0.8), false, true));
    S.addgeometry(new Sphere(Vector(20., 10., 0.), 9.9, Vector(0.5, 0.3, 0.8), false, true, true));
    S.addgeometry(new Sphere(Vector(20., 10., 0.), 10., Vector(0.5, 0.3, 0.8), false, true));

    // Scene
    S.addgeometry(new Sphere(Vector(0.0, 0.0, -1000.), 940., Vector(0., 1., 0.)));
    S.addgeometry(new Sphere(Vector(0.0, 1000.0, 0.), 940., Vector(1., 0., 0.)));
    S.addgeometry(new Sphere(Vector(0.0, 0.0, 1000.), 940, Vector(1., 0., 0.5)));
    S.addgeometry(new Sphere(Vector(0.0, -1000.0, 0.), 990, Vector(0., 0., 1.)));
    S.addgeometry(new Sphere(Vector(-1000.0, 0.0, 0.), 940, Vector(1., 1., 0.)));
    S.addgeometry(new Sphere(Vector(1000.0, 0.0, 0.), 940, Vector(0.8, 0., 1.)));

    // Chat
    TriangleMesh m;
    m.readOBJ("cat.obj");
    m.loadTEXTURE("cat_diff.png");
    //m.rotate_z(-3 * M_PI / 4);
    m.translate(Vector(0, -10, 0));
    m.scale(0.6);
    m.setBVH(m.bvh, 0, m.indices.size());
    S.addgeometry(&m);


    std::vector<unsigned char> image(W * H * 3, 0);

    #pragma omp parallel for
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                Vector color(0, 0, 0);
                for (int k = 0; k < chemins; k++) {
                    double r1 = uniform(engine) - 0.5;
                    double r2 = uniform(engine) - 0.5;
                    Vector orientation(j - W / 2 + 0.5 + r1, -i + H / 2 - 0.5 + r2, -d);
                    orientation.normalize();
                    if (blur) {
                        double r1_blur = uniform(engine);
                        double r2_blur = uniform(engine);
                        Vector blur_camera = camera + Vector(cos(2 * M_PI * r1_blur), sin(2 * M_PI * r1_blur), 0) * sqrt(-2 * log(r2_blur)) * lens;
                        orientation = camera + focus_distance * orientation - blur_camera;
                        orientation.normalize();
                        Ray r(blur_camera, orientation);
                        color += S.getColor(r, bounce) / chemins;
                    }
                    else {
                        Ray r(camera, orientation);
                        color += S.getColor(r, bounce) / chemins;
                    }
                    
                }
                image[(i * W + j) * 3 + 0] += std::min(255., std::pow(color[0], 1. / gamma));
                image[(i * W + j) * 3 + 1] += std::min(255., std::pow(color[1], 1. / gamma));
                image[(i * W + j) * 3 + 2] += std::min(255., std::pow(color[2], 1. / gamma));
            }
        }
    stbi_write_png("image.png", W, H, 3, &image[0], 0);


    return 0;
}
