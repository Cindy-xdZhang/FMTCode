// Snapshot adapter for the Objective, direct-window branch of GGT17.
// The generated header contains the original computeWithOutFilter body verbatim.
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <execution>
#include <numeric>
#include <string>
#include <stdexcept>
#include <vector>
#include <omp.h>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
using V3 = Eigen::Vector3f;
using M3 = Eigen::Matrix3f;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
static M3 skewMat(const V3& x) {
    M3 a; a << 0,-x.z(),x.y(),x.z(),0,-x.x(),-x.y(),x.x(),0; return a;
}
static M3 makeEigenMatrix3f(float a,float b,float c,float d,float e,float f,float g,float h,float i) {
    M3 r; r << a,b,c,d,e,f,g,h,i; return r;
}
static int factorial(int n) {return n <= 1 ? 1 : n*factorial(n-1);}
static void ComputeDisplacementSystemMatrixCoefficients(int,int,int,const V3&,const V3&,const M3&,M3&,M3&) {
    throw std::runtime_error("Only the user-requested Objective branch is supported.");
}
static auto policy = std::execution::seq;
#define LOG_D(...) ((void)0)
#define STARTTIMER_STRING(...) ((void)0)
#define STOPTIMER_STRING(...) ((void)0)
enum class EInvariance {Objective,Displacement,Similarity,Affine};
struct FieldInfo {int nx,ny,nz,numberOfTimeSteps; Eigen::Vector3d low,high; double t0,t1;};
template<class T> class Discrete3DFlowField {
public:
    FieldInfo info;
    Eigen::Matrix<float,Eigen::Dynamic,3> data;
    Discrete3DFlowField(const FieldInfo& f):info(f),data(f.nx*f.ny*f.nz*f.numberOfTimeSteps,3) {data.setZero();}
    bool HasDiscreteData() const {return true;}
    void ReSampleAnlyticalToDiscreteData() {}
    std::string getPrintOutGridInfo() const {return "headless grid-node adapter";}
    FieldInfo GetFieldInfo() const {return info;}
    double GetMinTime() const {return info.t0;}
    double GetMaxTime() const {return info.t1;}
    int GetNumberOfTimeSteps() const {return info.numberOfTimeSteps;}
    Eigen::Vector3i GetSpatialGridSize() const {return {info.nx,info.ny,info.nz};}
    Eigen::Vector3d GetSpatialMin() const {return info.low;}
    Eigen::Vector3d GetSpatialMax() const {return info.high;}
    auto& GetData() {return data;}
    const auto& GetDataView() const {return data;}
    V3 node(int x,int y,int z,int t) const {return data.row(((t*info.nz+z)*info.ny+y)*info.nx+x);}
    V3 spatial(const V3& p,int t) const {
        int dims[3]={info.nx,info.ny,info.nz},low[3],high[3];float w[3][2];
        for(int a=0;a<3;++a) {
            float c=(p[a]-info.low[a])*((dims[a]-1)/(info.high[a]-info.low[a]));
            low[a]=std::clamp(int(std::floor(c)),0,dims[a]-1);high[a]=std::clamp(low[a]+1,0,dims[a]-1);
            float alpha=c-low[a];w[a][0]=1-alpha;w[a][1]=alpha;
        }
        return (w[0][0]*w[1][0]*w[2][0])*node(low[0],low[1],low[2],t)
             + (w[0][1]*w[1][0]*w[2][0])*node(high[0],low[1],low[2],t)
             + (w[0][0]*w[1][1]*w[2][0])*node(low[0],high[1],low[2],t)
             + (w[0][1]*w[1][1]*w[2][0])*node(high[0],high[1],low[2],t)
             + (w[0][0]*w[1][0]*w[2][1])*node(low[0],low[1],high[2],t)
             + (w[0][1]*w[1][0]*w[2][1])*node(high[0],low[1],high[2],t)
             + (w[0][0]*w[1][1]*w[2][1])*node(low[0],high[1],high[2],t)
             + (w[0][1]*w[1][1]*w[2][1])*node(high[0],high[1],high[2],t);
    }
    V3 GetVector(const V3& p,float time) const {
        double c=(double(time)-info.t0)*((info.numberOfTimeSteps-1)/(info.t1-info.t0));
        int a=std::clamp(int(std::floor(c)),0,info.numberOfTimeSteps-1);
        int b=std::clamp(int(std::ceil(c)),0,info.numberOfTimeSteps-1);
        V3 result=spatial(p,a);
        if(a!=b) {float f=c-float(a);result=result*(1.0-f)+spatial(p,b)*f;}
        return result;
    }
    M3 getVelocityGradientTensor(int x,int y,int z,int t) const {
        const auto dims=GetSpatialGridSize(); int q[3]={x,y,z}; M3 J;
        for(int a=0;a<3;++a) {
            int lo[3]={x,y,z},hi[3]={x,y,z}; lo[a]=std::max(0,q[a]-1);hi[a]=std::min(dims[a]-1,q[a]+1);
            float inverse=float((dims[a]-1)/(info.high[a]-info.low[a]));
            J.col(a)=((node(hi[0],hi[1],hi[2],t)-node(lo[0],lo[1],lo[2],t))*inverse)/float(hi[a]-lo[a]);
        } return J;
    }
    V3 getPartialDerivativeT(int x,int y,int z,int t) const {
        int lo=std::max(t-1,0),hi=std::min(t+1,info.numberOfTimeSteps-1);
        return ((node(x,y,z,hi)-node(x,y,z,lo))*float((info.numberOfTimeSteps-1)/(info.t1-info.t0)))*float(1.0/(hi-lo));
    }
};
template<class T> struct DiscreteScalarField3D {
    FieldInfo mFieldInfo;
    DiscreteScalarField3D(FieldInfo f):mFieldInfo(f) {}
    int GetNumberOfTimeSteps() const {return mFieldInfo.numberOfTimeSteps;}
    void SetValue(int,int,int,int,double) {}
};
static Eigen::Vector2i ClampTimstep2ValidRange(const Discrete3DFlowField<float>& f,int a,int b) {
    return {std::max(0,a),b<0?f.GetNumberOfTimeSteps()-1:std::min(b,f.GetNumberOfTimeSteps()-1)};
}
class GenericLocalOptimization3d {
public:
    EInvariance mInvariance=EInvariance::Objective;
    int NeighborhoodU=9; bool UseSummedAreaTables=false;
    void computeWithOutFilter(Discrete3DFlowField<float>&,int,int,Discrete3DFlowField<float>&,
                              Discrete3DFlowField<float>&,DiscreteScalarField3D<float>&);
};
#include "original_compute.inc"


API int task6_cpp_reference(const float* series,int nx,int ny,int nz,
                           const double* lower,const double* upper,double t0,double dt,
                           int radius,int sat,int threads,float* output) {
    omp_set_num_threads(threads);
    FieldInfo info{nx,ny,nz,3,Eigen::Vector3d(lower),Eigen::Vector3d(upper),t0,t0+2*dt};
    Discrete3DFlowField<float> field(info),u(info),relative(info);
    const int count=nx*ny*nz*3;
    for(int i=0;i<count;++i)for(int a=0;a<3;++a)field.data(i,a)=series[3*i+a];
    DiscreteScalarField3D<float> diagnostic(info);diagnostic.mFieldInfo.numberOfTimeSteps=-1;
    GenericLocalOptimization3d optimizer;optimizer.NeighborhoodU=radius;optimizer.UseSummedAreaTables=bool(sat);
    optimizer.computeWithOutFilter(field,1,1,u,relative,diagnostic);
    for(int i=0;i<nx*ny*nz;++i)for(int a=0;a<3;++a)output[3*i+a]=relative.data(nx*ny*nz+i,a);
    return 0;
}
