#include <math.h>
#define EPS2 0.000001

__global__ void update(float4 *pos, float3 *vel, float4 *pos_, float3 *vel_, int n, float timedelta)
{
    float3 acc;
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    for (int sub_id = 0; sub_id < n; sub_id ++)
    {
        float3 r;

        r.x = pos_[sub_id].x - pos_[id].x;
        r.y = pos_[sub_id].y - pos_[id].y;
        r.z = pos_[sub_id].z - pos_[id].z;

        float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
        float distSixth = distSqr * distSqr * distSqr;
        float invDistCube = 1.0f/sqrtf(distSixth);

        float s = pos_[id].w * invDistCube;

        acc.x += r.x * s;
        acc.y += r.y * s;
        acc.z += r.z * s;
        
    }

    vel[id].x = vel_[id].x + timedelta * acc.x;
    vel[id].y = vel_[id].y + timedelta * acc.y;
    vel[id].z = vel_[id].z + timedelta * acc.z;

    pos[id].x = pos_[id].x + timedelta * vel[id].x;
    pos[id].y = pos_[id].y + timedelta * vel[id].y;
    pos[id].z = pos_[id].z + timedelta * vel[id].z;
    pos[id].w = pos_[id].w;
}