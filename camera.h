#ifndef CAMERA
#define CAMERA

#include "ray.h"

class camera {
    public:
        __device__ camera(){
            int width = 4;
            int height = 2;
            lower_left_corner = vec3(-width/2,-height/2,-1);
            horizental = vec3(width,0,0);
            vertical = vec3(0,height,0);
            origin = vec3(0,1.0,1.0);
        }
        __device__ ray get_ray(float u, float v){
            return ray(origin,lower_left_corner+u*horizental+v*vertical-origin);
        }
        vec3 origin;
        vec3 lower_left_corner;
        vec3 horizental;
        vec3 vertical;
};

#endif