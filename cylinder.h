#ifndef CYLINDER
#define CYLINDER

#include "hitable.h"
#include <algorithm>

class cylinder: public hitable  {
    public:
        __device__ cylinder() {}
        __device__ cylinder(vec3 pos, vec3 dir, float rad, material *mat) : position(pos), direction(dir), radius(rad), material_pointer(mat)  {};
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        vec3 position;
		vec3 direction;
        float radius;
		material *material_pointer;
};

__device__ bool cylinder::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
    vec3 A = r.origin();
	vec3 B = r.direction();
	vec3 n = direction;
	vec3 O = position;
	float a1 = dot(B,B);
	float a2 = dot(B,n)*dot(B,n);
	float b1 = dot(B,A-O);
	float b2 = dot(B,n)*dot(A-O,n);
	float c1 = dot(A-O,A-O);
	float c2 = dot(A-O,n)*dot(A-O,n);
	float a = a1 - a2;
	float b = b1 - b2;
	float c = c1 - c2 - radius*radius;
	float D = b * b - a * c;
    if (D > 0.0)
    {
		float t0 = (-b - sqrt(D)) / a;
		
        if((t0>tmin)&&(t0<tmax)){
			rec.t = t0;
			rec.p = r.point_at_parameter(rec.t); 
			vec3 u = rec.p - position;
			vec3 toHitPosition = u-dot(u,direction)*direction;
			vec3 normal_root = position + dot(u,direction)*direction;
			bool enteringPrimitive = (normal_root-r.origin()).length()<(normal_root-rec.p).length()+0.001? false:true; 
			rec.normal = enteringPrimitive? (toHitPosition)/radius: -(toHitPosition)/radius; 
            return true;
        }
		float t1 = (-b + sqrt(D)) / a;
		
		if((t1>tmin)&&(t1<tmax)){
			rec.t = t1;
			rec.p = r.point_at_parameter(rec.t); 
			vec3 u = rec.p - position;
			vec3 toHitPosition = u-dot(u,direction)*direction;
			vec3 normal_root = position + dot(u,direction)*direction;
			bool enteringPrimitive = (normal_root-r.origin()).length()<(normal_root-rec.p).length()+0.001? false:true; 
			rec.normal = enteringPrimitive? (toHitPosition)/radius: -(toHitPosition)/radius; 
            return true;
        }
	}
    return false;
}


#endif