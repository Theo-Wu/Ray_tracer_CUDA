#ifndef MATERIAL
#define MATERIAL

struct hit_record;

#include "ray.h"
#include "hitable.h"

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state){
    vec3 p;
    do{
        p = 2.0f*RANDVEC3 - vec3(1,1,1);
    }while (p.squared_length()>=1.0f);
    return p;
}

__device__ float schlick(float cos, float IOR){
    float r0 = (1.0f-IOR)/(1.0f+IOR);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f-cos),5.0f);
}

__device__ vec3 reflect(const vec3& r_in, const vec3& normal){
    return r_in - 2.0f*dot(r_in,normal)*normal;
}

__device__ bool refract(const vec3& r_in, const vec3& normal, float IOR, vec3& refract_dir){
    vec3 uv = unit_vector(r_in);
    float dt = dot(uv,normal);
    float discriminant = 1.0f - IOR*IOR*(1-dt*dt);
    if(discriminant > 0){
        refract_dir = IOR*(uv - normal*dt) - normal*sqrt(discriminant);
        return true;
    }else{
        return false;
    }
}

class material{
    public:
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& throughput, ray& scatter_ray, curandState *local_rand_state) const = 0;
};

class diffuse: public material{
    public:
        __device__ diffuse(const vec3& a):albedo(a){}
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& throughput, ray& scatter_ray, curandState *local_rand_state) const {
            vec3 target = rec.normal + random_in_unit_sphere(local_rand_state);
            scatter_ray = ray(rec.p,target);
            throughput = albedo;
            return true;
        }
        vec3 albedo;
};

class metal: public material{
    public:
        __device__ metal(const vec3& a, float f):albedo(a){if(f<1) fuzz = f;else fuzz = 1;}
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& throughput, ray& scatter_ray, curandState *local_rand_state) const {
            vec3 reflect_dir = reflect(r_in.direction(), rec.normal);
            scatter_ray = ray(rec.p,reflect_dir+fuzz*random_in_unit_sphere(local_rand_state));
            throughput = albedo;
            return (dot(scatter_ray.direction(), rec.normal)>0.0f);
        }
        vec3 albedo;
        float fuzz;
};

class dielectric: public material{
    public:
        __device__ dielectric(float ior): IOR(ior){}
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& throughput, ray& scatter_ray, curandState *local_rand_state) const {
            vec3 outward_normal;
            vec3 reflect_dir = reflect(r_in.direction(),rec.normal);
            float ri;
            throughput = vec3(1.0f,1.0f,1.0f);
            vec3 refract_dir;
            float reflect_prob;
            float cos;
            if(dot(r_in.direction(),rec.normal)>0.0f){
                outward_normal = -rec.normal;
                ri = IOR;
                cos = dot(r_in.direction(),rec.normal) / r_in.direction().length();
                cos = sqrt(1.0f - IOR*IOR*(1-cos*cos));
            }
            else{
                outward_normal = rec.normal;
                ri = 1.0f/IOR;
                cos = -dot(r_in.direction(),rec.normal) / r_in.direction().length();
            }
            
            if(refract(r_in.direction(),outward_normal,ri,refract_dir)){
                reflect_prob = schlick(cos,ri);
            }else{
                reflect_prob = 1.0f;
            }
            if(curand_uniform(local_rand_state) < reflect_prob){
                scatter_ray = ray(rec.p,reflect_dir);
            }else{
                scatter_ray = ray(rec.p,refract_dir);
            }
            return true;
        }
        float IOR;
};

#endif