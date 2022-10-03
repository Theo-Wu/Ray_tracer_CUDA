#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <float.h>
#include "common.h"
#include "sphere.h"
#include "cylinder.h"
#include "hitable_list.h"
#include "vec3.h"
#include "camera.h"
#include "material.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line){
    if(result){
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))


__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state){
    ray cur_ray = r;
    vec3 cur_throughput = vec3(1,1,1);
    for(int i = 0; i < 50; i++){
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            // return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f); //normal
            ray scatter_ray;
            vec3 throughput;
            if(rec.material_pointer->scatter(cur_ray,rec,throughput,scatter_ray,local_rand_state)){
                cur_throughput *= throughput;
                cur_ray = scatter_ray;
            }else{
                return vec3(0,0,0);
            }
        }
        else { // sky
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_throughput * c;
        }
    }
    return vec3(0,0,0);
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)){
        return ;
    }
    int pixel_index = j*max_x + i;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)){
        return ;
    }
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s = 0; s < ns; s++){
        float u = float(i+curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j+curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u,v);
        col += color(r,world,&local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list)   = new sphere(vec3(0,0,-1), 0.5, new diffuse(vec3(0.8,0.3,0.3)));
        *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100, new diffuse(vec3(0.8,0.8,0.0)));
        *(d_list+2) = new sphere(vec3(1,0,-1), 0.5, new metal(vec3(0.8,0.6,0.2),1.0));
        *(d_list+3) = new sphere(vec3(-1,0,-1),0.5, new dielectric(1.5));
        *(d_list+4) = new sphere(vec3(-1,0,-1),-0.45, new dielectric(1.5));
        *(d_list+5) = new sphere(vec3(0,0.5,-3.5), 2., new metal(vec3(0.2,0.6,0.8),0.0));
        // *(d_list+4) = new cylinder(vec3(-1.5,-0.2,-2),unit_vector(vec3(1.0,1.0,0)),0.3, new metal(vec3(0.8,0.8,0.8),0.3));
        *d_world    = new hitable_list(d_list,6);
        *d_camera = new camera();
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    delete *(d_list);
    delete *(d_list+1);
    delete *(d_list+2);
    delete *(d_list+3);
    delete *(d_list+4);
    delete *(d_list+5);
    delete *d_world;
    delete *d_camera;
}

int main(){
    // allocate an nx*ny image-sized frame buffer (FB) on the host 
    // to hold the RGB float values calculated by the GPU
    const int nx = 1200;
    const int ny = 600;
    const int ns = 100; //number of samples
    int tx = 8; //thread number
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << "blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 6*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks,threads>>>(nx,ny,d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks,threads>>>(fb,nx,ny,ns,d_camera,d_world,d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--){
        for (int i = 0; i < nx; i++){
            size_t pixel_index = j*nx + i;
            float r = fb[pixel_index].r();
            float g = fb[pixel_index].g();
            float b = fb[pixel_index].b();
            int ir = int(255.999*r);
            int ig = int(255.999*g);
            int ib = int(255.999*b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
    cudaDeviceReset();
}