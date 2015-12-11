#include "median9.h"

// Helper minimum and maximum functions.
int minn(int x, int y) {
  return (x > y) ? y : x;
}

int maxx(int x, int y) {
  return (x > y) ? x : y;
}

/*
check the boundary of the buffer. Set the value to the boundary 
value if the index is negative or greater than the size
*/
float get_values(__global __read_only float *in_values,
              int w, int h,
              int x, int y)
{
    int a = maxx(minn(x, w-1), 0);
    int b = maxx(minn(y, h-1), 0);
    return in_values[b*w + a];
}

// anisotropic diffusion function g
inline float g(float dif, float kappa){
    return exp(-1.0*(dif*dif/(kappa * kappa)));
}

// Read from the global memory directly (no buffer)
__kernel void
aniso_nobufferparallel(__global __read_only float *in_values,
           __global __write_only float *out_values,
           int w, int h,
           int buf_w, int buf_h,
           const int halo,
           float lambda)
{

    const int x = get_global_id(0);
    const int y = get_global_id(1);
    float k = 35.0;
    // write output by computing the difference of the neighbors.
    if ((y < h) && (x < w)) { // stay in bounds
        // Anisotropic diffusion uses four neighbors to calculate diffusion.
        float cur_pix = in_values[y * w + x];
        
        float dif1 = get_values(in_values, w, h, x+1, y) - cur_pix;
        float dif2 = get_values(in_values, w, h, x-1, y) - cur_pix;
        float dif3 = get_values(in_values, w, h, x, y+1) - cur_pix;
        float dif4 = get_values(in_values, w, h, x, y-1) - cur_pix;
    
        out_values[y * w + x] = cur_pix;
        float value = g(dif1, k)*dif1 + g(dif2, k)*dif2 + g(dif3, k)*dif3 + g(dif4, k)*dif4;
        // Compute the updated pixel value after one iteration.
        out_values[y * w + x] += lambda/4*value;    
    }   
}

// block parallel.
__kernel void
aniso_blockparallel(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer,
           int w, int h,
           int buf_w, int buf_h,
           const int halo,
           float lambda)
{

  
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    int row;
    float k = 35.0;
    if (idx_1D < buf_w)
        for (row = 0; row < buf_h; row++) {
            buffer[row * buf_w + idx_1D] = \
                get_values(in_values, w, h, buf_corner_x + idx_1D, buf_corner_y + row);
        }

    barrier(CLK_LOCAL_MEM_FENCE);

    // write output by computing the difference of the neighbors
    if ((y < h) && (x < w)) { // stay in bounds
        
        // Anisotropic diffusion uses four neighbors to calculate diffusion.
        float cur_pix = buffer[buf_y * buf_w + buf_x];
        float dif1 = buffer[buf_y * buf_w + buf_x+1] - cur_pix;
        float dif2 = buffer[buf_y * buf_w + buf_x-1] - cur_pix;
        float dif3 = buffer[(buf_y+1) * buf_w + buf_x] - cur_pix;
        float dif4 = buffer[(buf_y-1) * buf_w + buf_x] - cur_pix;

        out_values[y * w + x] = cur_pix;
        float value = g(dif1, k)*dif1 + g(dif2, k)*dif2 + g(dif3, k)*dif3 + g(dif4, k)*dif4;
        // Compute the updated pixel value after one iteration.
        out_values[y * w + x] += lambda/4*value;      
    }   

}


// column parallel
__kernel void
aniso_colparallel(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer,
           int w, int h,
           int buf_w, int buf_h,
           const int halo,
           float lambda)
{

    const int x = get_global_id(0);
    int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    int row;
    float k = 35.0;
    // The column wise parallel programming uses a loop to read in data to the buffer for each work group.
    for (int base = 0; base < h; base += buf_h - 2) {
        if (idx_1D < buf_w)
            for (row = 0; row < buf_h; row++) {
                buffer[row * buf_w + idx_1D] = \
                    get_values(in_values, w, h, buf_corner_x + idx_1D, buf_corner_y + row + base);
            }
        barrier(CLK_LOCAL_MEM_FENCE);
        if ((y < h) && (x < w)) { // stay in bounds
    
            float cur_pix = buffer[buf_y * buf_w + buf_x];
            float dif1 = buffer[buf_y * buf_w + buf_x+1] - cur_pix;
            float dif2 = buffer[buf_y * buf_w + buf_x-1] - cur_pix;
            float dif3 = buffer[(buf_y+1) * buf_w + buf_x] - cur_pix;
            float dif4 = buffer[(buf_y-1) * buf_w + buf_x] - cur_pix;

            out_values[y * w + x] = cur_pix;
            float value = g(dif1, k)*dif1 + g(dif2, k)*dif2 + g(dif3, k)*dif3 + g(dif4, k)*dif4;
            //if (y == 16 && x == 0) printf("base = %f\n", value);
            out_values[y * w + x] += lambda/4*value;      
        }
        y += buf_h-2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


// reuse buffer with the index trick.
__kernel void
aniso_reusedparallel(__global __read_only float *in_values,
           __global __write_only float *out_values,
           __local float *buffer,
           int w, int h,
           int buf_w, int buf_h,
           const int halo,
           float lambda)
{
  
    const int x = get_global_id(0);
    int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx;

    int row;
    float dif1, dif2, dif3, dif4, cur_pix, value;
    float k = 35;
    // Due to the halo, some values in the buffer can be reused. This version takes advantage of this by reading new values to the
    // local memory only when it is necessary.
    for (int base = 0; base < h; base += buf_h - 2) {
        //if (base == 0) start = 0;
        //else start = 2;
        if (base == 0) {
            if (idx_1D < buf_w)
                for (row = 0; row < buf_h; row++) {
                    buffer[row * buf_w + idx_1D] = \
                        get_values(in_values, w, h, buf_corner_x + idx_1D, buf_corner_y + row + base);
                }
            barrier(CLK_LOCAL_MEM_FENCE);
            if ((y < h) && (x < w)) { // stay in bounds
        
                cur_pix = buffer[buf_y * buf_w + buf_x];
                dif1 = buffer[buf_y * buf_w + buf_x+1] - cur_pix;
                dif2 = buffer[buf_y * buf_w + buf_x-1] - cur_pix;
                dif3 = buffer[(buf_y+1) * buf_w + buf_x] - cur_pix;
                dif4 = buffer[(buf_y-1) * buf_w + buf_x] - cur_pix;   
            }
        }
        else {
            // Index trick to avoid rereading values that are already in the buffer.
            if (idx_1D < buf_w)
                for (row = 2; row < buf_h; row++) {
                    buffer[((row + base)&3) * buf_w + idx_1D] = \
                        get_values(in_values, w, h, buf_corner_x + idx_1D, buf_corner_y + row + base);
                }
            barrier(CLK_LOCAL_MEM_FENCE);
            if ((y < h) && (x < w)) { // stay in bounds
                int c = (y+halo)&3;
                int d = (y - 1 + halo)&3;
                int u = (y + 1 + halo)&3;
                cur_pix = buffer[c * buf_w + buf_x];
                dif1 = buffer[c * buf_w + buf_x+1] - cur_pix;
                dif2 = buffer[c * buf_w + buf_x-1] - cur_pix;
                dif3 = buffer[u * buf_w + buf_x] - cur_pix;
                dif4 = buffer[d * buf_w + buf_x] - cur_pix;   
            }
        }
        if ((y < h) && (x < w)) {
            out_values[y * w + x] = cur_pix;
            value = g(dif1, k)*dif1 + g(dif2, k)*dif2 + g(dif3, k)*dif3 + g(dif4, k)*dif4;
            out_values[y * w + x] += lambda/4*value; 
        }
        y += buf_h-2;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

