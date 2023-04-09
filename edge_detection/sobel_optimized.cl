#define WGSIZE 16 // this must match with the wg size in host code, uses local memory
#define UNROLL 1 // enable loop unrolling
#define CONSTANT_MEM 1 // store in constant memory accessible to all threads
#define VECTORIZE 1 // enable vectorization

#if CONSTANT_MEM
__constant float kernel_x[3][3] = { {-1, 0, 1},
                                    {-2, 0, 2},
                                    {-1, 0, 1} };

__constant float kernel_y[3][3] = { {-1, -2, -1},
                                    { 0,  0,  0},
                                    { 1,  2,  1} };
#endif

__kernel void sobel(__read_only image2d_t inputImage, __write_only image2d_t outputImage, int width, int height)
{
#if !CONSTANT_MEM
float kernel_x[3][3] = { {-1, 0, 1},
                         {-2, 0, 2},
                         {-1, 0, 1} };

float kernel_y[3][3] = { {-1, -2, -1},
                         { 0,  0,  0},
                         { 1,  2,  1} };
#endif

    const int2 coord = (int2)(get_global_id(0), get_global_id(1));
    const int2 dim = (int2)(width, height);
    const int2 localCoord = (int2)(get_local_id(0), get_local_id(1));
    const int2 groupOffset = (int2)(get_group_id(0) * WGSIZE, get_group_id(1) * WGSIZE);

#if VECTORIZE
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
#else
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;
#endif

#if CONSTANT_MEM
#pragma unroll
#endif
    for(int j=-1; j<=1; j++) {
        for(int i=-1; i<=1; i++) {
            const int2 offset = (int2)(i, j);
            const int2 coord_offset = clamp(coord + offset, (int2)(0, 0), dim - 1);
#if VECTORIZE
            const float4 pixel = read_imagef(inputImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR, coord_offset);
            sum.x += pixel.x * kernel_x[i+1][j+1];
            sum.y += pixel.y * kernel_x[i+1][j+1];
            sum.z += pixel.z * kernel_x[i+1][j+1];
            sum.x += pixel.x * kernel_y[i+1][j+1];
            sum.y += pixel.y * kernel_y[i+1][j+1];
            sum.z += pixel.z * kernel_y[i+1][j+1];
#else
            float pixel_x = read_imagef(inputImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR, coord_offset).x;
            float pixel_y = read_imagef(inputImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR, coord_offset).x;
            float pixel_z = read_imagef(inputImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR, coord_offset).x;
            // x-direction
            sum_x += pixel_x * kernel_x[i+1][j+1];
            sum_y += pixel_y * kernel_x[i+1][j+1];
            sum_z += pixel_z * kernel_x[i+1][j+1];

            // y-direction
            sum_x += pixel_x * kernel_y[i+1][j+1];
            sum_y += pixel_y * kernel_y[i+1][j+1];
            sum_z += pixel_z * kernel_y[i+1][j+1];
#endif
        }
    }

#if VECTORIZE
    const float4 gradient = (float4)(sqrt(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z), 0.0f, 0.0f, 0.0f);
#else
    float4 gradient = (float4)(sqrt(sum_x * sum_x + sum_y * sum_y + sum_z * sum_z), 0.0f, 0.0f, 0.0f);
#endif

    const int2 outputCoord = groupOffset + localCoord;
    write_imagef(outputImage, outputCoord, gradient);
}
