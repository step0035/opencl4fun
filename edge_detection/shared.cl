#define WGSIZE 16
#define RADIUS 1
#define SHAREDDIM_X (WGSIZE + 2 * RADIUS)
#define SHAREDDIM_Y (WGSIZE + 2 * RADIUS)

__constant float kernel_x[3][3] = { {-1, 0, 1},
                                    {-2, 0, 2},
                                    {-1, 0, 1} };

__constant float kernel_y[3][3] = { {-1, -2, -1},
                                    { 0,  0,  0},
                                    { 1,  2,  1} };

__kernel void sobel(__read_only image2d_t inputImage, __write_only image2d_t outputImage, int width, int height)
{
    const int2 coord = (int2)(get_global_id(0), get_global_id(1));
    const int2 dim = (int2)(width, height);
    const int2 localCoord = (int2)(get_local_id(0), get_local_id(1));
    const int2 groupOffset = (int2)(get_group_id(0) * WGSIZE, get_group_id(1) * WGSIZE);

    __local float4 sharedImage[SHAREDDIM_X][SHAREDDIM_Y];

    const int2 globalCoord = groupOffset + localCoord;

    const int2 localCoordWithRadius = (int2)(localCoord.x + RADIUS, localCoord.y + RADIUS);

    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for(int j=-RADIUS; j<=RADIUS; j++) {
        for(int i=-RADIUS; i<=RADIUS; i++) {
            const int2 offset = (int2)(i, j);
            const int2 coord_offset = globalCoord + offset;
            const int2 sharedCoord_offset = localCoordWithRadius + offset;
            sharedImage[sharedCoord_offset.x][sharedCoord_offset.y] = read_imagef(inputImage, CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR, coord_offset);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int j=-RADIUS; j<=RADIUS; j++) {
        for(int i=-RADIUS; i<=RADIUS; i++) {
            const int2 offset = (int2)(i, j);
            const int2 sharedCoord_offset = localCoordWithRadius + offset;
            sum.x += sharedImage[sharedCoord_offset.x][sharedCoord_offset.y].x * kernel_x[i+1][j+1];
            sum.y += sharedImage[sharedCoord_offset.x][sharedCoord_offset.y].y * kernel_x[i+1][j+1];
            sum.z += sharedImage[sharedCoord_offset.x][sharedCoord_offset.y].z * kernel_x[i+1][j+1];
            sum.x += sharedImage[sharedCoord_offset.x][sharedCoord_offset.y].x * kernel_y[i+1][j+1];
            sum.y += sharedImage[sharedCoord_offset.x][sharedCoord_offset.y].y * kernel_y[i+1][j+1];
            sum.z += sharedImage[sharedCoord_offset.x][sharedCoord_offset.y].z * kernel_y[i+1][j+1];
        }
    }

    const float4 gradient = (float4)(sqrt(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z), 0.0f, 0.0f, 0.0f);
    write_imagef(outputImage, coord, gradient);
}

