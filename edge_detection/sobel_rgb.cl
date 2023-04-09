__kernel void sobel(__read_only image2d_t inputImage, __write_only image2d_t outputImage, int width, int height)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    int2 dim = (int2)(width, height);
    
    float kernel_x[3][3] = { {-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1} };
                             
    float kernel_y[3][3] = { {-1, -2, -1},
                             { 0,  0,  0},
                             { 1,  2,  1} };
                             
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;
    
    for(int j=-1; j<=1; j++) 
    {
        for(int i=-1; i<=1; i++)
        {
            int2 offset = (int2)(i, j);
            int2 coord_offset = clamp(coord + offset, (int2)(0, 0), dim - 1);
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
        }
    }
    
    float4 gradient = (float4)(sqrt(sum_x * sum_x + sum_y * sum_y + sum_z * sum_z), 0.0f, 0.0f, 0.0f);
    write_imagef(outputImage, coord, gradient);
}

