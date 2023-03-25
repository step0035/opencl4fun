__kernel void sobel(__read_only image2d_t input_image,
                    __write_only image2d_t output_image,
                    const int width,
                    const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    float h_kernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    float v_kernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    float h_value = 0.0f;
    float v_value = 0.0f;
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
        for (int i = -1; i <= 1; i++)
        {
            for (int j = -1; j <= 1; j++)
            {
                float pixel = read_imagef(input_image, CLK_NORMALIZED_COORDS_FALSE, (int2)(x+j, y+i)).x;
                h_value += pixel * h_kernel[i+1][j+1];
                v_value += pixel * v_kernel[i+1][j+1];
            }
        }
        float mag = sqrt(h_value * h_value + v_value * v_value);
        if (mag > 0.2)
        {
            mag = clamp(mag, 0.0f, 1.0f);
            write_imagef(output_image, (int2)(x, y), (float4)(mag, mag, mag, 1.0f));
        }
        else 
        {
            write_imagef(output_image, (int2)(x, y), (float4)(0.0f, 0.0f, 0.0f, 1.0f));
        }
    }
};
