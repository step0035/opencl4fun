#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

int sobel_x_kernel[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};
int sobel_y_kernel[3][3] = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}
};

void detect_edges(unsigned char* input_image, int width, int height, int channels, unsigned char* output_image) {
    int i, j, k, l, index;
    int gray_value, gx, gy, gradient_magnitude;
    unsigned char* gray_image = malloc(width*height*sizeof(unsigned char));

    if (channels == 1)
    {
        gray_image = input_image;
    }
    else if (channels == 3)
    {
        // convert rgb to grayscale by taking average of 3 channels
        for (i = 0; i < height; i++)
        {
            for (j = 0; j < width; j++)
            {
                index = (i*width+j)*channels;
                gray_value = 0;
                for (k = 0; k < channels; k++) {
                    gray_value += input_image[index+k];
                }
                gray_value /= channels;
                gray_image[i*width+j] = (unsigned char)gray_value;
            }
        }
    }
    
    // convolution
    for (i = 1; i < height-1; i++)
    {
        for (j = 1; j < width-1; j++)
        {
            gx = 0;
            gy = 0;
            for (k = -1; k <= 1; k++)
            {
                for (l = -1; l <= 1; l++)
                {
                    index = ((i+k)*width+(j+l));
                    gx += gray_image[index] * sobel_x_kernel[k+1][l+1];
                    gy += gray_image[index] * sobel_y_kernel[k+1][l+1];
                }
            }
            gradient_magnitude = (int)sqrt(gx*gx + gy*gy);
            output_image[i*width+j] = (unsigned char)gradient_magnitude;
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s <input image>\n", argv[0]);
        return -1;
    }
    char* output_filename = "output_cpu.jpg";
    int width, height, channels;
    unsigned char* input_image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!input_image)
    {
        printf("Error loading input image.\n");
        return -1;
    }

    unsigned char* output_image = malloc(width*height*sizeof(unsigned char));
    
    // measure elapse time
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    detect_edges(input_image, width, height, channels, output_image);
    gettimeofday(&end_time, NULL);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (double)(end_time.tv_usec - start_time.tv_usec)/1000000.0;
    printf("Elapsed time: %f seconds.\n", elapsed_time);

    if (!stbi_write_jpg(output_filename, width, height, 1, output_image, width*sizeof(unsigned char)))
    {
        printf("Error writing output image.\n");
        stbi_image_free(input_image);
        free(output_image);
        return -1;
    }

    stbi_image_free(input_image);
    free(output_image);
    return 0;
}
