#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PROGRAM_FILE "sobel.cl"
#define KERNEL_NAME "sobel"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("Usage: %s <input image>\n", argv[0]);
        return -1;
    }

    // read the input image
    int og_width;
    int og_height;
    int channels; // this one should be 1 since we assume input is grayscale

    unsigned char* og_image = stbi_load(argv[1], &og_width, &og_height, &channels, STBI_grey);
    if (!og_image)
    {
        perror("failed to load the input image!");
        exit(1);
    }

    printf("og_width: %d\r\n", og_width);
    printf("og_height: %d\r\n", og_height);

    // we pad the image with +1 on entire boundary to preserve the og resolution
    int padded_width = og_width + 1;
    int padded_height = og_height + 1;
    printf("padded_width: %d\r\n", padded_width);
    printf("padded_height: %d\r\n", padded_height);


    // malloc the padded image
    // NOTE: if input image is too big, there might be segmentation fault
    // input image size threshold depends on your heap size
    unsigned char* padded_image = (unsigned char*) malloc(padded_width * padded_height * channels);

    // create the padded image
    for (int y = 0; y < padded_height; y++)
    {
        for (int x = 0; x < padded_width; x++)
        {
            if (x < 1 || x >= padded_width - 1 || y < 1 || y >= padded_height - 1)
            {
                // Set the pixel to zero if it is in the border.
                for (int c = 0; c < channels; c++)
                {
                    padded_image[(y * padded_width + x) * channels + c] = 0;
                }
            }
            else
            {
                // Copy the pixel from the og image.
                int og_x = x - 1;
                int og_y = y - 1;
                for (int c = 0; c < channels; c++)
                {
                    padded_image[(y * padded_width + x) * channels + c] = og_image[(og_y * og_width + og_x) * channels + c];
                }
            }
        }
    }

    // malloc for the output data
    // should have the og_width and og_height
    unsigned char* output_data = (unsigned char*) malloc(og_width * og_height);

    // opencl initializations
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
    cl_int err;

    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
 
    // Create a context  
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
 
    // Create a command queue 
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);

    // Create the compute program from the source buffer
    // first we read the kernel code string from the file
    FILE *fhandle;
    size_t fsize;
    char *fbuffer;
    fhandle = fopen(PROGRAM_FILE, "rb");
    if (fhandle == NULL)
    {
        perror("Couldn't open the program file");
        exit(1);
    }
    fseek(fhandle, 0, SEEK_END);
    fsize = ftell(fhandle);
    rewind(fhandle);
    fbuffer = (char *) malloc(fsize + 1);
    fbuffer[fsize] = '\0';
    fread(fbuffer, sizeof(char), fsize, fhandle);
    fclose(fhandle);

    program = clCreateProgramWithSource(context, 1, (const char **) & fbuffer, NULL, &err);

    // Build the program executable 
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, KERNEL_NAME, &err);

    // Create input and output image buffers
    cl_image_format format;
    format.image_channel_order = CL_R;
    format.image_channel_data_type = CL_UNORM_INT8;

    cl_image_desc clImageDescInput;
    clImageDescInput.image_type = CL_MEM_OBJECT_IMAGE2D;
    clImageDescInput.image_width = padded_width;
    clImageDescInput.image_height = padded_height;
    clImageDescInput.image_row_pitch = 0;
    clImageDescInput.image_slice_pitch = 0;
    clImageDescInput.num_mip_levels = 0;
    clImageDescInput.num_samples = 0;
    clImageDescInput.buffer = NULL;

    cl_image_desc clImageDescOutput;
    clImageDescOutput.image_type = CL_MEM_OBJECT_IMAGE2D;
    clImageDescOutput.image_width = og_width;
    clImageDescOutput.image_height = og_height;
    clImageDescOutput.image_row_pitch = 0;
    clImageDescOutput.image_slice_pitch = 0;
    clImageDescOutput.num_mip_levels = 0;
    clImageDescOutput.num_samples = 0;
    clImageDescOutput.buffer = NULL;

    cl_mem input_image = clCreateImage(context, CL_MEM_READ_ONLY, &format, &clImageDescInput, NULL, &err);
    if (err != CL_SUCCESS)
    {
        perror("failed to create input image buffer");
        exit(1);
    }

    cl_mem output_image = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &clImageDescOutput, NULL, &err);
    if (err != CL_SUCCESS)
    {
        perror("failed to create output image buffer");
        exit(1);
    }

    // write the input image into the buffer
    size_t origin[3] = {0, 0, 0};
    size_t input_region[3] = {padded_width, padded_height, 1};
    err = clEnqueueWriteImage(queue, input_image, CL_TRUE, origin, input_region, 0, 0, padded_image, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        perror("failed to write input image to buffer");
        exit(1);
    }

    // set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_image);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_image);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &padded_width);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &padded_height);
    if (err != CL_SUCCESS)
    {
        perror("failed to set kernel arguments");
        exit(1);
    }

    // Execute kernel
    size_t global_ws[2] = {og_width, og_height};
    size_t local_ws[2] = {1, 1};

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_ws, local_ws, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        perror("failed to execute kernel");
        exit(1);
    }
    // flush the command queue
    clFlush(queue);

    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    // Read output image data from buffer
    size_t output_region[3] = {og_width, og_height, 1};
    err = clEnqueueReadImage(queue, output_image, CL_TRUE, origin, output_region, 0, 0, output_data, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        perror("failed to read output image from buffer");
        exit(1);
    }

    // finally, we write the output into a jpg image
    // this output image should be (og_width * og_height)
    stbi_write_jpg("output.jpg", og_width, og_height, 1, output_data, 100);

    // Cleanup
    clReleaseMemObject(input_image);
    clReleaseMemObject(output_image);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(output_data);
    free(padded_image);

    return 0;
}
