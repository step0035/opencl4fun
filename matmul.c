#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
 
#define PROGRAM_FILE "matmul.cl"
#define KERNEL_FUNC "matmul"
#define WIDTH_A     8
#define HEIGHT_A    64
#define WIDTH_B     1
#define HEIGHT_B    8

// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                           "\n" \
"__kernel void matmul(  __global int *outputC,                      \n" \
"                       int widthA,                                   \n" \
"                       int heightA,                                  \n" \
"                       int widthB,                                   \n" \
"                       int heightB,                                  \n" \
"                       __global int *inputA,                       \n" \
"                       __global int *inputB)                       \n" \
"{                                                                    \n" \
"    // X axis                                                        \n" \
"    int col = get_global_id(0);                                      \n" \
"                                                                     \n" \
"    // Y axis                                                        \n" \
"    int row = get_global_id(1);                                      \n" \
"                                                                     \n" \
"    int sum = 0;                                                   \n" \
"                                                                     \n" \
"    for (size_t i=0; i<widthA; i++)                                  \n" \
"    {                                                                \n" \
"        sum += inputA[row * widthA + i] * inputB[i * widthB + col];  \n" \
"    }                                                                \n" \
"                                                                     \n" \
"    outputC[row * widthB + col] = sum / 256;                               \n" \
"}                                                                    \n" \
                                                                     "\n" ;
int main( int argc, char* argv[] )
{
    // matrices width and height
    int wA = WIDTH_A;
    int hA = HEIGHT_A;
    int wB = WIDTH_B;
    int hB = HEIGHT_B;
    int wC = wB;
    int hC = hA;
 
    // Host input vectors
    int h_a[HEIGHT_A][WIDTH_A] = {
        {160,129,104,71,184,193,55,86},
        {39,31,145,150,32,65,90,45},
        {153,159,93,24,149,136,108,61},
        {133,73,183,124,151,39,68,67},
        {91,24,115,144,125,0,255,42},
        {148,92,126,71,175,204,100,156},
        {170,149,168,58,198,192,48,86},
        {91,79,47,91,104,71,146,75},
        {64,73,151,145,78,41,78,39},
        {96,12,126,98,177,40,118,78},
        {26,108,116,129,0,85,54,92},
        {135,167,172,73,161,159,58,64},
        {116,104,171,124,132,166,108,72},
        {57,64,157,177,83,71,156,22},
        {146,14,127,144,151,94,0,45},
        {138,111,101,84,175,136,48,103},
        {137,140,111,45,157,255,98,75},
        {161,156,172,124,170,189,110,125},
        {69,64,77,98,92,178,136,67},
        {151,181,181,82,175,152,95,89},
        {104,90,177,106,121,128,85,89},
        {61,25,71,117,129,53,63,45},
        {255,34,255,236,192,36,60,136},
        {95,2,118,98,124,85,115,50},
        {166,219,156,82,253,225,43,120},
        {92,184,107,84,120,153,83,67},
        {43,41,119,98,57,55,68,117},
        {137,117,144,131,147,165,88,83},
        {117,73,127,91,142,149,33,92},
        {89,111,129,137,210,63,75,86},
        {83,60,214,124,60,49,136,92},
        {140,19,141,137,162,57,15,33},
        {86,12,114,157,83,94,93,106},
        {156,225,129,98,170,207,141,97},
        {118,18,111,137,138,70,43,42},
        {100,138,116,71,143,201,88,72},
        {148,41,47,58,173,140,108,22},
        {177,159,141,11,184,251,141,58},
        {96,0,88,111,122,65,173,45},
        {149,226,137,112,162,172,100,100},
        {160,214,149,61,200,214,90,117},
        {136,116,197,116,170,191,141,103},
        {124,141,107,8,143,146,90,83},
        {73,86,81,55,74,88,93,184},
        {64,40,130,161,63,76,161,56},
        {58,43,131,157,51,136,38,42},
        {180,167,145,45,255,188,80,114},
        {117,135,101,65,120,137,123,78},
        {83,6,79,111,116,80,43,61},
        {137,23,116,98,161,85,183,50},
        {152,184,183,157,175,182,125,86},
        {120,110,203,65,138,152,100,111},
        {191,255,153,78,203,213,136,106},
        {0,44,0,0,81,90,90,50},
        {83,21,48,71,78,135,125,45},
        {118,73,164,99,122,90,125,78},
        {123,175,183,117,170,184,141,173},
        {111,185,149,53,129,226,83,72},
        {176,184,156,80,212,181,50,106},
        {146,183,186,124,157,220,95,139},
        {96,120,115,111,129,97,118,255},
        {126,200,181,86,175,185,110,125},
        {111,80,115,108,101,114,118,45},
        {141,144,146,66,157,216,88,159}
    };

    int h_b[HEIGHT_B] = {
        75,
        66,
        57,
        0,
        1,
        46,
        14,
        37
    };

    // Host output vector
    int *h_c = (int *) calloc(hC * wC, sizeof(int));
 
    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;
 
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
#if 0
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
    printf("%s\r\n", fbuffer);

    program = clCreateProgramWithSource(context, 1,
                            (const char **) & fbuffer, NULL, &err);

#else
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernelSource, NULL, &err);
#endif
 
    // Build the program executable 
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "matmul", &err);
 
    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, wA*hA*sizeof(int), NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, wB*hB*sizeof(int), NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, wC*hC*sizeof(int), NULL, NULL);
 
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   wA*hA*sizeof(int), h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   wB*hB*sizeof(int), h_b, 0, NULL, NULL);
 
    if (err != CL_SUCCESS)
    {
        perror("Failed to clCreateBuffer");
        exit(1);
    }

    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_int), &wA);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_int), &hA);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &wB);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_int), &hB);
    err |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_b);
 
    if (err != CL_SUCCESS)
    {
        perror("Failed to clSetKernelArg");
        exit(1);
    }

    // Execute the kernel over the entire range of the data set  
    size_t globalws[2] = {wC, hC};
    size_t localws[2] = {2, 2};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalws, localws,
                                                              0, NULL, NULL);

    if (err != CL_SUCCESS)
    {
        perror("Failed to clEnqueueNDRangeKernel");
        exit(1);
    }
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
 
    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                                wC*hC*sizeof(int), h_c, 0, NULL, NULL );
 
    // print to verify
    for(size_t i=0; i<wC*hC; i++)
    {
        printf("%d\r\n", h_c[i]);
    }
 
    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    //release host memory
    free(h_c);
 
    return 0;
}
