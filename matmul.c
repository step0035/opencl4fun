#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <CL/cl.h>
 
#define PROGRAM_FILE "matmul.cl"
#define KERNEL_FUNC "matmul"
#define WIDTH_A     8
#define HEIGHT_A    64
#define WIDTH_B     1
#define HEIGHT_B    8

char buf1[1821];
char buf2[22];

struct timeval time_curr;
unsigned int start_time;
unsigned int end_time;

int main( int argc, char* argv[] )
{
    printf("OpenCL GPU Matmul\r\n");
    // matrices width and height
    int wA = WIDTH_A;
    int hA = HEIGHT_A;
    int wB = WIDTH_B;
    int hB = HEIGHT_B;
    int wC = wB;
    int hC = hA;
    // Host input vectors
    int h_a[HEIGHT_A][WIDTH_A];
    int h_b[HEIGHT_B];
    // Host output vector
    int *h_c;

    size_t filesize;
    char *token1;
    char *token2;
    char *rest1;
    char *rest2;
    int row;
    int col;

    // Read csv files for host input vectors
    FILE* fp = fopen("A.csv", "r");
    if(fp != NULL)
    {
        filesize = fread(buf1, sizeof(char), 1821, fp);
        fclose(fp);
    }

    fp = fopen("B.csv", "r");
    if(fp != NULL)
    {
        filesize = fread(buf2, sizeof(char), 22, fp);
        fclose(fp);
    }

    row = 0;
    rest1 = buf1;
    while((token1 = strtok_r(rest1, "\n", &rest1)))
    {
        col = 0;
        rest2 = token1;
        while((token2 = strtok_r(rest2, ",", &rest2)))
        {
            h_a[row][col] = atoi(token2);
            col++;
        }
        row++;
    }

    row = 0;
    rest1 = buf2;
    while((token1 = strtok_r(rest1, "\n", &rest1)))
    {
        h_b[row] = atoi(token1);
        row++;
    }

    // allocate host output vector
    h_c = (int *) calloc(hC * wC, sizeof(int));
 
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

    program = clCreateProgramWithSource(context, 1,
                            (const char **) & fbuffer, NULL, &err);
 
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

    size_t globalws[2] = {wC, hC};
    size_t localws[2] = {1, 1};

    // get start_time
    gettimeofday(&time_curr, NULL);
    start_time = time_curr.tv_sec * (int) 1e6 + time_curr.tv_usec;

    // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalws, localws,
                                                              0, NULL, NULL);

    if (err != CL_SUCCESS)
    {
        perror("Failed to clEnqueueNDRangeKernel");
        exit(1);
    }

    // get end_time
    gettimeofday(&time_curr, NULL);
    end_time = time_curr.tv_sec * (int) 1e6 + time_curr.tv_usec;
 
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

    printf("Time taken = %d ms\r\n", end_time - start_time);
 
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
