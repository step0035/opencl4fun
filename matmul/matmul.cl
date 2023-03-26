__kernel void matmul(  __global int *outputC,
                       int widthA,
                       int heightA,
                       int widthB,
                       int heightB,
                       __global int *inputA,
                       __global int *inputB)
{
    // X axis
    int col = get_global_id(0);

    // Y axis
    int row = get_global_id(1);

    int sum = 0;
    for (size_t i=0; i<widthA; i++)
    {
        sum += inputA[row * widthA + i] * inputB[i * widthB + col];
    }

    outputC[row * widthB + col] = sum / 256;
}

