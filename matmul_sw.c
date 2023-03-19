#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

char buf1[1821];
char buf2[22];
int matrix_a[64][8];
int matrix_b[8];
int matrix_r[64];
int sum_of_prod_row;

struct timeval time_curr;
unsigned int start_time;
unsigned int end_time;

int main()
{
    size_t filesize;
    char *token1;
    char *token2;
    char *rest1;
    char *rest2;
    int row;
    int col;

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
            matrix_a[row][col] = atoi(token2);
            col++;
        }
        row++;
    }

    row = 0;
    rest1 = buf2;
    while((token1 = strtok_r(rest1, "\n", &rest1)))
    {
        matrix_b[row] = atoi(token1);
        row++;
    }
    
    // get start_time
    gettimeofday(&time_curr, NULL);
    start_time = time_curr.tv_sec * (int) 1e6 + time_curr.tv_usec;

    for (size_t i=0; i<64; i++)
    {
        matrix_r[i] = 0;
        for (size_t j=0; j<8; j++)
        {
            matrix_r[i] += matrix_a[i][j] * matrix_b[j];
        }
        matrix_r[i] /= 256;
    }

    // get end_time
    gettimeofday(&time_curr, NULL);
    end_time = time_curr.tv_sec * (int) 1e6 + time_curr.tv_usec;

    for (size_t i=0; i<64; i++)
    {
        printf("%d\r\n", matrix_r[i]);
    }

    printf("Time taken = %d ms\r\n", end_time - start_time);
}
