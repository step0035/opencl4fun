#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "Usage: " << argv[0] << " <grayscale_image_path>" << endl;
        return -1;
    }

    // Load the input grayscale image
    Mat inputImage = imread(argv[1], IMREAD_GRAYSCALE);

    if (inputImage.empty())
    {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Apply Gaussian blur with a kernel size of 5x5
    Mat blurredImage;
    GaussianBlur(inputImage, blurredImage, Size(5, 5), 0, 0);

    // Apply Sobel filter for edge detection
    Mat sobelX, sobelY, edgeImage;
    Sobel(blurredImage, sobelX, CV_16S, 1, 0);
    Sobel(blurredImage, sobelY, CV_16S, 0, 1);

    // Convert the Sobel output images to 8-bit images
    convertScaleAbs(sobelX, sobelX);
    convertScaleAbs(sobelY, sobelY);

    // Combine the Sobel X and Y outputs to get the final edge image
    addWeighted(sobelX, 0.5, sobelY, 0.5, 0, edgeImage);

    // Show the input and output images
    imshow("Input Image", inputImage);
    imshow("Edge Image", edgeImage);
    waitKey(0);

    return 0;
}
