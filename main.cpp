#include <stdio.h>
#include<iostream>

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat lImage;
    lImage = imread( argv[1], 1 );
    if ( !lImage.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", lImage);
    waitKey(0);

    double lMaxValueR, lMaxValueG, lMaxValueB;
    Mat lChannels[3];
    split(lImage, lChannels);
    minMaxIdx(lChannels[0], NULL, &lMaxValueR);
    minMaxIdx(lChannels[1], NULL, &lMaxValueG);
    minMaxIdx(lChannels[2], NULL, &lMaxValueB);

    cout << lMaxValueR << ", " << lMaxValueG << ", " << lMaxValueB << endl;

    Mat lNormalizedImage = Mat(lImage.rows, lImage.cols, CV_32FC3);

    for (int i=0 ; i<lImage.rows ; ++i) {
        for (int j=0 ; j<lImage.cols ; ++j) {
            Vec3f lTmpVect;
            lTmpVect[0] = (float)lChannels[0].at<uchar>(i,j)/(float)lMaxValueR;
            lTmpVect[1] = (float)lChannels[1].at<uchar>(i,j)/(float)lMaxValueG;
            lTmpVect[2] = (float)lChannels[2].at<uchar>(i,j)/(float)lMaxValueB;

            lNormalizedImage.at<Vec3f>(i,j) = lTmpVect;
        }
    }

    return 0;
}