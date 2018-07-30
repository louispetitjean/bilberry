#include <stdio.h>
#include<iostream>

#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    // Read the input image
    if ( argc != 3 )
    {
        printf("usage: ./Main <Image_Path> <Image_Results_Path>\n");
        return -1;
    }
    Mat lImageBGR;
    lImageBGR = imread( argv[1], 1 );
    if ( !lImageBGR.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );

    // Initialize all the needed images
    Mat lNormalizedImage = Mat(lImageBGR.rows, lImageBGR.cols, CV_32FC3);
    Mat lChromaticImage = Mat(lImageBGR.rows, lImageBGR.cols, CV_32FC3);
    Mat lExR = Mat(lImageBGR.rows, lImageBGR.cols, CV_32FC1);
    Mat lExG = Mat(lImageBGR.rows, lImageBGR.cols, CV_32FC1);
    Mat lDiff = Mat(lImageBGR.rows, lImageBGR.cols, CV_32FC1);
    Mat lThresholded = Mat(lImageBGR.rows, lImageBGR.cols, CV_8UC1);


    // Care, image is BGR, not RGB
    for (int i=0 ; i<lImageBGR.rows ; ++i) {
        for (int j=0 ; j<lImageBGR.cols ; ++j) {
            Vec3b lRGBPixels;
            lRGBPixels = lImageBGR.at<Vec3b>(i,j);

            // Compute normalized pixels
            Vec3f lNormalizedVect, lChromaticVect;
            lNormalizedVect[0] = (float)lRGBPixels[0]/255.0f;
            lNormalizedVect[1] = (float)lRGBPixels[1]/255.0f;
            lNormalizedVect[2] = (float)lRGBPixels[2]/255.0f;
            lNormalizedImage.at<Vec3f>(i,j) = lNormalizedVect;

            // Compute chromatic pixels
            float lSum;
            lSum = lNormalizedVect[0] + lNormalizedVect[1] + lNormalizedVect[2];
            lChromaticVect[0] = lNormalizedVect[0]/lSum;
            lChromaticVect[1] = lNormalizedVect[1]/lSum;
            lChromaticVect[2] = lNormalizedVect[2]/lSum;
            lChromaticImage.at<Vec3f>(i,j) = lChromaticVect;

            // ExG = 2G - R - B
            // ExR = 1.4R - G
            float lExGPixel, lExRPixel;
            lExRPixel = 1.4*lChromaticVect[2] - lChromaticVect[1];
            lExGPixel = 2*lChromaticVect[1] - lChromaticVect[0] - lChromaticVect[2];
            lExR.at<float>(i,j) = lExRPixel;
            lExG.at<float>(i,j) = lExGPixel;

            // Threshold
            lDiff.at<float>(i,j) = max(0.0f, lExGPixel - lExRPixel);
            if (lExGPixel - lExRPixel > 0)
                lThresholded.at<uchar>(i,j) = 255;
            else
                lThresholded.at<uchar>(i,j) = 0;

        }
    }

    // Display thresholded image
    imshow("Display Image", lThresholded);
    waitKey(0);


    // Post-process image
    blur(lThresholded, lThresholded, Size(9,9));
    // Threshold means 15 white pixels in the neighbourhoud
    threshold(lThresholded, lThresholded, 47, 255, 0);

    imshow("Display Image", lThresholded);
    waitKey(0);

    int lDilationSize = 9;
    Mat lElement = getStructuringElement(0,
                                       Size( 2*lDilationSize + 1, 2*lDilationSize+1 ),
                                       Point( lDilationSize, lDilationSize ) );
  /// Apply the dilation operation
    dilate(lThresholded, lThresholded, lElement );
    
    imshow("Display Image", lThresholded);
    waitKey(0);


    // Compute contours and draw bounding-boxes on the image
    vector<vector<Point> > lContours;
    vector<Vec4i> lHierarchy;
    findContours(lThresholded, lContours, lHierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    
    vector<Rect> lMinRect(lContours.size());
    vector<vector<Point> > contours_poly(lContours.size());
    for (int i = 0; i < lContours.size(); ++i)
    {
        approxPolyDP(Mat(lContours[i]), contours_poly[i], 5, true);        
        lMinRect[i] = boundingRect(Mat(contours_poly[i]));
        rectangle(lImageBGR, lMinRect[i].tl(), lMinRect[i].br(), Scalar(0,0,255));
    }
    imshow("Display Image", lImageBGR);
    waitKey(0);

    imwrite(argv[2], lImageBGR);
 
    return 0;
}