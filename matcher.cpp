// CamCapture.cpp: определяет точку входа для консольного приложения.
//

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"

enum {SURF_METHOD=0, SIFT_METHOD};
enum {cBLACK=0,cWHITE, cGREY, cRED, cORANGE, cYELLOW, cGREEN, cAQUA, cBLUE, cPURPLE, cPINK, NUM_COLOR_TYPES};

#define CUP_RADIUS 80
#define MIN_HESIAN 0
#define COLOR_DIFF_TRESHOLD 50
#define FEATURE_DET_METH SIFT_METHOD
const char* sCTypes[NUM_COLOR_TYPES] = {"Black", "White","Grey","Red","Orange","Yellow","Green","Aqua","Blue","Purple","Pink"};

using namespace std;
using namespace cv;

/** @function readme */
void readme()
{ std::cout << " Usage: matcher <Large Image> <Template Image>" << std::endl; }

int GetFeaturesMatchPossibility(int matchNumber)
{
    //Now just linear dependence (almost)
    int possiblity = 0;
    if (matchNumber < 4)
        possiblity = 5;
    if (matchNumber == 9)
        possiblity = 40;
    else if (matchNumber <= 20)
        possiblity = 50;
    else if (matchNumber <= 30)
        possiblity = 60;
    else if (matchNumber <= 40)
        possiblity = 70;
    else if (matchNumber <= 60)
        possiblity = 80;
    else if (matchNumber > 60)
        possiblity = 90;
    return possiblity;
}

int GetColorRatioMatchPossibility(vector<int> objColorRatio, vector<int> templColorRatio)
{
    int possiblity = 0;
    int totalDiff = 0;
    for (int i = 0; i < NUM_COLOR_TYPES; ++i)
    {
        totalDiff+= abs(templColorRatio[i] -  objColorRatio[i]) * (i == 0 || i == 1  || i == 2 ? 0.f : 1.f)/** (i == 0 ? 0.5f : 1.f)*/;
    }
    if (totalDiff <= 0)
        possiblity = 90;
    else if (totalDiff <= 15)
        possiblity = 75;
    else if (totalDiff <= 25)
        possiblity = 65;
    else if (totalDiff <= 38)
        possiblity = 50;
    else if (totalDiff <= 40)
        possiblity = 48;
    else if (totalDiff <= 45)
        possiblity = 45;
    else if (totalDiff <= 55)
        possiblity = 35;
    else
        possiblity = 15;
    return possiblity;
}

// Determine what type of color the HSV pixel is. Returns the colorType between 0 and NUM_COLOR_TYPES.
int GetPixelColorType(int H, int S, int V)
{
    int color;
    if (V < 75 || H < 5)
        color = cBLACK;
    else if (V > 190 && S < 27)
        color = cWHITE;
    else if (S < 53 && V < 185)
        color = cGREY;
    else {	// Is a color
        if (H < 18)
            color = cRED;
        else if (H < 25)
            color = cORANGE;
        else if (H < 34)
            color = cYELLOW;
        else if (H < 73)
            color = cGREEN;
        else if (H < 102)
            color = cAQUA;
        else if (H < 127)
            color = cBLUE;
        else if (H < 149)
            color = cPURPLE;
        else if (H < 175)
            color = cPINK;
        else	// full circle
            color = cRED;	// back to Red
    }
    return color;
}

void FeatureDetection(Mat img, int minHessian, bool fast,  std::vector<cv::Point2f> & corners,  std::vector<cv::KeyPoint> & keypoints, cv::Mat & descriptors, int mode = 0)
{
    if( !img.data ) 
        return; 
    if (descriptors.data)
        descriptors.release();
   if (mode == 0)
   {
       //-- Step 1: Detect the keypoints using SURF Detector
       if (!fast)
           cv::SurfFeatureDetector(minHessian).detect( img, keypoints );
       else
           cv::FastFeatureDetector(minHessian).detect( img, keypoints );
       //-- Step 2: Calculate descriptors (feature vectors)
       cv::SurfDescriptorExtractor().compute( img, keypoints, descriptors );
   }else
   {
       cv::SiftFeatureDetector (minHessian).detect( img, keypoints );
       //-- Step 2: Calculate descriptors (feature vectors)
       cv::SiftDescriptorExtractor().compute( img, keypoints, descriptors );
   }

    //Get the corners from the object
    corners[0] = (cvPoint(0,0));
    corners[1] = (cvPoint(img.cols,0));
    corners[2] = (cvPoint(img.cols,img.rows));
    corners[3] = (cvPoint(0, img.rows));
}

int FeatureMatch(Mat obj_img, Mat templ_img, 
    std::vector<cv::Point2f> obj_corners,  std::vector<cv::KeyPoint> obj_keypoints, cv::Mat obj_descriptors,
    std::vector<cv::Point2f> templ_corners,  std::vector<cv::KeyPoint> templ_keypoints, cv::Mat templ_descriptors, Mat & result)
{
    //-- Step 3: Matching descriptor vectors with a brute force matcher
    cv::FlannBasedMatcher matcher;
    std::vector<cv::vector<cv::DMatch > > matches;

    std::vector<cv::DMatch > good_matches;
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    std::vector<cv::Point2f> scene_corners(4);
    cv::Mat H;

    matcher.knnMatch( templ_descriptors, obj_descriptors, matches, 2);

    for(int i = 0; i < cv::min(obj_img.rows-1,(int) matches.size()); i++)  {

        if((matches[i][0].distance < 0.7*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))  {
            good_matches.push_back(matches[i][0]);
        }
    }
    cv::Mat img_matches;
    drawMatches( templ_img, templ_keypoints, obj_img, obj_keypoints, good_matches, img_matches );

    if (good_matches.size() >= 4)  {

        for( int i = 0; i < good_matches.size(); i++ ) {
            //Get the keypoints from the good matches
            obj.push_back( templ_keypoints[ good_matches[i].queryIdx ].pt );
            scene.push_back( obj_keypoints[ good_matches[i].trainIdx ].pt );
        }
        H = findHomography( obj, scene, CV_RANSAC );
        perspectiveTransform( templ_corners, scene_corners, H);

        //Draw lines between the corners (the mapped object in the scene image )
        line( obj_img, scene_corners[0], scene_corners[1], cvScalar(0, 255, 0), 4 );
        line( obj_img, scene_corners[1], scene_corners[2], cvScalar( 0, 255, 0), 4 );
        line( obj_img, scene_corners[2], scene_corners[3], cvScalar( 0, 255, 0), 4 );
        line( obj_img, scene_corners[3], scene_corners[0], cvScalar( 0, 255, 0), 4 );
        result = img_matches;
        //sprintf(buf, "obj_img %d", id);
        //imshow(buf, obj_img);

        return good_matches.size();
    }
    return 0;
}

vector<Vec3f> FindCups(Mat img_scene, bool showResult)
{
    vector<Vec3f> circles;
    Mat tempMat = img_scene.clone();
    //This is done so as to prevent a lot of false circles from being detected (not needed due the rouge HoughCircles parameters)
    //GaussianBlur( img_scene, tempMat, Size(3, 3), 0, 0 );
    HoughCircles(tempMat, circles, CV_HOUGH_GRADIENT,
        2,   // accumulator resolution (size of the image / 2)
        CUP_RADIUS,  // minimum distance between two circles
        10, // Canny high threshold
        100, // minimum number of votes
        CUP_RADIUS, CUP_RADIUS + 10); // min and max radius
    if (showResult)
    {
        Mat canny =  Mat(tempMat.rows, tempMat.cols, CV_8UC1);
        Mat rgbcanny =  Mat(tempMat.rows, tempMat.cols, CV_8UC3);

        cv::Canny(tempMat, canny, 50, 100, 3);
        cv::cvtColor(canny, rgbcanny, CV_GRAY2BGR);

        for (size_t i = 0; i < circles.size(); i++)
        {
            // round the floats to an int
            Vec3f p = circles[i];
            cv::Point center(cvRound(p[0]), cvRound(p[1]));
            int radius = cvRound(p[2]);
            // draw the circle center
            circle(rgbcanny, center, 3, CV_RGB(0,255,0), -1, 8, 0 );
            // draw the circle outline
            circle(rgbcanny, center, radius+1, CV_RGB(0,0,255), 2, 8, 0 );
            printf("Circle found: x: %d y: %d r: %d\n",center.x,center.y, radius);
        }
        imshow("Step 1. Looking for bodies:", rgbcanny);
    }
    return circles;
}

vector<int> FindColorRatio(Mat img)
{
    vector<int> res;
    Mat HSV;
    Mat redThreshold;
    Mat yellowThreshold;
    float initialConfidence = 1.0f;

    cvtColor(img, HSV, CV_BGR2HSV);

    int h = HSV.rows;	// Pixel height
    int w = HSV.cols;	// Pixel width
    int rowSize = HSV.step[0];	// Size of row in bytes, including extra padding
    uchar *imOfs = HSV.data;	// Pointer to the start of the image HSV pixels.
    // Create an empty tally of pixel counts for each color type
    int tallyColors[NUM_COLOR_TYPES];
    for (int i=0; i<NUM_COLOR_TYPES; i++)
        tallyColors[i] = 0;
    // Scan the image to find the tally of pixel colors
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            // Get the HSV pixel components
            uchar H = *(uchar*)(imOfs + y*rowSize + x*3 + 0);	// Hue
            uchar S = *(uchar*)(imOfs + y*rowSize + x*3 + 1);	// Saturation
            uchar V = *(uchar*)(imOfs + y*rowSize + x*3 + 2);	// Value (Brightness)

            // Determine what type of color the HSV pixel is.
            int ctype = GetPixelColorType(H, S, V);
            // Keep count of these colors.
            tallyColors[ctype]++;
        }
    }

    // Print a report about color types, and find the max tally
    //cout << "Number of pixels found using each color type (out of " << (w*h) << ":\n";
    int tallyMaxIndex = 0;
    int tallyMaxCount = -1;
    int pixels = w * h;
    //cout << endl;
    for (int i=0; i<NUM_COLOR_TYPES; i++) {
        int v = tallyColors[i];
        res.push_back(v*100/pixels);
        //cout << sCTypes[i] << " " << res.back() << "%, ";
        if (v > tallyMaxCount) {
            tallyMaxCount = tallyColors[i];
            tallyMaxIndex = i;
        }
    }
    //cout << endl;
    return res;
}

void EqualizeHistRGB(Mat & img)
{
    vector<Mat> channels; 
    Mat img_hist_equalized;

    cvtColor(img, img_hist_equalized, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format
    split(img_hist_equalized,channels); //split the image into channels
    equalizeHist(channels[0], channels[0]); //equalize histogram on the 1st channel (Y)
    merge(channels,img_hist_equalized); //merge 3 channels including the modified 1st channel into one image
    cvtColor(img_hist_equalized, img, CV_YCrCb2BGR); //change the color image from YCrCb to BGR format (to display image properly)

}

void ColorEnchance(Mat & mat)
{
    int heightc = mat.rows;
    int widthc = mat.cols;
    int temp=0;
    int units=0;

    Mat HSV, result;
    cvtColor(mat, HSV, CV_BGR2HSV);
    uchar *datac = (uchar *)HSV.data;
    int stepc= HSV.step[0];
    int channelsc= HSV.channels();

    for(int i=0;i< (heightc);i++) 
        for(int j=0;j<(widthc);j++)
        {/*Here datac means data of the HSV Image*/
            /*Here i want to Increase the saturation or the strength of the Colors in the Image and
            then I would be able to perform a good color detection*/

            temp=datac[i*stepc+j*channelsc+1]+units;/*increas the saturaion component is the second arrray.*/

            /*Here there is a small problem...when you add a value to the data and if it exceeds 255
            it starts all over again from zero and hence some of the pixels might go to zero.
            So to stop this we need to include this loop i would not try to explain the loop but
            please try and follow it is easy to do so..*/
            if(temp>255) 
                datac[i*stepc+j*channelsc+1]=255;
            else 
                datac[i*stepc+j*channelsc+1] = temp;//you may please remove and see what is happening if the if else loop is not there
        }
        cvtColor(HSV, mat, CV_HSV2BGR);
}


void readme();

/** @function main */
int main( int argc, char** argv )
{
    if (argc != 3)
    {
        readme();
        return 1;
    }
    Mat img_scene = imread(argv[1]);
    Mat img_template = imread(argv[2]);
    
    char buf[10];
    int matchFound = 0, k = 0;
    Mat img_scene_g, img_template_g, matchResult;
    float szCoef = (CUP_RADIUS * 2) / ((float)img_template.rows);

    //Step 1 - Resize the template to fit desirable cup size 
    resize(img_template, img_template, Size(cvRound(img_template.rows * szCoef), cvRound(img_template.cols * szCoef)));//resize template

    cvtColor(img_scene, img_scene_g, CV_BGR2GRAY);
    cvtColor(img_template, img_template_g, CV_BGR2GRAY);
    
    //Enhance histogram of source image
    EqualizeHistRGB(img_scene);

    //Step 2 - Find the color ratio for template (for matching)
    //There is a difficultly in using histograms matching (provided by openCV) because of undefined metric for each case
    //So we will use custom method which calculate percentage of each base color, then we will match it
    vector<int> templateColorRatio = FindColorRatio(img_template);
  
    //Step 3 - Find circles of each cup on image (result is a center and radius)
    vector<Vec3f> cupCircles = FindCups(img_scene_g, false);
    std::vector<cv::Point2f> template_corners(4), obj_corners(4); 
    std::vector<cv::KeyPoint> template_keypoints, obj_keypoints;
    Mat template_descriptors, obj_descriptors;

    //Step 4 - Calculating shape features of template (last parameter - is algorithm of feature detection. 0 - SURF, 1 - SIFT)
    FeatureDetection(img_template_g, MIN_HESIAN, false, template_corners, template_keypoints,  template_descriptors, FEATURE_DET_METH);

    vector <vector<int>> objsColorRatio;
    vector <int> objsAvgColorRatio(NUM_COLOR_TYPES);
   
    cout << "Found: '" << cupCircles.size() << "' cup circles. Starting iterate..." << endl;

    //Step 5 - Iterate over all cup circles and use combine of color ratio matching and features matching over them
    for(std::vector<Vec3f>::iterator b = cupCircles.begin(); b != cupCircles.end(); ++b) {
        //Bounding box of cup circle
        Rect cupRect = Rect (cvRound((*b)[0]) - cvRound((*b)[2]), cvRound((*b)[1]) - cvRound((*b)[2]) , 
            (cvRound((*b)[2]) ) * 2, (cvRound((*b)[2]) ) * 2);
        Mat cup_g = Mat(img_scene_g, cupRect);       
        Mat cup = Mat(img_scene,cupRect);

        //Some dataset specified enchantments
        blur( cup, cup,  Size( 2, 2 ) );
        cup.convertTo(cup, -1, 1.1f, 0);
        ColorEnchance(cup);
        
        //Drawing circles with wide border to remove the rest of image outside of cup
        circle(cup_g, Point(cvRound((*b)[2]), cvRound((*b)[2])),  
            cvRound((*b)[2]) + CUP_RADIUS, CV_RGB(255,255,255), CUP_RADIUS * 2, 8, 0 );
        circle(cup, Point(cvRound((*b)[2]), cvRound((*b)[2])),  
            cvRound((*b)[2]) + CUP_RADIUS, CV_RGB(255,255,255), CUP_RADIUS * 2, 8, 0 );
        
        //Step 6 - Find shape features of current cup circle
        FeatureDetection(cup_g, MIN_HESIAN, false, obj_corners, obj_keypoints,  obj_descriptors, FEATURE_DET_METH);
        //Step 7 - Match template and current cup features
        if (matchFound = FeatureMatch(cup_g, 
            img_template_g, obj_corners, obj_keypoints,  obj_descriptors, 
            template_corners, template_keypoints, template_descriptors, matchResult))
        {
            //Step 8 - Compute the possibility for features matching method
            int cPossibl = GetFeaturesMatchPossibility(matchFound);
            //Step 9 - Compute the possibility for color ratio matching method
            int fPossibl = GetColorRatioMatchPossibility(FindColorRatio(cup), templateColorRatio);
            //Some logic based on possibility (obtained by trial and error)
            if ((fPossibl > 50 && cPossibl > 50) || cPossibl >= 90 || fPossibl >= 90)
            {
                //Step 10 - If we are here then we found our cup!
                cv::Point center(cvRound((*b)[0]), cvRound((*b)[1]));
                circle(img_scene, center, 3, CV_RGB(0,255,0), -1, 8, 0 );
                circle(img_scene, center, cvRound((*b)[2]) + 1, CV_RGB(0,0,255), 2, 8, 0 );
                printf ("\n\n___________________________________________________________\n");
                printf ("Cup found! No: %d Center(x,y): (%d, %d), Radius: %d \n", k, cvRound((*b)[0]), cvRound((*b)[1]), cvRound((*b)[2]));
                printf ("Possibility: Matching by color ratio / features: %d / %d\n", cPossibl, fPossibl);
                printf ("___________________________________________________________");
                sprintf(buf, "Cup %d", k);
                imshow (buf, matchResult);
                ++k;
            }
        }
    }
    imshow ("result", img_scene);
    waitKey(0);
    return 0;
}

