#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat perspective, dst, graydst, edgedst, houghdst;
Mat per2, war2, finish, blurreddst,cannydst;

vector<Vec4i> lines;

int main() {
    VideoCapture cap("lane4.mp4");
    Mat frame;
    Mat manframe;
    VideoCapture mancap("man3.mp4");

    Point2f be[4] = { Point2f(200,380), Point2f(130,470), Point2f(420,380), Point2f(540,470) };   //lane4.mp4용
    Point2f af[4] = { Point2f(0,0), Point2f(0,480), Point2f(640,0), Point2f(640,480) };

    //Point2f be[4] = { Point2f(240,380), Point2f(70,470), Point2f(390,380), Point2f(540,470) };    //linedrive.mp4용
    //Point2f af[4] = { Point2f(0,0),Point2f(0,480),Point2f(640,0),Point2f(640,480) };

    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());


    if (cap.isOpened()) {
        while (true) {
            cap >> frame;
            mancap >> manframe;
            if (frame.empty() || manframe.empty()) {
                break;
            }
            Mat src;


            resize(frame, src, Size(640, 480));
            for (int i = 0; i < 4; i++) {
                circle(src, be[i], 3, Scalar(0, 0, 255), -1);
            }
            perspective = getPerspectiveTransform(be, af);
            warpPerspective(src, dst, perspective, Size(640, 480));

            cvtColor(dst, graydst, COLOR_BGR2GRAY);
            
            GaussianBlur(graydst, blurreddst, Size(5, 5), 0);
            Canny(blurreddst, cannydst, 20, 100);

            Mat grad_x, grad_y;
            Mat abs_grad_x, abs_grad_y;
            Sobel(cannydst, grad_x, CV_16S, 1, 0, 3);
            Sobel(cannydst, grad_y, CV_16S, 0, 1, 3);
            convertScaleAbs(grad_x, abs_grad_x);
            convertScaleAbs(grad_y, abs_grad_y);
            addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, edgedst);

            Mat edgedst_thresh;
            threshold(edgedst, edgedst_thresh, 50, 255, THRESH_BINARY);

            HoughLinesP(edgedst_thresh, lines, 1, CV_PI / 180, 50, 20, 30);
            cvtColor(edgedst_thresh, houghdst, COLOR_GRAY2BGR);

            vector<Vec4i> line_R, line_L;

            for (Vec4i l : lines) {
                float slope = ((float)l[3] - l[1]) / ((float)l[2] - l[0]);
                if (slope > 0.5) {
                    line_R.push_back(l);
                }
                else if (slope < -0.5) {
                    line_L.push_back(l);
                }
            }
            for (Vec4i l : line_R) {
                line(houghdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
            }
            for (Vec4i l : line_L) {
                line(houghdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
            }

            int midX = 0;
            int avgX_R = 0, avgX_L = 0;
            int midbase = houghdst.cols / 2;
            if (!line_R.empty() && !line_L.empty()) {
                avgX_R = (line_R[0][0] + line_R[0][2]) / 2;
                avgX_L = (line_L[0][0] + line_L[0][2]) / 2;
                midX = (avgX_R + avgX_L) / 2;
                circle(houghdst, Point(midX, houghdst.rows / 2), 30, Scalar(0, 255, 0), 2, LINE_AA);
            }
            else if (!line_R.empty()) {
                avgX_R = (line_R[0][0] + line_R[0][2]) / 2;
                midX = (midbase -150 + avgX_R) / 2;
                circle(houghdst, Point(midX, houghdst.rows / 2), 30, Scalar(0, 255, 0), 2, LINE_AA);
            }
            else if (!line_L.empty()) {
                avgX_L = (line_L[0][0] + line_L[0][2]) / 2;
                midX = (midbase + 150 + avgX_L) / 2;
                circle(houghdst, Point(midX, houghdst.rows / 2), 30, Scalar(0, 255, 0), 2, LINE_AA);
            }

            line(houghdst, Point(midbase, houghdst.rows), Point(midbase, 0), Scalar(255, 0, 255), 2, LINE_AA);

            if (midX == midbase) {
                cout << "직진" << endl;
            }
            else if (midX > midbase) {
                cout << "오른쪽" << endl;
            }
            else {
                cout << "왼쪽" << endl;
            }

            per2 = getPerspectiveTransform(af, be);
            warpPerspective(houghdst, war2, per2, Size(640, 480));
            addWeighted(src, 0.5, war2, 0.5, 0, finish);

            Mat gray;
            cvtColor(manframe, gray, COLOR_BGR2GRAY);

            vector<Rect> found;
            hog.detectMultiScale(gray, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);

            int pedestrianCount = found.size();

            for (const auto& rect : found) {
                rectangle(manframe, rect, Scalar(0, 255, 0), 2);


                Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
                circle(manframe, center, 3, Scalar(255, 0, 0), -1);
            }


            putText(manframe, "pedestrian num: " + to_string(pedestrianCount), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);

            Rect roi(0, 0, frame.cols, frame.rows / 2);
            resize(manframe, manframe, Size(finish.cols, finish.rows / 2));
            manframe.copyTo(finish(Rect(0, 0, finish.cols, finish.rows / 2)));
            imshow("Final Output", finish);

            imshow("보행자 검출", manframe);


            imshow("src", src);
            imshow("gray", graydst);

            imshow("Hough Transform", houghdst);
            
            if (waitKey(10) == 27) break; // Exit on ESC key
        }
    }
    return 0;
}
