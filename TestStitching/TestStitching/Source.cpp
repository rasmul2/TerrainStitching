#include <opencv2/core/core.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\stitching.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video/tracking.hpp>



//TO DO THE SHOT.JPG HAS THE CURRENT FRAME OF THE SCENE, TRY THIS

using namespace cv;
using namespace std;

float inlier_threshold = 8.5f; // Distance threshold to identify inliers
float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

vector<Mat> imgs;
vector<Mat> previous;

vector < vector<Mat> > Grid;

Point2f runningdistance;
Point2f previousrunningdistance;
int xloc = 0;
int yloc = 0;

int xprevious = 0;
int yprevious = 0;

int tempwidth = 500;
int tempheight = 300;
int adjustoffsetx = 0;
int adjustoffsety = 0;
int setx = 0;
int sety = 0;

bool first = true;
bool skip = false;
bool xchanged = false;
bool ychanged = false;

bool missed = false;
bool rotateThis = false;

float brightness = 0;
float contrast = 1.0;

double requiredratio = 0.9;

float collectiveangle = 0;

string movementx = "";
string previousmovementx = "";
string movementy = "";
string previousmovementy = "";
string folder = "C:/Users/swri11/Documents/GitHub/ChunkedTerrain/";

using namespace cv;
using namespace std;

Mat RotateImage(Mat image, double angle) {
	if (image.rows > 0) {
		collectiveangle += angle;
		if (collectiveangle > 18 || collectiveangle < -18) {
			if (collectiveangle > 36 || collectiveangle < -36) {
				collectiveangle = 0;
			}
		}

		int h = image.rows / 2;
		int w = image.cols / 2;

		cout << "The height of the image is " << h << "the width is " << w << endl;
		cout << "The angle is " << angle << endl;

		double scale = 1;
		Mat RotationMatrix = getRotationMatrix2D(Point2f(w, h), angle, scale);

		warpAffine(image, image, RotationMatrix, image.size(), WARP_INVERSE_MAP, BORDER_TRANSPARENT);
		RotationMatrix.release();
	}
	
	return image;
}

 Mat AddNextImage(vector<Mat> images, vector<Point2f> dist, vector<double> angls, Mat scan) {
	if (dist.size() > 0) {

		//average distances
		float xdistance = 0;
		float ydistance = 0;

		for (int x = 0; x < dist.size(); x++) {
			xdistance += dist[x].x;
			ydistance += dist[x].y;
		}

		xdistance /= dist.size();
		ydistance /= dist.size();



		runningdistance += Point2f(xdistance, ydistance);

		cout << "The distances are : " << xdistance << " x and " << ydistance << endl;
		cout << "The size being divided by is " << dist.size() << endl; 
		//cout << "The running distances for the windows edges are : " << runningdistance.x << " " << runningdistance.y;
		if (xdistance< 0 && ydistance < 0) {
			Mat resize;
			Rect rect = Rect(0, 0, images[1].cols, images[1].rows / 2.5);
			images[1] = images[1](rect);
			if (rotateThis == true) {
				images[1] = RotateImage(images[1], -angls[0]);
				//resize = RotateImage(resize , -angles[0]);
			}
			copyMakeBorder(scan, resize, abs(ydistance), 0, abs(xdistance), 0, 0);
			images[1].copyTo(resize(Rect(adjustoffsetx, adjustoffsety, images[1].cols, images[1].rows)));
			//imshow("Cropped", resize);

		
			imwrite("res.png", resize);

			scan = resize;
			images.clear();
		}
		if (xdistance > 0 && ydistance < 0) {
			Mat resize;
			Rect rect = Rect(0, 0, images[1].cols, images[1].rows / 2.5);
			images[1] = images[1](rect);
			if (rotateThis == true) {
				images[1] = RotateImage(images[1], angls[0]);
				//resize = RotateImage(resize, angles[0]);
			}
			copyMakeBorder(scan, resize, abs(ydistance), 0, 0, abs(xdistance), 0);
			images[1].copyTo(resize(Rect(xdistance + adjustoffsetx, adjustoffsety, images[1].cols, images[1].rows)));

			//imshow("Cropped", resize);
			imwrite("res.png", resize);

			adjustoffsetx += xdistance;

			scan = resize;
			images.clear();
		}
		if (xdistance > 0 && ydistance > 0) {
			Mat resize;
			Rect rect = Rect(0, 0, images[1].cols, images[1].rows / 2.5);
			images[1] = images[1](rect);
			if (rotateThis == true) {
				images[1] = RotateImage(images[1], angls[0]);
				//resize = RotateImage(resize, angles[0]);
			}
			copyMakeBorder(scan, resize, 0, abs(ydistance), 0, abs(xdistance), 0);
			images[1].copyTo(resize(Rect(xdistance + adjustoffsetx, ydistance + adjustoffsety, images[1].cols, images[1].rows)));
			//imshow("Cropped", resize);

			imwrite("res.png", resize);
			scan = resize;

			adjustoffsetx += xdistance;
			adjustoffsety += ydistance;


			images.clear();
		}
		if (xdistance < 0 && ydistance > 0) {
			Mat resize;
			Rect rect = Rect(0, 0, imgs[1].cols, imgs[1].rows / 2.5);
			images[1] = images[1](rect);
			if (rotateThis == true) {
				images[1] = RotateImage(images[1], -angls[0]);
				//resize = RotateImage(resize, -angles[0]);
			}
			copyMakeBorder(scan, resize, 0, abs(xdistance), abs(xdistance), 0, 0);

			images[1].copyTo(resize(Rect(adjustoffsetx, ydistance + adjustoffsety, images[1].cols, images[1].rows)));
			//imshow("Cropped", resize);

			imwrite("res.png", resize);

			adjustoffsety += ydistance;

			scan = resize;
			images.clear();
		}


		runningdistance.x = roundf(runningdistance.x);
		runningdistance.y = roundf(runningdistance.y);

		cout << "The rounded distance is :" << runningdistance.x << " " << runningdistance.y << endl;
		cout << scan.cols / 312 << " and rows " << scan.rows / 312 << endl;
		//if x is filled out
		return scan;
	}
	else {
		cout << "Distances was empty" << endl;
		return scan;
	}
}

vector<Point2f> GetDistancesAngle(vector<Point2f> mp1, vector<Point2f> mp2) {
	double largestangle = 0;
	vector<Point2f> dist;
	for (int x = 0; x < mp1.size(); x++) {
		Point2f distance;
		double angle;
		float len1 = sqrt(mp1[x].x * mp1[x].x + mp1[x].y * mp1[x].y);
		float len2 = sqrt(mp2[x].x * mp2[x].x + mp2[x].y * mp2[x].y);

		float dot = mp1[x].x * mp2[x].x + mp1[x].y * mp1[x].y;

		float a = dot / (len1 * len2);

		//cout << "The dot/length of the angle is: " << a << endl;

		if (a >= 1.0)
			angle = 0.0;
		else if (a <= -1.0)
			angle = 3.14;
		else
			angle = acos(a);

		if (largestangle < angle) {
			largestangle = angle;
		}
		distance.x = mp1[x].x - mp2[x].x;
		distance.y = mp1[x].y - mp2[x].y;
		if (distance.x < .15 && distance.y < .15) {
			rotateThis = true;
		}
		dist.push_back(distance);
	}
	dist.push_back(Point2f(largestangle, 0));
	return dist;
}

void Chunk(Mat total) {
	//for the corners
	cout << "The collective angle is " << collectiveangle << endl;
	cout << "The totals minus the offsets are " << total.rows - adjustoffsety << " " << total.cols - adjustoffsetx << endl;
	cout << "Just the offsets are " << adjustoffsety << " " << adjustoffsetx << endl;
	cout << "The running distances are " << runningdistance.y << ' ' << runningdistance.x << endl;
	cout << "The Previous running distances are " << previousrunningdistance.y << ' ' << previousrunningdistance.x << endl;

	

	//just a y
	cout << "The location as recorded is " << xloc << " " << yloc << endl;
	if (abs(runningdistance.x-previousrunningdistance.x) >= 312 || abs(runningdistance.y - previousrunningdistance.y) >= 312) {
		/*---------------------------if it fills another in the ydirection-------------------------------*/
		//look up and down of the current position yloc
		if (abs(runningdistance.x - previousrunningdistance.x) >= 312) {
			cout << "Trying to fill a x" << endl;
			cout << "The adjust of y is " << runningdistance.x << endl;

			if (runningdistance.x > previousrunningdistance.x) {
				xprevious++;
				xloc++;
				movementx = "right";
			}
			else if (runningdistance.x < previousrunningdistance.x) {
				xprevious--;
				xloc--;
				movementx = "left";
			}
		}
		else {
			if (runningdistance.y < previousrunningdistance.y) {
				yprevious++;
				yloc++;
				movementy = "up";
			}
			else if (runningdistance.y > previousrunningdistance.y) {
				yprevious--;
				yloc--;
				movementy = "down";
			}
		}
		
		/*--------------------adjustment set for change in direction-----------------------*/


		Mat temp;
		cout << "The current movement is " << movementx << " and the previous movment is " << previousmovementx << endl;
		cout << "The current movement is " << movementy << " and the previous movment is " << previousmovementy << endl;
		
				temp = total(Rect(0, 0, 312, 312));
		

		
		stringstream ss;
		ss << folder << "image" << xloc << " " << yloc << ".png";
		string filename = ss.str();
		imwrite(filename, temp);

		if (abs(runningdistance.x - previousrunningdistance.x) >= 312) {
			previousrunningdistance.x = runningdistance.x;
		}
		else {
			previousrunningdistance.y = runningdistance.y;
		}
		previousmovementx = movementx;
		previousmovementy = movementy;
	}

}

void GridGrowth() {
	//for filling in temp ones around the main one
	/*for (int i = xprevious; i <= xrowsfilled; i++) {
	for (int k = yprevious; k <= yrowsfilled; k++) {
	if (i != xprevious || k != yprevious) {
	Mat temp(312, 312, frame.type(), Scalar(0, 0, 0));
	imshow("Should be this size", temp);
	cout << "Working with temps" << endl;
	cout << "Right now x and y are: " << i << ' ' << k << endl;
	if (k == yrowsfilled  && i > 0 && i == xprevious ) {
	cout << "1" << endl;
	if (totalscan.rows - 312 * (yrowsfilled - 1) > 0) {
	Mat part = totalscan(Rect(totalscan.cols - 312 * (xrowsfilled - 1), 0, 312, totalscan.rows - 312 * (yrowsfilled - 1)));
	if (part.cols > 312 || part.rows > 312) {
	cout << "Something is wrong in the size" << endl;
	break;
	}
	part.copyTo(temp(Rect(312 - part.cols, 312 - part.rows, part.cols, part.rows)));
	}
	else {
	Mat part = totalscan(Rect(totalscan.cols - 312 * (xrowsfilled - 1), 0, 312, totalscan.rows - 312 * (yrowsfilled-2)));
	if (part.cols > 312 || part.rows > 312) {
	cout << "Something is wrong in the size" << endl;
	break;
	}
	part.copyTo(temp(Rect(312 - part.cols, 312 - part.rows, part.cols, part.rows)));
	}
	stringstream ss;
	ss << folder << "imaget" << i << " " << k << ".png";
	string filename = ss.str();
	imwrite(filename, temp);
	imshow(filename, temp);
	}
	else if (i==xrowsfilled && k > 0 && k== yrowsfilled) {
	cout << "2" << endl;
	if (totalscan.cols - 312 * (xrowsfilled - 1) > 0) {
	Mat part = totalscan(Rect(0, totalscan.rows - 312 * (yrowsfilled - 1), totalscan.cols - 312 * (xrowsfilled - 1), 312));
	if (part.cols > 312 || part.rows > 312) {
	cout << "Something is wrong in the size" << endl;
	break;
	}
	part.copyTo(temp(Rect(312 - part.cols, 312 - part.rows, part.cols, part.rows)));
	}
	else {
	Mat part = totalscan(Rect(0, totalscan.rows - 312 * (yrowsfilled - 1), totalscan.cols - 312 * (xrowsfilled - 2), 312));
	if (part.cols > 312 || part.rows > 312) {
	cout << "Something is wrong in the size" << endl;
	break;
	}
	part.copyTo(temp(Rect(312 - part.cols, 312 - part.rows, part.cols, part.rows)));
	}
	stringstream ss;
	ss << folder << "imaget" << i << " " << k << ".png";
	string filename = ss.str();
	imwrite(filename, temp);
	imshow(filename, temp);
	}
	else if (i == xrowsfilled && k == yrowsfilled && i > 0 && k > 0) {
	cout << "3" << endl;
	if (totalscan.rows - 312 * (yrowsfilled - 1) > 0  && totalscan.cols - 312 * (xrowsfilled - 1)) {
	Mat part = totalscan(Rect(0, 0, totalscan.cols - 312 * (xrowsfilled - 1), totalscan.rows - 312 * (yrowsfilled - 1)));
	if (part.cols > 312 || part.rows > 312) {
	cout << "Something is wrong in the size" << endl;
	break;
	}
	part.copyTo(temp(Rect(312 - part.cols, 312 - part.rows, part.cols, part.rows)));
	}
	else if(totalscan.rows - 312 * (yrowsfilled - 1) <= 0 && totalscan.cols - 312 * (xrowsfilled - 1)){
	Mat part = totalscan(Rect(0, 0, totalscan.cols - 312 * (xrowsfilled - 1), totalscan.rows - 312 * (yrowsfilled - 2)));
	if (part.cols > 312 || part.rows > 312) {
	cout << "Something is wrong in the size" << endl;
	break;
	}
	part.copyTo(temp(Rect(312 - part.cols, 312 - part.rows, part.cols, part.rows)));
	}
	else {
	Mat part = totalscan(Rect(0, 0, totalscan.cols - 312 * (xrowsfilled - 2), totalscan.rows - 312 * (yrowsfilled - 2)));
	if (part.cols > 312 || part.rows > 312) {
	cout << "Something is wrong in the size" << endl;
	break;
	}
	part.copyTo(temp(Rect(312 - part.cols, 312 - part.rows, part.cols, part.rows)));
	}
	stringstream ss;
	ss << folder << "imaget" << xrowsfilled << " " << yrowsfilled << ".png";
	string filename = ss.str();
	imwrite(filename, temp);
	imshow(filename, temp);
	}

	}
	}*/
}

int main(int, char* argv[])
{
	Mat frame;
	Mat totalscan;
	VideoCapture cap;
	vector< vector<KeyPoint>> keypoints;
	vector<Mat> outputimages;

	string filename = "";
	if (argv[1] != NULL) {
		 const string file = argv[1];
		 filename = file;
		 if (filename != "") {
			 cap = VideoCapture(file);
			 // Check VideoCapture documentation.
			 if (!cap.isOpened()) {  // check if we succeeded
				 cout << "Couldn't find File" << endl;
				 return -1;
			 }

			 cap.grab();
			 cap >> frame;
			 resize(frame, frame, frame.size() / 2, (0, 0), 1);
			 rotate(frame, frame, ROTATE_90_CLOCKWISE);
			 //rotate(frame, frame, ROTATE_180);
		 }
	}
	else {
		frame = imread("C:/Users/swri11/Documents/GitHub/VLCImages/images.png");
	}
	
	
	//frame.convertTo(frame, -1, contrast, brightness);
	imshow("Frame", frame);
	
	Rect rect = Rect(0, 0, frame.cols, frame.rows/2.5);
	frame = frame(rect);
	
	totalscan = frame;

	imshow("Camera_Output", totalscan);

	previousrunningdistance = Point2f(0, 0);

	int count = 0;
	while (true) {
		if (count < -1) {
			cap >> frame;
			count++;
		}
		else {
			/*if (previous.size() == 2) {
				imgs.push_back(previous[0]);
				imgs.push_back(previous[1]);
				previous.clear();
				cout << "Trying to link two previous up" << endl;
			}*/

			if (tempwidth > totalscan.cols) {
				tempwidth = totalscan.cols;
			}
			if (tempheight > totalscan.rows) {
				tempheight = totalscan.rows;
			}
			Mat temp = totalscan(Rect(adjustoffsetx, adjustoffsety, tempwidth, tempheight));


			cvtColor(temp, temp, CV_BGR2YCrCb);

			vector<Mat> channelstemp;
			split(temp, channelstemp);

			equalizeHist(channelstemp[0], channelstemp[0]);

			merge(channelstemp, temp);

			cvtColor(temp, temp, CV_YCrCb2BGR);
			temp.convertTo(temp, -1, contrast, brightness);

			imshow("Temp", temp);
			if (skip == false) {
				imgs.push_back(temp);
			}
			
			if (filename == "") {
				frame = imread("C:/Users/swri11/Documents/GitHub/VLCImages/images.png");
			}
			else {
				cap >> frame;
				resize(frame, frame, frame.size() / 2, (0, 0), 1);
				rotate(frame, frame, ROTATE_90_CLOCKWISE);
				//rotate(frame, frame, ROTATE_180);
			}

			//cap = VideoCapture("http://10.202.17.41:8080/shot.jpg");
			//cap.read(frame);
			if (!frame.empty()) {
				//Create image frames from capture
				cvtColor(frame, frame, CV_BGR2YCrCb);

				vector<Mat> channels;
				split(frame, channels);

				equalizeHist(channels[0], channels[0]);

				merge(channels, frame);

				cvtColor(frame, frame, CV_YCrCb2BGR);
				frame.convertTo(frame, -1, contrast, brightness);

				imgs.push_back(frame);
				imshow("Frame", frame);
			}


			
			

			Ptr<AKAZE> akaze = AKAZE::create();
			if (first == true || skip == true) {
				for (int k = 0; k < imgs.size(); k++) {
					vector<KeyPoint> keypoint;
					Mat outputimage = Mat();
					akaze->detectAndCompute(imgs[k], noArray(), keypoint, outputimage);
					outputimages.push_back(outputimage);
					keypoints.push_back(keypoint);
				}
			}
			else {
				for (int k = 1; k < imgs.size(); k++) {
					vector<KeyPoint> keypoint;
					Mat outputimage = Mat();
					akaze->detectAndCompute(imgs[k], noArray(), keypoint, outputimage);
					outputimages.push_back(outputimage);
					keypoints.push_back(keypoint);
				}
			}

			cout << "Here" << endl;
			cout << "The number of keypoints are " << keypoints.size() << endl;

			//have to do this for like every two images i guess
			BFMatcher matcher(NORM_HAMMING);
			vector< vector<DMatch> > nn_matches;
			for (int i = 1; i < outputimages.size(); i++){
				matcher.knnMatch(outputimages[0], outputimages[i], nn_matches, 2);
			}

			vector<KeyPoint> matched1, matched2, inliers1, inliers2;
			vector<DMatch> good_matches;
			vector<cv::Point2f> train, query;

			for (size_t i = 0; i < nn_matches.size(); i++) {
				DMatch first = nn_matches[i][0];
				float dist1 = nn_matches[i][0].distance;
				float dist2 = nn_matches[i][1].distance;

				if (dist1 < nn_match_ratio * dist2) {
					matched1.push_back(keypoints[0][first.queryIdx]);
					matched2.push_back(keypoints[1][first.trainIdx]);
					train.push_back(keypoints[1][first.trainIdx].pt);
					query.push_back(keypoints[0][first.queryIdx].pt);
				}
			}

			Mat output_mask;
			Mat H;

			if (train.size() != 0 && query.size() != 0) {
				H = findHomography(query, train, CV_RANSAC, 3, output_mask);
			}
			
			vector<Point2f> matchingpoints1, matchingpoints2;
			if (H.type() == CV_64FC1 || H.type() == CV_64FC2 || H.type() == CV_32FC1 || H.type() == CV_32FC2) {

				for (unsigned i = 0; i < matched1.size(); i++) {
					Mat col = Mat::ones(3, 1, CV_64F);
					col.at<double>(0) = matched1[i].pt.x;
					col.at<double>(1) = matched1[i].pt.y;


					col = H * col;
					col /= col.at<double>(2);
					double dist = sqrt(pow(col.at<double>(0) - matched2[i].pt.x, 2) +
						pow(col.at<double>(1) - matched2[i].pt.y, 2));
					if (dist < inlier_threshold) {
						int new_i = static_cast<int>(inliers1.size());
						inliers1.push_back(matched1[i]);
						inliers2.push_back(matched2[i]);
						matchingpoints1.push_back(matched1[i].pt);
						matchingpoints2.push_back(matched2[i].pt);
						good_matches.push_back(DMatch(new_i, new_i, 0));
					}
				}

				/*---------------------------------------------FOR ADDING ON TO TOTALSCAN-----------------------------------------------------*/

				double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
				if (inlier_ratio > requiredratio) {
					vector<Point2f> distances = GetDistancesAngle(matchingpoints1, matchingpoints2);


					vector<double>angles;
					angles.push_back(distances[distances.size() - 1].x);

					totalscan = AddNextImage(imgs, distances, angles, totalscan);
					imgs.clear();
					distances.clear();
					angles.clear();
					//imshow("Total", totalscan);

					cout << "A-KAZE Matching Results" << endl;
					cout << "*******************************" << endl;
					cout << "# Keypoints 1:                        \t" << keypoints[0].size() << endl;
					cout << "# Keypoints 2:                        \t" << keypoints[1].size() << endl;
					cout << "# Matches:                            \t" << matched1.size() << endl;
					cout << "# Inliers:                            \t" << inliers1.size() << endl;
					cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
					cout << endl;

					matchingpoints1.clear();
					matchingpoints2.clear();
					keypoints[0] = keypoints[1];
					keypoints.pop_back();
					outputimages[0] = outputimages[1];
					outputimages.pop_back();
					matched1.clear();
					matched2.clear();
					inliers1.clear();
					inliers2.clear();
					/*--------------------------------------------------------------------------FOR CHUNKING---------------------------------------------------------------------------------------------*/
					Chunk(totalscan);


					count++;
					first = false;
					skip = false;
				}
				else {
					matchingpoints1.clear();
					matchingpoints2.clear();
					keypoints.clear();

					outputimages.clear();

					matched1.clear();
					matched2.clear();
					inliers1.clear();
					inliers2.clear();

					cout << "The size of the compared images is inside" << imgs.size() << endl;
					imgs[imgs.size() - 2] = imgs[imgs.size() - 1];
					//imshow("Image left", imgs[0]);
					imgs.pop_back();
					skip = true;
				}
				

			}
			else {
				if (first == true) {
					cout << "No matches found" << endl;
					previous.push_back(totalscan);
					string filename = "previmage";

					filename.append(".png");
					imwrite(filename, totalscan);
					if (filename == "") {
						frame = imread("C:/Users/swri11/Documents/GitHub/VLCImages/images.png");
					}
					else {
						cap >> frame;
						resize(frame, frame, frame.size() / 2, (0, 0), 1);
						rotate(frame, frame, ROTATE_90_CLOCKWISE);
						//rotate(frame, frame, ROTATE_180);
						imshow("FrameSkipped", frame);
					}

					Rect rect = Rect(0, 0, frame.cols, frame.rows / 2.5);
					frame = frame(rect);

					totalscan = frame;

					nn_match_ratio += 1.0;
					contrast += .2;
					brightness -= 5;
					keypoints.clear();
					outputimages.clear();

					cout << "The current match ratio is " << nn_match_ratio << endl;
					cout << "The increased contrast is " << contrast << " and decreased brightness is " << brightness;

					adjustoffsetx = 0;
					adjustoffsety = 0;

					tempwidth += 50;
					tempheight += 50;
					imgs.clear();
				}
				else {
					matchingpoints1.clear();
					matchingpoints2.clear();
					keypoints.clear();

					outputimages.clear();
					matched1.clear();
					matched2.clear();
					inliers1.clear();
					inliers2.clear();

					cout << "The size of the compared images is " << imgs.size() << endl;
					imgs[imgs.size() - 2] = imgs[imgs.size() - 1];
					imgs.pop_back();

					skip = true;
				}

				
			}
			char c = cvWaitKey(5);
			//if escape key is pressed
			if (c == 27) {
				break;
			}
		}
	}

	waitKey(0); // Wait for a keystroke in the window
	return 0;
}