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

const float inlier_threshold = 20.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 100.8f;   // Nearest neighbor matching ratio

vector<Mat> imgs;
vector<Mat> previous;

vector < vector<Mat> > Grid;

Point2f runningdistance;
Point2f previousrunningdistance;
int xrowsfilled = 1;
int yrowsfilled = 1;

int xprevious = 0;
int yprevious = 0;

int tempwidth = 500;
int tempheight = 300;
int adjustoffsetx = 0;
int adjustoffsety = 0;


double previousangle = 0;

bool missed = false;
bool rotateThis = false;

int brightness = 0;
int contrast = 1;

float collectiveangle = 0;
string folder = "C:/Users/swri11/Documents/GitHub/ChunkedTerrain/";

using namespace cv;
using namespace std;

Mat RotateImage(Mat image, double angle) {
	if (image.rows > 0) {
		collectiveangle += angle;
		if (collectiveangle > 18 || collectiveangle < -18) {
			angle = -angle;
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

		float sin = RotationMatrix.at<uchar>(0, 0);
		cout << "This is the sin?" << sin << endl;
		float cos = RotationMatrix.at<uchar>(0, 1);
		cout << "This is the cos?" << cos << endl;

		float nW = int((h * sin) + (w * cos));
		float nH = int((h * cos) + (w * sin));

		RotationMatrix.at<uchar>(0, 2) += (nW / 2) - h;
		RotationMatrix.at<uchar>(1, 2) += (nH / 2) - w;

		warpAffine(image, image, RotationMatrix, image.size(), WARP_INVERSE_MAP, BORDER_TRANSPARENT);
		RotationMatrix.release();
		return image;
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

		//cout << "The distances are : " << distances[1].x << "x and " << distances[1].y << endl;
		//cout << "The running distances for the windows edges are : " << runningdistance.x << " " << runningdistance.y;
		if (xdistance< 0 && ydistance < 0) {
			Mat resize;
			Rect rect = Rect(0, 0, images[1].cols, images[1].rows / 2.5);
			images[1] = images[1](rect);
			if (rotateThis == true) {
				images[1] = RotateImage(images[1], angls[0]);
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
				images[1] = RotateImage(images[1], -angls[0]);
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
				images[1] = RotateImage(images[1], -angls[0]);
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
				images[1] = RotateImage(images[1], angls[0]);
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
	for (float x = xprevious; x <= total.cols / 312;) {
		for (float y = yprevious; y <= total.rows / 312;) {
			//if has expanded
			cout << "xprevious is :" << xprevious << "yprevious is :" << yprevious << endl;
			cout << "xfilled is: " << xrowsfilled << "yfilled is :" << yrowsfilled << endl;
			if (yprevious < yrowsfilled && xprevious < xrowsfilled && yprevious != total.rows / 312 && xprevious != total.cols / 312) {
				yprevious = yrowsfilled;
				xprevious = xrowsfilled;
				cout << "Broke on first" << endl;
				Mat temp;

				temp = total(Rect(total.cols - 312 * xprevious, total.rows - 312 * yprevious, 312, 312));
				stringstream ss;
				ss << folder << "image" << xprevious << " " << yprevious << ".png";
				string filename = ss.str();
				imwrite(filename, temp);
				//imshow(filename, temp);


				//check which direction they're moving

				//moving forward

				yrowsfilled++;

				xrowsfilled++;



				previousrunningdistance = runningdistance;
			}

			else if (yprevious < yrowsfilled && yprevious != total.rows / 312) {
				yprevious = yrowsfilled;
				cout << xprevious << " and y previous: " << yprevious << endl;
				//if y changed, only add y
				Mat temp;
				cout << "Made it here on the right one" << endl;
				cout << "What does this equal " << int(runningdistance.y) % 312 << endl;
				temp = total(Rect(total.cols - 312 * xprevious, total.rows - 312 * yprevious, 312, 312));
				stringstream ss;
				ss << folder << "image" << xprevious << " " << yprevious << ".png";
				string filename = ss.str();
				imwrite(filename, temp);
				//imshow(filename, temp);

				if (runningdistance.y < previousrunningdistance.y) {
					//moving forward
					yrowsfilled++;
				}
				else {
					yprevious--;
				}


				previousrunningdistance = runningdistance;
			}

			else if (xprevious < xrowsfilled && xprevious != total.cols / 312) {
				xprevious = xrowsfilled;
				//if y changed, only add y
				Mat temp;
				cout << "Made it here on x" << endl;
				cout << xprevious << " and y previous: " << yprevious << endl;
				temp = total(Rect(total.cols - 312 * xprevious, total.rows - 312 * yprevious, 312, 312));
				stringstream ss;
				ss << folder << "image" << xprevious << " " << yprevious << ".png";
				string filename = ss.str();
				imwrite(filename, temp);
				//imshow(filename, temp);

				if (runningdistance.x > previousrunningdistance.x) {
					xrowsfilled++;
				}
				else {
					xprevious--;
				}

				previousrunningdistance = runningdistance;
			}
			y++;
		}
		x++;

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
			Mat temp = totalscan(Rect(adjustoffsetx, adjustoffsety, tempwidth, tempheight));


			cvtColor(temp, temp, CV_BGR2YCrCb);

			vector<Mat> channelstemp;
			split(temp, channelstemp);

			//equalizeHist(channelstemp[0], channelstemp[0]);

			merge(channelstemp, temp);

			cvtColor(temp, temp, CV_YCrCb2BGR);
			temp.convertTo(temp, -1, contrast, brightness);

			imshow("Temp", temp);
			imgs.push_back(temp);
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




			vector< vector<KeyPoint>> keypoints;
			vector<Mat> outputimages;

			Ptr<AKAZE> akaze = AKAZE::create();
			for (int k = 0; k < 2; k++) {
				vector<KeyPoint> keypoint;
				Mat outputimage = Mat();
				akaze->detectAndCompute(imgs[k], noArray(), keypoint, outputimage);
				outputimages.push_back(outputimage);
				keypoints.push_back(keypoint);
			}

			//have to do this for like every two images i guess
			BFMatcher matcher(NORM_HAMMING);
			vector< vector<DMatch> > nn_matches;
			matcher.knnMatch(outputimages[0], outputimages[1], nn_matches, 2);

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
			Mat H = findHomography(query, train, CV_RANSAC, 3, output_mask);
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
				vector<Point2f> distances = GetDistancesAngle(matchingpoints1, matchingpoints2);
				matchingpoints1.clear();
				matchingpoints2.clear();
				
				vector<double>angles;
				angles.push_back(distances[distances.size()-1].x);

				totalscan = AddNextImage(imgs, distances, angles, totalscan);
				imgs.clear();
				distances.clear();
				angles.clear();
				//imshow("Total", totalscan);

				double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
				cout << "A-KAZE Matching Results" << endl;
				cout << "*******************************" << endl;
				cout << "# Keypoints 1:                        \t" << keypoints[0].size() << endl;
				cout << "# Keypoints 2:                        \t" << keypoints[1].size() << endl;
				cout << "# Matches:                            \t" << matched1.size() << endl;
				cout << "# Inliers:                            \t" << inliers1.size() << endl;
				cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
				cout << endl;

/*--------------------------------------------------------------------------FOR CHUNKING---------------------------------------------------------------------------------------------*/
				Chunk(totalscan);

			
				count++;

			}
			else {
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

				adjustoffsetx = 0;
				adjustoffsety = 0;
				imgs.clear();

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