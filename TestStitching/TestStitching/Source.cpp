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

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

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

string folder = "C:/Users/swri11/Documents/GitHub/ChunkedTerrain/";

int main(int, char* argv[])
{

	//const string filename = argv[1];

	//VideoCapture cap = VideoCapture("http://10.202.17.41:8080/shot.jpg"); 
	// Check VideoCapture documentation.
	//if (!cap.isOpened())  // check if we succeeded
		//return -1;

	Mat frame;
	Mat totalscan;

	frame = imread("C:/Users/swri11/Documents/GitHub/VLCImages/images.png");
	
	//cap.read(frame);
	//if (frame.rows < 312) {
		//cout << "Resolution is too low for chunking" << endl;
		//return -1;
	//}
	imshow("Frame", frame);
	//resize(frame, frame, frame.size() / 2, (0, 0), 1);
	Rect rect = Rect(0, 0, frame.cols, frame.rows/2.5);
	frame = frame(rect);
	//rotate(frame, frame, ROTATE_90_CLOCKWISE);
	totalscan = frame;

	imshow("Camera_Output", totalscan);

	previousrunningdistance = Point2f(0, 0);

	int count = 0;
	int fillimage = 0;
	while (true) {
		if (count < 0) {
			//cap >> frame;
			count++;
		}
		else {
			if (tempheight > totalscan.rows) {
				tempheight = totalscan.rows;
			}
			if (tempwidth > totalscan.cols) {
				tempwidth = totalscan.cols;
			}
			/*if (previous.size() == 2) {
				imgs.push_back(previous[0]);
				imgs.push_back(previous[1]);
				previous.clear();
				cout << "Trying to link two previous up" << endl;
			}*/
				Mat temp = totalscan(Rect(0, 0, tempwidth, tempheight));
				imshow("Temp", temp);
				imgs.push_back(temp);
				frame = imread("C:/Users/swri11/Documents/GitHub/VLCImages/images.png");
				//cap = VideoCapture("http://10.202.17.41:8080/shot.jpg");
				//cap.read(frame);
				//if (!frame.empty()) {
					//resize(frame, frame, frame.size() / 2, (0, 0), 1);
					//rotate(frame, frame, ROTATE_90_CLOCKWISE);
					//Create image frames from capture
					imgs.push_back(frame);
					imshow("Frame", frame);
				//}




			vector< vector<KeyPoint>> keypoints;
			vector<Mat> outputimages;

			Ptr<AKAZE> akaze = AKAZE::create();
			for (int k = 0; k < imgs.size(); k++) {
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

			if (query.size() != 0 || train.size() != 0) {
				Mat output_mask;
				Mat H = findHomography(query, train, CV_RANSAC, 3, output_mask);
				vector<Point2f> matchingpoints1, matchingpoints2;

				for (unsigned i = 0; i < matched1.size(); i++) {
					Mat col = Mat::ones(3, 1, CV_64F);
					col.at<double>(0) = matched1[i].pt.x;
					col.at<double>(1) = matched1[i].pt.y;

					if (H.type() == CV_64FC1 || H.type() == CV_64FC2 || H.type() == CV_32FC1 || H.type() == CV_32FC2) {
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
					else {
						cout << "There was an invalid homography" << endl;
					}

				}
				vector<Point2f> distances;
				for (int x = 0; x < matchingpoints1.size(); x++) {
					Point2f distance;
					distance.x = matchingpoints1[x].x - matchingpoints2[x].x;
					distance.y = matchingpoints1[x].y - matchingpoints2[x].y;
					distances.push_back(distance);
				}


				if (distances.size() > 0) {

					//average distances
					float xdistance;
					float ydistance;
					for (int x = 0; x < distances.size(); x++) {
						xdistance += distances[x].x;
						ydistance += distances[x].y;
					}

					xdistance /= distances.size();
					ydistance /= distances.size();

					runningdistance += Point2f(xdistance, ydistance);


					cout << "The distances are : " << distances[1].x << "x and " << distances[1].y << endl;
					cout << "The running distances for the windows edges are : " << runningdistance.x << " " << runningdistance.y;
					if (xdistance < 0 && ydistance < 0) {
						Mat resize;
						Rect rect = Rect(0, 0, imgs[1].cols, imgs[1].rows/2.5);
						imgs[1] = imgs[1](rect);
						copyMakeBorder(totalscan, resize, abs(ydistance), 0, abs(xdistance), 0, 0);
						imgs[1].copyTo(resize(Rect(0, 0, imgs[1].cols, imgs[1].rows)));
						//imshow("Cropped", resize);

						Mat res;

						drawKeypoints(imgs[0], keypoints[0], imgs[0]);
						drawKeypoints(imgs[1], keypoints[1], imgs[1]);
						drawMatches(totalscan, inliers1, imgs[1], inliers2, good_matches, res);
						imwrite("res.png", resize);

						//imshow("Res", res);
						totalscan = resize;
						imgs.clear();
					}
					if (xdistance > 0 && ydistance < 0) {
						Mat resize;
						Rect rect = Rect(0, 0, imgs[1].cols, imgs[1].rows/2.5);
						imgs[1] = imgs[1](rect);
						copyMakeBorder(totalscan, resize, abs(ydistance), 0, 0, abs(xdistance), 0);
						imgs[1].copyTo(resize(Rect(0, 0, imgs[1].cols, imgs[1].rows)));

						//imshow("Cropped", resize);

						Mat res;

						drawKeypoints(imgs[0], keypoints[0], imgs[0]);
						drawKeypoints(imgs[1], keypoints[1], imgs[1]);
						drawMatches(totalscan, inliers1, imgs[1], inliers2, good_matches, res);
						imwrite("res.png", resize);

						//imshow("Res", res);
						totalscan = resize;
						imgs.clear();
					}
					if (xdistance > 0 && ydistance > 0) {
						Mat resize;
						Rect rect = Rect(0, 0, imgs[1].cols, imgs[1].rows/2.5);
						imgs[1] = imgs[1](rect);
						copyMakeBorder(totalscan, resize, 0, abs(ydistance), 0, abs(xdistance), 0);
						imgs[1].copyTo(resize(Rect(0, 0, imgs[1].cols, imgs[1].rows)));
						//imshow("Cropped", resize);

						Mat res;

						drawKeypoints(imgs[0], keypoints[0], imgs[0]);
						drawKeypoints(imgs[1], keypoints[1], imgs[1]);
						drawMatches(totalscan, inliers1, imgs[1], inliers2, good_matches, res);
						imwrite("res.png", resize);

						//imshow("Res", res);
						totalscan = resize;
						imgs.clear();
					}
					if (xdistance < 0 && ydistance > 0) {
						Mat resize;
						Rect rect = Rect(0, 0, imgs[1].cols, imgs[1].rows/2.5);
						imgs[1] = imgs[1](rect);
						copyMakeBorder(totalscan, resize, 0, abs(ydistance), abs(xdistance), 0, 0);
						imgs[1].copyTo(resize(Rect(0, 0, imgs[1].cols, imgs[1].rows)));
						//imshow("Cropped", resize);

						Mat res;
						drawKeypoints(imgs[0], keypoints[0], imgs[0]);
						drawKeypoints(imgs[1], keypoints[1], imgs[1]);
						drawMatches(totalscan, inliers1, imgs[1], inliers2, good_matches, res);
						imwrite("res.png", resize);

						//imshow("Res", res);
						totalscan = resize;
						imgs.clear();
					}

					double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
					cout << "A-KAZE Matching Results" << endl;
					cout << "*******************************" << endl;
					cout << "# Keypoints 1:                        \t" << keypoints[0].size() << endl;
					cout << "# Keypoints 2:                        \t" << keypoints[1].size() << endl;
					cout << "# Matches:                            \t" << matched1.size() << endl;
					cout << "# Inliers:                            \t" << inliers1.size() << endl;
					cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
					cout << endl;
				}
				else {
					cout << "We make it to the skip" << endl;
					previous.push_back(totalscan);
					string filename = "previmage";
					for (int i = 0; i < previous.size(); i++) {
						filename.append("0");
					}
					filename.append(".png");
					imwrite(filename, totalscan);
					Rect rect = Rect(0, 0, imgs[1].cols, imgs[1].rows/2.5);
					frame = frame(rect);
					totalscan = frame;
					imgs.clear();
				}

				runningdistance.x = roundf(runningdistance.x);
				runningdistance.y = roundf(runningdistance.y);

				cout << "The rounded distance is :" << runningdistance.x << " " << runningdistance.y << endl;
				cout << totalscan.cols / 312 << " and rows " << totalscan.rows / 312 << endl;
				//if x is filled out


				for (float x = xprevious; x <= totalscan.cols / 312;) {
					for (float y = yprevious; y <= totalscan.rows / 312;) {
						//if has expanded
						cout << "xprevious is :" << xprevious << "yprevious is :" << yprevious << endl;
						cout << "xfilled is: " << xrowsfilled << "yfilled is :" << yrowsfilled << endl;
						if (yprevious < yrowsfilled && xprevious < xrowsfilled && yprevious != totalscan.rows / 312 && xprevious != totalscan.cols / 312) {
							yprevious = yrowsfilled;
							xprevious = xrowsfilled;
							cout << "Broke on first" << endl;
							Mat temp;

							temp = totalscan(Rect(totalscan.cols - 312 * xprevious, totalscan.rows - 312 * yprevious, 312, 312));
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

						else if (yprevious < yrowsfilled && yprevious != totalscan.rows / 312) {
							yprevious = yrowsfilled;
							cout << xprevious << " and y previous: " << yprevious << endl;
							//if y changed, only add y
							Mat temp;
							cout << "Made it here on the right one" << endl;
							cout << "What does this equal " << int(runningdistance.y) % 312 << endl;
							temp = totalscan(Rect(totalscan.cols - 312 * xprevious, totalscan.rows - 312 * yprevious, 312, 312));
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

						else if (xprevious < xrowsfilled && xprevious != totalscan.cols / 312) {
							xprevious = xrowsfilled;
							//if y changed, only add y
							Mat temp;
							cout << "Made it here on x" << endl;
							cout << xprevious << " and y previous: " << yprevious << endl;
							temp = totalscan(Rect(totalscan.cols - 312 * xprevious, totalscan.rows - 312 * yprevious, 312, 312));
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
			else {
				if (count == 0) {
					//attempt to restart the whole thing
				}
				tempheight += 50;
				tempwidth += 50;
			}
		}
		char c = cvWaitKey(30);
		//if escape key is pressed
		if (c == 27) {
			break;
		}

	}
	



    waitKey(0); // Wait for a keystroke in the window
    return 0;
}