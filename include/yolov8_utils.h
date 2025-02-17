#pragma once
#include<iostream>
#include <numeric>
#include<opencv2/opencv.hpp>
#include<io.h>

#define ORT_OLD_VISON 13  //ort1.12.0 之前的版本为旧版本API

struct PoseKeyPoint {
	float x = 0;
	float y = 0;
	float confidence = 0;
};

struct OutputParams {
	int id;             //结果类别id
	float confidence;   //结果置信度
	cv::Rect box;       //矩形框
	cv::RotatedRect rotatedBox;  //obb结果矩形框
	cv::Mat boxMask;       //矩形框内mask，节省内存空间和加快速度
	std::vector<PoseKeyPoint> keyPoints; //pose key points

};
struct MaskParams {
	//int segChannels = 32;
	//int segWidth = 160;
	//int segHeight = 160;
	int netWidth = 640;
	int netHeight = 640;
	float maskThreshold = 0.5;
	cv::Size srcImgShape;
	cv::Vec4d params;
};

struct PoseParams {
	float kptThreshold = 0.5;
	int kptRadius = 5;
	bool isDrawKptLine = true; //If True, the function will draw lines connecting keypoint for human pose.Default is True.
	cv::Scalar personColor = cv::Scalar(0, 0, 255);
	std::vector<std::vector<int>>skeleton = {
		{11,0},{0,1},{1,2},
		{2,3},{3,4},{4,5},
		{5,6},{6,7},{7,8},
		{8,9},{9,10},{10,11}
	};
	std::vector<cv::Scalar> posePalette =
	{
	cv::Scalar(255, 128, 0) ,
	cv::Scalar(255, 153, 51),
	cv::Scalar(255, 178, 102),
	cv::Scalar(230, 230, 0),
	cv::Scalar(255, 153, 255),
	cv::Scalar(153, 204, 255),
	cv::Scalar(255, 102, 255),
	cv::Scalar(255, 51, 255),
	cv::Scalar(102, 178, 255),
	cv::Scalar(51, 153, 255),
	cv::Scalar(255, 153, 153),
	cv::Scalar(255, 102, 102),
	cv::Scalar(255, 51, 51),
	cv::Scalar(153, 255, 153),
	cv::Scalar(102, 255, 102),
	cv::Scalar(51, 255, 51),
	cv::Scalar(0, 255, 0),
	cv::Scalar(0, 0, 255),
	cv::Scalar(255, 0, 0),
	cv::Scalar(255, 255, 255),
	};
	std::vector<int> limbColor = { 9, 9, 9,  10 ,10 ,10  , 11, 11, 11,  12, 12, 12};
	std::vector<int> kptColor = { 1, 1, 2,  3, 3, 4,  5, 5, 6,  7, 7, 8 };
	std::map<unsigned int, std::string> kptBodyNames{
		{0,"top_left"},		{1,	"top_right"},
		{2, "right_shoulder"},
		{3, "right_top"},   {4, "right_bottom"},
		{5, "right_hip"},
		{6, "bottom_right"}, {7, "bottom_left"},
		{8, "left_hip"},
		{9, "left_bottom"},  {10, "left_top"},
		{11, "left_shoulder"}
	
	};
};


std::vector<cv::Point> getEdgePointsFromMask(const OutputParams& output);
bool CheckModelPath(std::string modelPath);
bool CheckParams(int netHeight, int netWidth, const int* netStride, int strideSize);
void DrawPred(cv::Mat& img,
	std::vector<OutputParams> result,
	std::vector<std::string> classNames,
	std::vector<cv::Scalar> color,
	bool isVideo = false
);
void DrawPredPose(cv::Mat& img, std::vector<OutputParams> result, PoseParams& poseParams, bool isVideo = false);

void DrawRotatedBox(cv::Mat& srcImg, cv::RotatedRect box, cv::Scalar color, int thinkness);
void LetterBox(const cv::Mat& image, cv::Mat& outImage,
	cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
	const cv::Size& newShape = cv::Size(640, 640),
	bool autoShape = false,
	bool scaleFill = false,
	bool scaleUp = true,
	int stride = 32,
	const cv::Scalar& color = cv::Scalar(114, 114, 114));
void GetMask(const cv::Mat& maskProposals, const cv::Mat& maskProtos, std::vector<OutputParams>& output, const MaskParams& maskParams);
void GetMask2(const cv::Mat& maskProposals, const cv::Mat& maskProtos, OutputParams& output, const MaskParams& maskParams);
int BBox2Obb(float centerX, float centerY, float boxW, float boxH, float angle, cv::RotatedRect& rotatedRect);

