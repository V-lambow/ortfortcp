#ifndef __MYUTIL_HPP_
#define __MYUTIL_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric> 
#include <variant>
namespace myutil
{
    /// select_shape_std
    void maxAreaContour(const cv::Mat &binaryImage, cv::Mat &outputImage, std::vector<std::vector<cv::Point>> &maxContour);

    ///  @brief  计算角度
    ///  @param pulseVal 脉冲值
    ///  @param pStart 起始点
    ///  @param pEnd 终止点
    ///  @return 角度
    double caldegree(int pulseVal, cv::Point pStart, cv::Point pEnd);

    ///  @brief  圆形结构体
    ///  @param center 圆心
    ///  @param radius 半径
    struct Circled
    {
        cv::Point2f center;
        float radius;
    };

    myutil::Circled calCircled(const std::vector<cv::Point2f> &contour);


    double angleBetweenThreePoints(const cv::Point &a, const cv::Point &b, const cv::Point &c);

    void lineLenFilter(std::vector<cv::Vec4f>& lines,std::function<bool(float)>lenthreshold);


    double dist_P2Line(const cv::Point2f &p, const cv::Vec4f &line);


    std::vector<cv::Point2f> cvpt2cvptf(const std::vector<cv::Point> &pts);
    

    cv::Mat safeCrop(const cv::Mat& colorImage, cv::Rect croppedRect);

    std::string getCurrentDate();


}



#endif