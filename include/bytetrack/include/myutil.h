#ifndef __MYUTIL_HPP_
#define __MYUTIL_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric> 
namespace myutil
{
    /// select_shape_std
    void maxAreaContour(const cv::Mat &binaryImage, cv::Mat &outputImage, std::vector<std::vector<cv::Point>> &maxContour);

    /// calibratione degree
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
    /// 计算点到圆的距离
    std::pair<double, cv::Point2f> dist_P2Circle(const cv::Point2f &p, const Circled &circle);

    double angleBetweenThreePoints(const cv::Point &a, const cv::Point &b, const cv::Point &c);

    void lineLenFilter(std::vector<cv::Vec4f>& lines,std::function<bool(float)>lenthreshold);

    cv::Point2f findOptimalPoint(const std::vector<cv::Vec4f>& lines, const cv::TermCriteria& criteria);

    double totalDist(cv::Point2f  pt, const std::vector<cv::Vec4f>& lines);  

    double dist_P2Line(const cv::Point2f &p, const cv::Vec4f &line);

    std::vector<cv::Point2f> pt2SubpixPtf(cv::Mat src, std::vector<cv::Point> pts);

}



#endif