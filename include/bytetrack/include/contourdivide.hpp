#ifndef _CONTOURDIVIDE_HPP
#define _CONTOURDIVIDE_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include <cmath>
#include <iostream>
#include <limits>
#include <algorithm>
#include <random> 


cv::Scalar generateRandomScalar() {  
    // 创建随机数生成器  
    std::random_device rd;  // 使用随机设备  
    std::mt19937 gen(rd()); // 使用梅尔森旋转算法生成随机数  
    std::uniform_int_distribution<> dis(0, 255); // 均匀分布，范围0-255  

    // 生成随机的BGR颜色  
    return cv::Scalar(dis(gen), dis(gen), dis(gen)); // OpenCV使用BGR格式  
}  

/// @brief  计算斜率
/// @param p1 点1
/// @param p2 点2
/// @return  斜率
double calculateSlope(const cv::Point &p1, const cv::Point &p2)
{
    if (p2.x == p1.x)
    {
        return std::numeric_limits<double>::infinity(); // 处理垂直线段
    }
    return static_cast<double>(p2.y - p1.y) / (p2.x - p1.x);
}
std::vector<cv::Point> mergeContours(const std::vector<std::vector<cv::Point>> &contours)
{
    std::vector<cv::Point> mergedContour;

    for (const auto &contour : contours)
    {
        mergedContour.insert(mergedContour.end(), contour.begin(), contour.end());
    }
    return mergedContour;
}

std::vector<std::vector<cv::Point>> kMeansClustering(const std::vector<cv::Point> &contour, size_t cnt)
{
    if (contour.size() < cnt)
    {
        throw std::invalid_argument("Contour points must be greater than the number of clusters.");
    }

    // 计算斜率与长度
    std::vector<double> slopes;
    std::vector<double> lenths;
    size_t n = contour.size();

    for (size_t i = 0; i < n; ++i)
    {
        cv::Point p1 = contour[i];
        cv::Point p2 = contour[(i + 1) % n]; // 确保是闭合的轮廓
        slopes.push_back(calculateSlope(p1, p2));
        lenths.push_back(cv::norm(p1-p2));
    }

    
   
    ///特征向量
    std::vector<std::vector<double>> eigenvectors;
    std::for_each(eigenvectors.begin(),eigenvectors.end(),[&](auto &ele){
        ele.push_back(contour[i].x);
        ele.push_back(contour[i].y);
        ele.push_back(slopes[i]);
        ele.push_back(contour[i].y-slopes[i]*contour[i].x);
        ele.push_back(floor(i/(contour.size()/cnt)));
        ele.push_back(lenths[i]);
    });
    std::vector<int> labels;
    cv::kmeans(eigenvectors, cnt, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1), 3, cv::KMEANS_RANDOM_CENTERS);

    // 输出聚类结果
    std::vector<std::vector<cv::Point>> clustersContours(cnt);
    for (size_t i = 0; i < contour.size(); ++i)
    {
        clustersContours[labels[i]].push_back(contour[i]);
    }
    return clustersContours;
} 
#endif