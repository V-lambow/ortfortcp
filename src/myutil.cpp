#include "myutil.h"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <ctime>

void myutil::maxAreaContour(const cv::Mat &binaryImage, cv::Mat &outputImage, std::vector<std::vector<cv::Point>> &maxContour)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        outputImage = cv::Mat::zeros(binaryImage.size(), CV_8UC1);
        return;
    }

    auto maxIt = std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point> &a, const std::vector<cv::Point> &b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });

    maxContour.clear();
    maxContour.push_back(*maxIt);

    outputImage = cv::Mat::zeros(binaryImage.size(), CV_8UC1);
    cv::drawContours(outputImage, maxContour, -1, cv::Scalar(255), cv::FILLED);
}

cv::Mat myutil::safeCrop(const cv::Mat &colorImage, cv::Rect croppedRect)
{
    cv::Rect safeRect(
        std::max(0, croppedRect.x),
        std::max(0, croppedRect.y),
        std::min(croppedRect.width, colorImage.cols - croppedRect.x),
        std::min(croppedRect.height, colorImage.rows - croppedRect.y)
    );

    safeRect.width = std::min(safeRect.width, colorImage.cols - safeRect.x);
    safeRect.height = std::min(safeRect.height, colorImage.rows - safeRect.y);

    return colorImage(safeRect).clone();
}

std::string myutil::getCurrentDate()
{
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d");
    return ss.str();
}
