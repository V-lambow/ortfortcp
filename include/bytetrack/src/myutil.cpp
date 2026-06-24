
#include "myutil.h"

void myutil::maxAreaContour(const cv::Mat &binaryImage, cv::Mat &outputImage, std::vector<std::vector<cv::Point>> &maxContour)
{
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double maxArea = 0;
    maxContour.resize(1);

    // 遍历所有轮廓并找到最大面积的轮廓
    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);
        if (area > maxArea)
        {
            maxArea = area;
            maxContour[0] = contour;
        }
    }

    // 如果找到了最大轮廓，绘制到输出图像
    if (!maxContour.empty())
    {
        outputImage = cv::Mat::zeros(binaryImage.size(), CV_8UC1);
        cv::drawContours(outputImage, maxContour, -1, cv::Scalar(255), cv::FILLED);
    }
}

double myutil::caldegree(int pulseVal, cv::Point pStart, cv::Point pEnd)
{

    int dy = pEnd.y - pStart.y;
    int dx = pEnd.x - pStart.x;
    double resolution = pulseVal / sqrt(pow(dx, 2) + pow(dy, 2));
    assert(abs(pulseVal) - abs(resolution * dx) > 0);
    return sqrt((pulseVal + resolution * dx) * (pulseVal - resolution * dx)) / resolution * dy;
}

myutil::Circled myutil::calCircled(const std::vector<cv::Point2f> &contour)
{

    // 确保有足够的点来计算
    if (contour.size() < 3)
    {
        throw std::invalid_argument("至少需要三个点来计算一个圆");
    }

    // 生成设计矩阵和目标向量
    cv::Mat A(contour.size(), 3, CV_32F);
    cv::Mat b(contour.size(), 1, CV_32F);

    for (size_t i = 0; i < contour.size(); ++i)
    {
        float x = static_cast<float>(contour[i].x);
        float y = static_cast<float>(contour[i].y);
        A.at<float>(i, 0) = x;
        A.at<float>(i, 1) = y;
        A.at<float>(i, 2) = 1.0f;
        b.at<float>(i, 0) = (x * x + y * y);
    }

    // 最小二乘法求解  Ax = b
    cv::Mat x;
    cv::solve(A, b, x, cv::DECOMP_NORMAL);

    // 提取圆心和半径
    float a = -x.at<float>(0, 0) / 2;
    float b_center = -x.at<float>(1, 0) / 2;
    float radius = std::sqrt(a * a + b_center * b_center - x.at<float>(2, 0));

    return Circled{cv::Point2f(a, b_center), radius};
}

// 计算由三点 A、B、C 形成的角度 ∠ABC
double myutil::angleBetweenThreePoints(const cv::Point &a, const cv::Point &b, const cv::Point &c)
{
    double ab = cv::norm(a - b); // AB 的长度
    double bc = cv::norm(b - c); // BC 的长度
    double ac = cv::norm(a - c); // AC 的长度

    // 使用余弦定理计算角度
    double cosAngle = (std::pow(ab, 2) + std::pow(bc, 2) - std::pow(ac, 2)) / (2 * ab * bc);

    // 限制 cosAngle 的范围 [-1, 1]，以避免舍入误差
    if (cosAngle < -1.0)
        cosAngle = -1.0;
    if (cosAngle > 1.0)
        cosAngle = 1.0;

    // 返回角度（弧度转为度）
    return std::acos(cosAngle) * (180.0 / CV_PI);
}

void myutil::lineLenFilter(std::vector<cv::Vec4f> &lines, std::function<bool(float)> lenthreshold)
{
    std::vector<cv::Vec4f> res;

    for (auto line : lines)
    {
        float x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        auto len = cv::norm(cv::Point2f(x1, y1) - cv::Point2f(x2, y2));
        if (lenthreshold(len))
        {
            res.push_back(line); // 添加符合条件的线段到结果中
        }
    }

    lines = res; // 用新过滤过的线段替换原有的线段 vector
}


double myutil::dist_P2Line(const cv::Point2f &p, const cv::Vec4f &line){
   // 直线的两个点  
    cv::Point2f p1(line[0], line[1]); // 线的起点  
    cv::Point2f p2(line[2], line[3]); // 线的终点  

    // 计算直线的斜率和截距  
    double A = p2.y - p1.y; // dy  
    double B = p1.x - p2.x; // -dx  
    double C = A * p1.x + B * p1.y; // Ax + By = C  

    // 使用垂直距离的公式  
    return  std::abs(A * p.x + B * p.y - C) / std::sqrt(A * A + B * B);  

}

std::vector<cv::Point2f> myutil::cvpt2cvptf(const std::vector<cv::Point> &pts)
{
    std::vector<cv::Point2f> points2f;

    for (auto &point : pts)
    {
        points2f.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
    }
    return points2f;
}

//cv:mat croppedImg = src(cv::Rect()); 安全版本
cv::Mat myutil::safeCrop(const cv::Mat& colorImage, cv::Rect croppedRect) {

    // 调整Rect的宽高为正数
    if (croppedRect.width < 0) {
        croppedRect.x += croppedRect.width;
        croppedRect.width = -croppedRect.width;
    }
    if (croppedRect.height < 0) {
        croppedRect.y += croppedRect.height;
        croppedRect.height = -croppedRect.height;
    }

    // 创建目标图像，初始化为全黑
    cv::Mat croppedImg(croppedRect.height, croppedRect.width, colorImage.type(), cv::Scalar(0, 0, 0));

    // 原图的边界
    cv::Rect imageRect(0, 0, colorImage.cols, colorImage.rows);
    // 计算实际相交区域
    cv::Rect actualRect = croppedRect & imageRect;

    if (actualRect.area() > 0) {
        // 计算目标图像中的偏移量
        int xOffset = actualRect.x - croppedRect.x;
        int yOffset = actualRect.y - croppedRect.y;

        // 目标图像中的复制区域
        cv::Rect destRect(xOffset, yOffset, actualRect.width, actualRect.height);

        // 确保目标区域在范围内
        if (destRect.x >= 0 && destRect.y >= 0 &&
            destRect.x + destRect.width <= croppedImg.cols &&
            destRect.y + destRect.height <= croppedImg.rows) 
        {
            // 复制有效区域到目标图像
            colorImage(actualRect).copyTo(croppedImg(destRect));
        }
    }

    return croppedImg;
}

// YYYY-MM-DD
std::string myutil::getCurrentDate() {  

    auto now = std::chrono::system_clock::now();  
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);  
    std::tm now_tm = *std::localtime(&now_time_t);  
    
    std::ostringstream oss;  
    oss << std::put_time(&now_tm, "%Y-%m-%d"); // 格式化为 YYYY-MM-DD  
    return oss.str();  
}  

