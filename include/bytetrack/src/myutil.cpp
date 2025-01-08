
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

std::pair<double, cv::Point2f> myutil::dist_P2Circle(const cv::Point2f &p, const Circled &circle)
{
    double distance = std::abs(cv::norm(p - circle.center) - circle.radius);

    // 计算最近的圆上点
    cv::Point2f closestPoint;
    if (distance == 0)
    {
        closestPoint = p; // 点在圆上
    }
    else
    {
        cv::Vec2f direction = (p - circle.center) / cv::norm(p - circle.center); // 归一化的方向向量

        // 手动计算圆上的点
        closestPoint.x = circle.center.x + direction[0] * circle.radius; // x坐标
        closestPoint.y = circle.center.y + direction[1] * circle.radius; // y坐标
    }

    return std::make_pair(distance, closestPoint); // 返回距离和最近的点
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
    void myutil::lineLenFilter(std::vector<cv::Vec4f>& lines,std::function<bool(float)>lenthreshold){
        std::vector<cv::Vec4f> res;  

        for (auto line : lines) {  
            float x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];  
            auto len = cv::norm(cv::Point2f(x1, y1) - cv::Point2f(x2, y2));  
            if (lenthreshold(len)) {  
                res.push_back(line); // 添加符合条件的线段到结果中  
            }  
        }  

        lines = res; // 用新过滤过的线段替换原有的线段 vector  
    }

// 找到最优点  
cv::Point2f myutil::findOptimalPoint(const std::vector<cv::Vec4f>& lines, const cv::TermCriteria& criteria){  
    // 初始点位置（可以是所有线段的中点）  
    cv::Point2f optimalPoint(0, 0);  
    for (const auto& line : lines) {  
        optimalPoint += cv::Point2f((line[0] + line[2]) / 2, (line[1] + line[3]) / 2);  
    }  
    optimalPoint *= (1.0f / lines.size());  

      // 动态步长  
    float stepSize = 5.0f;  // 初始步长  
    float shrinkFactor = 0.5f; // 步长缩小因子  

    // 设置初始条件  
    double minDistance = totalDist(optimalPoint, lines);  
    cv::Point2f bestPoint = optimalPoint;  

    // 开始迭代  
    for (int count = 0; count < criteria.maxCount; count++) {  
        bool improved = false;  

        // 搜索周围邻域  
        for (float dx = -stepSize; dx <= stepSize; dx += stepSize) {  
            for (float dy = -stepSize; dy <= stepSize; dy += stepSize) {  
                // 排除中心点的情况  
                if (dx == 0 && dy == 0) continue;  

                cv::Point2f newPoint = optimalPoint + cv::Point2f(dx, dy);  
                double newDistance = totalDist(newPoint, lines);  

                // 如果新的点距离小于当前最优点，更新最优点  
                if (newDistance < minDistance) {  
                    minDistance = newDistance;  
                    bestPoint = newPoint;  
                    improved = true;  
                }  
            }  
        }  

        // 更新当前点  
        optimalPoint = bestPoint;  

        // 如果没有找到更优解，停止迭代  
        if (!improved) {  
            break; 
            std::cout << "没有找到更优解,当前迭代次数：" << count << "步长："<<stepSize<<std::endl; 
        }  

        // 如果达到精度条件，停止迭代  
        if (minDistance < criteria.epsilon) {  
            std::cout << "达到精度条件，当前迭代次数：" << count << "步长："<<stepSize<<std::endl; 
            break;  
        }  

        // 动态缩小步长，适应更精细的搜索  
        stepSize *= shrinkFactor;   
    }  
    std::cout << "迭代结束，当前步长："<<stepSize<<std::endl; 
    return bestPoint;  
}  

double myutil::totalDist(cv::Point2f  pt, const std::vector<cv::Vec4f>& lines)
{
    std::vector<double> distances;
    for (const auto& line : lines) 
    {
        double dictance = myutil::dist_P2Line(pt,line);
        distances.push_back(pow(dictance,2));
    }
    return sqrt(std::accumulate(distances.begin(), distances.end(), 0.0));
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
    double distance = std::abs(A * p.x + B * p.y - C) / std::sqrt(A * A + B * B);  
    return distance;  
}

std::vector<cv::Point2f> myutil::pt2SubpixPtf(cv::Mat src, std::vector<cv::Point> pts)
{
    std::vector<cv::Point2f> subpix;
    for (const auto &pt : pts)
    {
        subpix.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
    }
    try
    {
        cv::Mat src_mono;
        cv::cvtColor(src, src_mono, cv::COLOR_BGR2GRAY);
        src_mono.convertTo(src_mono, CV_8UC1);
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01);

        cv::cornerSubPix(src_mono, subpix, cv::Size(5, 5), cv::Size(-1, -1), criteria);
        return subpix;
    }
    catch (const cv::Exception &e)
    {
        std::cerr << e.what() << '\n';
        return subpix;
    }
}