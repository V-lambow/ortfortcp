#ifndef _LINESFIT_HPP_
#define _LINESFIT_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <variant>
#include <algorithm>
#include <cmath> 
#include <opencv2/ximgproc.hpp>
#include "myutil.h"



// #define EXPORT_API
// #ifdef EXPORT_API
// #define EXPORT_DLL __declspec(dllexport)
// #else
// #define EXPORT_DLL __declspec(dllimport)
// #endif

class FitLines
{
    using Linef = std::pair<cv::Point2f, cv::Point2f>;
    using Contours = std::vector<std::vector<cv::Point>>;

public:
    enum class fitMode
    {
        LEASTSQUARE = 0,
        PROSAC = 1,
        HOUGH = 2,
        HOUGHP = 3,
        LSD =4
    };

    // 构造函数
    FitLines(const Contours &contours, size_t cnt = 0)
        : m_contours(contours), m_cnt(cnt) {}
    FitLines(const std::vector<std::vector<cv::Point2f>> contours2f, size_t cnt = 0)
        : m_cnt(cnt)
    {
        m_contours = {};
        for (const auto &contourf : contours2f)
        {
            m_contours.emplace_back(myutil::cvptf2cvpt(contourf));
        }
    }
    // 析构函数
    ~FitLines() {}



    // 主拟合函数，返回拟合的结果
    std::variant<std::vector<Linef>, std::string> fit(
        fitMode mode = fitMode::LEASTSQUARE,
        int distType = cv::DIST_L2,
        double reps = 0.01,
        double aeps = 0.01)
    {
        // 轮廓对象为空
        if (m_contours.empty() || m_cnt <= 0)
        {
            return std::string("Contours are empty or count is zero");
        }
        // 初始化结果
        std::vector<Linef> res;

        for (const auto &contour : m_contours)
        {
            switch (mode)
            {
            case fitMode::LEASTSQUARE:
                res = fitLeastSquares(contour, distType, reps, aeps);
                break;

            case fitMode::PROSAC:
                res = fitProsac(contour, distType, reps, aeps);
                break;

            case fitMode::HOUGH:
                res = fitHough(contour);
                break;

            case fitMode::HOUGHP:
                res = fitHoughP(contour);
                break;

            case fitMode ::LSD:
                // res = fitLSD(contour);

            default:
                return std::string("Error: unknown fitting mode");
            }
            // 输出拟合结果
            // for (const auto& line : res)
            // {
            //     std::cout << "Line: (" << line.first.x << ", " << line.first.y << ") -> ("
            //               << line.second.x << ", " << line.second.y << ")\n";
            // }
        }

        return res; // 返回拟合结果
    }

private:
    Contours m_contours; // 保存多个轮廓
    size_t m_cnt;        // 需要拟合的直线数量

    // 最小二乘法拟合函数
    std::vector<Linef> fitLeastSquares(const std::vector<cv::Point> &contour, int distType, double reps, double aeps)
    {
        std::vector<Linef> res;
        std::vector<cv::Vec4f> lines;

        std::cout << "Fitting lines using least squares...\n";
        cv::fitLine(contour, lines, distType, 0, reps, aeps);

        for (const auto &line : lines)
        {
            float vx = line[0], vy = line[1], x0 = line[2], y0 = line[3];
            Linef lineTmp{
                cv::Point2f(x0 - 1000 * vx, y0 - 1000 * vy),
                cv::Point2f(x0 + 1000 * vx, y0 + 1000 * vy)};
            res.push_back(lineTmp);
        }   
        // cv::Ptr<ximgproc::EdgeDrawing> ed = ximgproc::createEdgeDrawing();
        // ed->params.EdgeDetectionOperator = ximgproc::EdgeDrawing::LSD;
        // ed->detectEdges(mat);
        // ed->detectLines();
        return res;
    }

    // PROSAC 拟合函数
    std::vector<Linef> fitProsac(const std::vector<cv::Point> &contour, int distType, double reps, double aeps)
    {
        std::cout << "Fitting lines using PROSAC...\n";
        std::vector<bool> pointUsed(contour.size(), false);
        std::vector<Linef> res;

        for (size_t i = 0; i < contour.size(); i++)
        {
            std::vector<cv::Point> sample;

            // 随机选择样本点
            while (sample.size() < 10)
            {
                size_t index = rand() % contour.size();
                if (!pointUsed[index])
                {
                    sample.push_back(contour[index]);
                    pointUsed[index] = true; // 标记点为已用
                }
            }
        
            
            cv::Vec4f fittedLine;
            cv::fitLine(sample, fittedLine, distType, 0, reps, aeps);
            Linef line;
            float vx = fittedLine[0], vy = fittedLine[1], x0 = fittedLine[2], y0 = fittedLine[3];
            line.first = cv::Point2f((x0 - 1000 * vx), (y0 - 1000 * vy));
            line.second = cv::Point2f((x0 + 1000 * vx), (y0 + 1000 * vy));

            res.push_back(line);

            // 标记内点为已用
            for (const auto &inlier : sample)
            {
                auto it = std::find(contour.begin(), contour.end(), inlier);
                if (it != contour.end())
                {
                    pointUsed[std::distance(contour.begin(), it)] = true;
                }
            }

            // 如果达到所需数量的直线，退出循环
            if (res.size() >= m_cnt)
            {
                break;
            }
        }

        return res;
    }

    // Hough 变换拟合函数
    std::vector<Linef> fitHough(const std::vector<cv::Point> &contour)
    {
        std::vector<Linef> res;
        std::vector<cv::Vec2f> linesVec;

        std::cout << "Fitting lines using Hough Transform...\n";
        auto mat = points2Mat(contour);                       // 转换为矩阵
        cv::HoughLines(mat, linesVec, 1, CV_PI / 180, m_cnt); // Hough 变换

        for (const auto &line : linesVec)
        {
            float rho = line[0];   // 距离
            float theta = line[1]; // 角度
            cv::Point2f pt1(rho * cos(theta), rho * sin(theta));
            cv::Point2f pt2(pt1.x + 1000 * (-sin(theta)), pt1.y + 1000 * cos(theta));
            res.emplace_back(pt1, pt2);
        }

        return res;
    }

    std::vector<Linef> fitHoughP(const std::vector<cv::Point> &contour){
        std::vector<Linef> res;
        std::vector<cv::Vec4i> linesVec;
        std::cout << "Fitting lines using HoughP Transform...\n";
        auto mat = points2Mat(contour);                       // 转换为矩阵
        cv::HoughLinesP(mat, linesVec, 1, CV_PI / 180, m_cnt); // Hough 变换
        for (const auto &line : linesVec)
        {
            float x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
            res.emplace_back(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
        }
        return res;
    }
    cv::Mat points2Mat(const std::vector<cv::Point> &points)
    {
        cv::Mat mat(points.size(), 1, CV_32FC2);
        for (size_t i = 0; i < points.size(); i++)
        {
            mat.at<cv::Vec2f>(i, 0) = cv::Vec2f(static_cast<float>(points[i].x), static_cast<float>(points[i].y));
        }
        return mat;
    }

    inline static cv::Scalar generateRandomScalar()
    {
        // 创建随机数生成器
        std::random_device rd;                       // 使用随机设备
        std::mt19937 gen(rd());                      // 使用梅尔森旋转算法生成随机数
        std::uniform_int_distribution<> dis(0, 255); // 均匀分布，范围0-255

        // 生成随机的BGR颜色
        return cv::Scalar(dis(gen), dis(gen), dis(gen)); // OpenCV使用BGR格式
    }
    // 将点转换为矩阵的辅助函数

public:
    /// @brief  计算斜率
    /// @param p1 点1
    /// @param p2 点2
    /// @return  斜率
    inline static double calculateSlope(const cv::Point &p1, const cv::Point &p2)
    {
        if (p2.x == p1.x)
        {
            return std::numeric_limits<double>::infinity(); // 处理垂直线段
        }
        return static_cast<double>(p2.y - p1.y) / (p2.x - p1.x);
    }
    inline static std::vector<cv::Point> mergeContours(const std::vector<std::vector<cv::Point>> &contours)
    {
        std::vector<cv::Point> mergedContour;

        for (const auto &contour : contours)
        {
            mergedContour.insert(mergedContour.end(), contour.begin(), contour.end());
        }

        return mergedContour;
    }

    inline static std::vector<std::vector<cv::Point>> kMeansClustering(const std::vector<cv::Point> &contour, size_t cnt)
    {
        if (contour.size() < cnt)
        {
            throw std::invalid_argument("Contour points must be greater than the number of clusters.");
        }

        // 计算斜率
        std::vector<double> slopes;
        size_t n = contour.size();

        for (size_t i = 0; i < n; ++i)
        {
            cv::Point p1 = contour[i];
            cv::Point p2 = contour[(i + 1) % n]; // 确保是闭合的轮廓
            slopes.push_back(calculateSlope(p1, p2));
        }
        /// 特征向量
        std::vector<std::vector<double>> eigenvectors{contour.size()};
        int i = 0;
        std::for_each(eigenvectors.begin(), eigenvectors.end(), [&](auto &ele)
                      {
        ele.push_back(contour[i].x);
        ele.push_back(contour[i].y);
        ele.push_back(slopes[i]);
        ele.push_back(contour[i].y-slopes[i]*contour[i].x);
        ele.push_back(floor(i/(contour.size()/cnt)));
        i++; });
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
};

#endif // _LINESFIT_HPP_