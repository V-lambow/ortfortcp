#ifndef CENTERSEARCH_HPP
#define CENTERSEARCH_HPP

#include <iostream>
#include "linesfit.hpp"
#include "myutil.h"

class  CenterSearch
{

public:
    struct LinesFitParams
    {
        std::vector<std::vector<cv::Point2f>> contours{};
        /// @brief         LEASTSQUARE = 0,
        ///                PROSAC = 1,
        ///                HOUGH = 2
        FitLines::fitMode lineMode = FitLines::fitMode::LEASTSQUARE;
        size_t cnt = 0;
        int distType = cv::DIST_L2;
        double reps = 0.01f;
        double aeps = 0.01f;
        cv::Point2f p0 = cv::Point2f(0, 0);
        size_t maxIter = 1000;
        double learningRate = 0.1f;
    };
    enum class CenterMode
    {
        PBASE = 0,
        LBASE = 1,
        CBASE = 2
    };
    CenterSearch(std::vector<std::vector<cv::Point2f>> contours = {}, CenterMode mode = CenterMode::PBASE, size_t cnt = 0) : m_mode(mode)

    {
        m_linesFitParams.contours = contours;
        m_linesFitParams.cnt = cnt;
    }
    void setParam(LinesFitParams p)
    {
        this->m_linesFitParams = p;
    }

    /// @brief  点到直线的距离
    /// @param pt cv::Point2f
    /// @param l 直线std::pair<cv::Point2f, cv::Point2f>
    /// @return 点到直线的距离
    float dist_pl(const cv::Point2f &pt,
                  const std::pair<cv::Point2f, cv::Point2f> &l)
    {
        float lineLength = cv::norm(l.first - l.second);

        if (lineLength == 0.0f)
        {
            return cv::norm(pt - l.first); // 线段长度为0，返回到起点的距离
        }
        // 计算投影比例t
        float t = ((pt.x - l.first.x) * (l.second.x - l.first.x) +
                   (pt.y - l.first.y) * (l.second.y - l.first.y)) /
                  (lineLength * lineLength);
        t = std::clamp(t, 0.0f, 1.0f); // 限制t在[0, 1]范围内

        cv::Point2f projection = l.first + t * (l.second - l.first);
        return cv::norm(pt - projection);
    }

    /// @brief
    /// @param lines 直线族
    /// @param point 初始点
    /// @param maxIter 最大迭代次数
    /// @param learningRate 学习率
    /// @return  中点
    cv::Point2f centerWithMinNorm(std::vector<std::pair<cv::Point2f, cv::Point2f>> &lines,
                                  cv::Point2f point,
                                  int maxIter = 1000,
                                  float learningRate = 0.1f)
    {
        for (int i = 0; i < maxIter; ++i)
        {
            cv::Point2f adjustment(0.0f, 0.0f);
            float totalDistance = 0.0f;

            // 计算当前点到每条线段的距离并累加
            for (const auto &line : lines)
            {
                float dist = dist_pl(point, line);
                totalDistance += dist * dist; // 计算平方和
            }

            // 计算每条线段对调整点位置的影響
            for (const auto &line : lines)
            {
                float dist = dist_pl(point, line);
                if (dist > 0)
                {
                    cv::Point2f lineVec = line.second - line.first;           // 线段的方向向量
                    cv::Point2f normVec = cv::Point2f(-lineVec.y, lineVec.x); // 法向量

                    // 计算当前点到线段的单位法向量
                    cv::Point2f unitNormal = normVec / cv::norm(normVec); // 法向量标准化

                    // 通过当前距离的平方，调整点的位置
                    adjustment += unitNormal * (dist); // 调整量与距离成正比
                }
            }

            // 更新点的位置
            point -= learningRate * adjustment; // 根据计算的总调整进行更新

            // 迭代输出进度（可选）
            // if (i % 10 == 0) { // 每10次迭代输出一次进度
            //     std::cout << "Iteration " << i << ": Total Distance Norm = " << totalDistance << std::endl;
            // }
        }
        std::cout << "center Pt" << point << "\n";
        return point;
    }

    std::variant<cv::Point2f, std::string> search_mono(std::vector<std::vector<cv::Point>> contours,
                                                  CenterMode mode = CenterMode::PBASE,
                                                  size_t cnt = 0)
                                                  {
                                                    std::vector<std::vector<cv::Point2f>> contours2f;
                                                    std::vector<cv::Point2f> contour2f;
                                                    for (auto &c : contours)
                                                    {
                                                        contour2f = myutil::cvpt2cvptf(c);
                                                        contours2f.push_back(contour2f);
                                                    }
                                                    return search_mono(contours2f, mode, cnt);

                                                  }




    std::variant<cv::Point2f, std::string> search_mono(std::vector<std::vector<cv::Point2f>> contours2f,
                                                  CenterMode mode = CenterMode::PBASE,
                                                  size_t cnt = 0)
    {
        /// 需要合并
            std::vector<cv::Point2f> contour2f;
            for (auto &c : contours2f)
            {
                contour2f.insert(contour2f.end(), c.begin(), c.end());
            }
            auto centerRes =contourCenter(contour2f);
            ///失败
            if(centerRes.index()){
                return std::get<std::string>(centerRes);
            }
            else{
                return std::get<cv::Point2f>(centerRes);
            }
    }

    std::variant<std::vector<cv::Point2f>,std::string> search_mul(std::vector<std::vector<cv::Point2f>> contours2f,
                                                  CenterMode mode = CenterMode::PBASE,
                                                  size_t cnt = 0)
    {
        std::vector<cv::Point2f> res;
        for (const auto &c : contours2f)
        {
            auto centerRes = contourCenter(c);
            /// 失败
            if (centerRes.index())
            {
                return std::get<std::string>(centerRes);
            }
            else
            {
                res.push_back(std::get<cv::Point2f>(centerRes));
            }
        }
        return res;
    }




    



    std::variant<cv::Point2f, std::string> contourCenter(const std::vector<cv::Point2f> &contour2f)
    {
        cv::Point2f res;
        switch (m_mode)
        {
        /// 质心
        case CenterMode::PBASE:
        {

            cv::Moments m = cv::moments(contour2f);
            /// 计算质心
            if (m.m00 == 0)
            {
                return std::string("m.m00 is 0");
            }
            res = cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
        }
        break;
        /// 拟合直线
        case CenterMode::LBASE:
        {
            std::vector<std::vector<cv::Point2f>> contours2f = {contour2f};
            m_linesFit = std::make_unique<FitLines>(contours2f, m_linesFitParams.cnt);
            std::cout << "fitting... ...\n";
            auto r = m_linesFit->fit(m_linesFitParams.lineMode, m_linesFitParams.distType, m_linesFitParams.reps, m_linesFitParams.aeps);
            std::cout << "line fitted... ...\n";
            /// 报错
            if (r.index() == 1)
            {
                return std::get<std::string>(r);
            }
            /// 拟合成功
            if (r.index() == 0)
            {
                auto lines = std::get<std::vector<std::pair<cv::Point2f, cv::Point2f>>>(r);
                if (lines.empty())
                {
                    return std::string("lines is empty");
                }
                /// 搜索中心点
                std::cout << "center searching... ...\n";
                res = centerWithMinNorm(lines, m_linesFitParams.p0, m_linesFitParams.maxIter, m_linesFitParams.learningRate);
                return res;
            }
        }

        break;
            /// 质心
        case CenterMode::CBASE:
        {
            try
            {
                std::vector<cv::Point2f> hullcontour;

                cv::convexHull(contour2f, hullcontour);
                cv::Moments m = cv::moments(hullcontour);
                /// 计算质心
                if (m.m00 == 0)
                {
                    return std::string("m.m00 is 0");
                }
                res = cv::Point2f(m.m10 / m.m00, m.m01 / m.m00);
            }
            catch (const cv::Exception &e)
            {
                std::cerr << e.what() << '\n';
            }
        }
        break;
        default:

            break;
        }
        return res;
    }

    ~CenterSearch()
    {
    }

  std::unique_ptr<FitLines> m_linesFit;
public:
    CenterMode m_mode;
    LinesFitParams m_linesFitParams;
};
#endif