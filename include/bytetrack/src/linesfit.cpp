#include "linesfit.h"
_CLASS_LINEFANDCONTOURS_

std::variant<std::vector<Linef>, std::string> myutil::FitLeastSquare::fit(size_t cnt) { return calledFunc(this); }
std::variant<std::vector<Linef>, std::string> myutil::FitLSD::fit(size_t cnt) { return calledFunc(this); }
std::variant<std::vector<Linef>, std::string> myutil::Hough::fit(size_t cnt) { return calledFunc(this); }
std::variant<std::vector<Linef>, std::string> myutil::HoughP::fit(size_t cnt) { return calledFunc(this); }

std::variant<std::vector<Linef>, std::string> myutil::calledFunc(FitLeastSquare *fitmethod)
{

    /// 初始化输出
    std::vector<Linef> res;
    std::vector<cv::Vec4f> lines;
    /// 都未初始化
    if ((!fitmethod->m_contours) && (!fitmethod->m_contoursMat))
    {
        return std::string("输入点集/矩阵内容为空");
    }
    else if ((fitmethod->m_contours) && (fitmethod->m_contoursMat))
    {
        return std::string("输入点集/矩阵内容不能同时存在");
    }
    else if (fitmethod->m_contours)
    {
        cv::fitLine(*fitmethod->m_contours, lines, fitmethod->m_params.m_distType, 0, fitmethod->m_params.m_reps, fitmethod->m_params.m_aeps);
    }
    else if (fitmethod->m_contoursMat)
    {
        cv::fitLine(*fitmethod->m_contoursMat, lines, fitmethod->m_params.m_distType, 0, fitmethod->m_params.m_reps, fitmethod->m_params.m_aeps);
    }
    else
    {
        return std::string("未知错误");
    }
    for (const auto &line : lines)
    {
        float vx = line[0], vy = line[1], x0 = line[2], y0 = line[3];
        Linef lineTmp{
            cv::Point2f(x0 - 1000 * vx, y0 - 1000 * vy),
            cv::Point2f(x0 + 1000 * vx, y0 + 1000 * vy)};
        res.push_back(lineTmp);
    }
    return res;
}

std::variant<std::vector<Linef>, std::string> myutil::calledFunc(Hough *fitmethod)
{
    if (!fitmethod->m_binaryImage)
    {
        return std::string("输入图像为空");
    }
    std::vector<cv::Vec3f> lines;
    std::vector<Linef> res;
    cv::HoughLines(*fitmethod->m_binaryImage, lines, fitmethod->m_params.rho, fitmethod->m_params.theta, fitmethod->m_params.threshold, fitmethod->m_params.srn, fitmethod->m_params.stn, fitmethod->m_params.min_theta, fitmethod->m_params.max_theta);
    // 遍历每一个r和theta
    for (size_t i = 0; i < lines.size(); i++)
    {
        float r = lines[i][0];
        float theta = lines[i][1];

        // 存储cos(theta)和sin(theta)的值
        float a = cos(theta);
        float b = sin(theta);

        // 存储rcos(theta)和rsin(theta)的值
        float x0 = a * r;
        float y0 = b * r;

        // 计算直线的两个端点
        cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * a));
        cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * a));
        res.push_back(Linef(pt1, pt2));
    }
    return res;
}

std::variant<std::vector<Linef>, std::string> myutil::calledFunc(FitLSD *fitmethod)
{
    if (!fitmethod->m_grayImage)
    {
        return std::string("输入图像为空");
    }
    auto lsd = cv::createLineSegmentDetector(fitmethod->m_params.refine,
                                             fitmethod->m_params.scale,
                                             fitmethod->m_params.sigma_scale,
                                             fitmethod->m_params.quant,
                                             fitmethod->m_params.ang_th,
                                             fitmethod->m_params.log_eps,
                                             fitmethod->m_params.density_th,
                                             fitmethod->m_params.n_bins);
    std::vector<cv::Vec4f> lines;
    lsd->detect(*fitmethod->m_grayImage, lines);
    auto res = vec4f2Linf(lines);
    return res;
}

std::variant<std::vector<Linef>, std::string> myutil::calledFunc(HoughP *fitmethod)
{
    if (!fitmethod->m_binaryImage)
    {
        return std::string("输入图像为空");
    }
    std::vector<cv::Vec4f> lines;
    cv::HoughLinesP(*fitmethod->m_binaryImage, lines, fitmethod->m_params.rho, fitmethod->m_params.theta, fitmethod->m_params.threshold, fitmethod->m_params.minLineLength, fitmethod->m_params.maxLineGap);
    auto res =vec4f2Linf(lines);
    return res;
}

std::vector<Linef> myutil::vec4f2Linf(std::vector<cv::Vec4f> lines)
{
    std::vector<Linef> res;
    for (const auto &line : lines)
    {
        float x1 = line[0], x2 = line[1], x3 = line[2], x4 = line[3];
        res.push_back({cv::Point2f(x1, x2),
                       cv::Point2f(x3, x4)});
    }
    return res;
}
