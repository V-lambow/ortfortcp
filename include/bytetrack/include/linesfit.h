#ifndef _LINESFIT_HPP_
#define _LINESFIT_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <variant>
#include <algorithm>
#include <cmath>
#include <opencv2/ximgproc.hpp>
#define _CLASS_LINEFANDCONTOURS_                       \
    using Linef = std::pair<cv::Point2f, cv::Point2f>; \
    using Contours = std::vector<std::vector<cv::Point>>;

#define CVIMSHOW(imgName)                          \
    cv::namedWindow("imgName", cv::WINDOW_NORMAL); \
    cv::imshow("imgName", imgName);

namespace myutil
{
    _CLASS_LINEFANDCONTOURS_

    class FitlineMethod
    {

    protected:
        FitlineMethod() = default;
        ~FitlineMethod() = default;
        virtual std::variant<std::vector<Linef>, std::string> fit(size_t cnt = 0) {}
    };

    class FitLeastSquare : public FitlineMethod
    {

    public:
        /// @brief 参数
        /// @param m_distType 距离类型
        /// @param m_reps 距离阈值
        /// @param m_aeps 角度阈值
        struct Params
        {
            cv::DistanceTypes m_distType = cv::DIST_L2;
            double m_reps = 0.01;
            double m_aeps = 0.01;
        };
        FitLeastSquare(Contours *contours, FitLeastSquare::Params params) : m_contours(contours), m_params(params) {}
        FitLeastSquare(cv::Mat *contoursMat, FitLeastSquare::Params params) : m_contoursMat(contoursMat), m_params(params) {}
        std::variant<std::vector<Linef>, std::string> fit(size_t cnt = 0) override ;

    public:
        cv::Mat *m_contoursMat = nullptr;
        Contours *m_contours = nullptr;
        /// @brief 需要拟合的直线数量
        size_t m_cnt = 0;
        Params m_params;
    };

    class FitProsac: public FitlineMethod
    {
    public:
    struct Params
    {
        /* data */
    };
    
    public:
        FitProsac(/* args */);
        ~FitProsac();
    };

    class Hough : public FitlineMethod
    {

    public:
        /// @brief 参数
        /// @param rho 累加器的距离分辨率（像素）。
        /// @param theta 累加器的角度分辨率（弧度）。
        /// @param srn 距离分辨率 rho 的除数
        /// @param stn 距离分辨率 theta 的除数
        /// @param min_theta 线条的最小角度, [0,max_theta]
        /// @param max_theta 线条的最大角度, [min_theta,max_theta]
        /// @param threshold 累加器阈值参数
        struct Params
        {
            double rho = 1,
                   theta = CV_PI / 180,
                   srn = 0,
                   stn = 0,
                   min_theta = 0,
                   max_theta = CV_PI;
            int threshold = 150;
        };

        Hough(cv::Mat *binaryImage, Params params) : m_binaryImage(binaryImage), m_params(params) {}
        ~Hough(){}
        std::variant<std::vector<Linef>, std::string> fit(size_t cnt = 0);

    public:
        Params m_params;
        /// @brief 8位二值图像
        cv::Mat *m_binaryImage = nullptr;
        size_t m_cnt = 0;
    };

    class HoughP :public FitlineMethod
    {
    public:
        /// @brief 参数
        /// @param rho 累加器的距离分辨率（像素）。
        /// @param theta 累加器的角度分辨率（弧度）。
        /// @param threshold 累加器阈值参数
        /// @param minLineLength 最小线段长度
        /// @param maxLineGap 最大线段间隔
        struct Params
        {
            double rho = 1,
                   theta = CV_PI / 180,
                   threshold = 150,
                   minLineLength = 100,
                   maxLineGap = 10;
        };
        HoughP(cv::Mat *binaryImage, Params params) : m_binaryImage(binaryImage), m_params(params) {}
        ~HoughP()=default;
        std::variant<std::vector<Linef>, std::string> fit(size_t cnt = 0);

    public:
        Params m_params;
        /// @brief 8位二值图像
        cv::Mat *m_binaryImage = nullptr;
        size_t m_cnt = 0;
    };



    class FitLSD : public FitlineMethod
    {

        /// @brief 参数
        /// @param refine LSD_REFINE_STD 采用标准细化。 例如，将拱形分解成更小的直线近似值
        ///               LSD_REFINE_ADV计算误报数量，通过提高精度、减小尺寸等方式细化线条。
        /// @param scale 用于查找线条的图像比例。 范围 (0.1]。
        /// @param sigma_scale   高斯滤波器的西格玛值。 计算公式为 sigma = sigma_scale/scale。
        /// @param quant 		梯度规范的量化误差约束。
        /// @param ang_th 	梯度角公差（度）。
        /// @param log_eps 检测阈值： -log10(NFA)>log_eps。 仅在选择提前细化时使用。
        /// @param density_th 	包围矩形内对齐区域点的最小密度。
        /// @param n_bins 	梯度模数伪排序的分段数。
        struct Params
        {
            cv::LineSegmentDetectorModes refine = cv::LSD_REFINE_STD;
            double scale = 0.8,
                   sigma_scale = 0.6,
                   quant = 2.0,
                   ang_th = 22.5,
                   log_eps = 0,
                   density_th = 0.7;
            int n_bins = 1024;
        };

    public:
        FitLSD(cv::Mat *grayImage, Params params) : m_grayImage(grayImage), m_params(params) {}
        ~FitLSD() {}
  

        std::variant<std::vector<Linef>, std::string> fit(size_t cnt = 0);

    public:
        Params m_params;
        /// @brief 8uc1图像
        cv::Mat *m_grayImage = nullptr;
    };

    /// @brief 最小二乘法拟合
    /// @param fitmethod
    std::variant<std::vector<Linef>, std::string> calledFunc(FitLeastSquare *fitmethod);
    /// @brief LSD直线检测
    /// @param fitmethod
    std::variant<std::vector<Linef>, std::string> calledFunc(FitLSD *fitmethod);
    /// @brief 霍夫多尺度直线检测
    /// @param fitmethod
    std::variant<std::vector<Linef>, std::string> calledFunc(Hough *fitmethod);
    /// @brief 霍夫概率直线检测
    /// @param fitmethod
    std::variant<std::vector<Linef>, std::string> calledFunc(HoughP *fitmethod);

    std::vector<Linef> vec4f2Linf(std::vector<cv::Vec4f> lines);
}
#endif // _LINESFIT_HPP_