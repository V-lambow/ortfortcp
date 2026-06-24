#include "myutil.h"
#include "lm.hpp"

void test_myutil_caldegree()
{
    cv::Point p1(3789.12183, 1347.69324);
    cv::Point p2(3907.16040 - 1349.87195);
    int pulseVal = 300;
    double degree = myutil::caldegree(pulseVal, p1, p2);
    std::cout << "degree:" << degree << std::endl;
    std::cout << "hello world" << std::endl;
    system("pause");
}

void test_myutil_calCircled()
{
    myutil::Circled circle = myutil::calCircled({{3837.38647, 2160.12915}, {3820.16553, 2608.33862} /*,{3815.24902,2833.43726}*/, {3834.17065, 2385.50171}});
    std::cout << "center:" << circle.center << std::endl;
    std::cout << "radius:" << circle.radius << std::endl;
}

void test_myutil_LM()
{
    std::cout << "hello world" << std::endl;

    // ||a1-x0+r1*cosx|+|b1-y0+r1*sinx|-|c1-z0+r2*cosx|-|d1-q0+r2*sinx||
    double a1 = 1.0, b1 = 2.0, c1 = 3.0, d1 = 4.0;
    double x0 = 0.5, y0 = 1.5, z0 = 2.5, q0 = 3.5;
    double r1 = 1.0, r2 = 2.0;

    // 设置初始值
    double initial_x = 0.0;

    // 创建代价函数对象
    CostFunctor cost_function(a1, b1, c1, d1, x0, y0, z0, q0, r1, r2);

    // 运行Levenberg-Marquardt算法
    double estimated_x = levenbergMarquardt(cost_function, initial_x);

    // 输出结果
    std::cout << "Estimated x: " << estimated_x << "\n";
}
// ||a1-x0+r1*cosx|+|b1-y0+r1*sinx|-|c1-z0+r2*cosx|-|d1-q0+r2*sinx||
int main()
{
    // test_myutil_LM();

    test_myutil_calCircled();
    system("pause");
}