#include <iostream>  
#include <Eigen/Dense>  
#include <cmath>  

// 代价函数结构体  
struct CostFunctor {  
    CostFunctor(double a1, double b1, double c1, double d1,   
                 double x0, double y0, double z0, double q0,  
                 double r1, double r2)  
        : a1(a1), b1(b1), c1(c1), d1(d1),  
          x0(x0), y0(y0), z0(z0), q0(q0),  
          r1(r1), r2(r2) {}  

    // 计算残差  
    void operator()(const double x, Eigen::VectorXd& residuals) const {  
        double cos_x = std::cos(x);  
        double sin_x = std::sin(x);  

        residuals[0] = a1 - x0 - r1 * cos_x;  
        residuals[1] = b1 - y0 - r1 * sin_x;  
        residuals[2] = c1 - z0 - r2 * cos_x;  
        residuals[3] = d1 - q0 - r2 * sin_x;  

        // 计算目标函数的绝对值和  
        residuals[4] = std::abs(residuals[0]) + std::abs(residuals[1])   
                     - std::abs(residuals[2]) - std::abs(residuals[3]);  
    }  

    const double a1, b1, c1, d1;  
    const double x0, y0, z0, q0;  
    const double r1, r2;  
};  

// Levenberg-Marquardt算法实现  
double levenbergMarquardt(const CostFunctor& func, double initial_x) {  
    double lambda = 1e-3; // 初始增益  
    double x = initial_x;  
    const int max_iter = 100;  
    const double tol = 1e-10;  
    
    for (int iter = 0; iter < max_iter; ++iter) {  
        Eigen::VectorXd residuals(5);  
        func(x, residuals);  
        Eigen::VectorXd J(5); // 雅可比矩阵的近似  
        for (int i = 0; i < 5; ++i) {  
            double h = 1e-6;  
            double x_plus = x + h;  
            Eigen::VectorXd residuals_plus(5);  
            func(x_plus, residuals_plus);  
            J[i] = (residuals_plus[i] - residuals[i]) / h; // 计算雅可比  
        }  

        // 计算更新步长  
        double delta_x = residuals.dot(J) / (J.dot(J) + lambda);  
        
        // 更新 x  
        x -= delta_x;  
        
        // 计算轮次收敛  
        if (std::abs(delta_x) < tol) {  
            break;  
        }  

        // 检查是否需要调整 lambda  
        lambda *= 0.9; // 可根据实验进行优化  
    }  
    return x;  
}  