#include "model_base.h"


std::optional<std::string> ModelBase::setClassesList(std::string classesPath){
    std::ifstream file(classesPath);
    if (!file.is_open())
    {
        return "error: classes file not found";
    }
    std::string line;
    while (std::getline(file, line))
    {
        m_classesList.push_back(line);
    }
    file.close();
    return std::nullopt;
}

std::variant<std::string,std::string> ModelBase::getClassName(int index){
    if (index < 0 || index >= m_classesList.size())
    {
        return "error: index out of range";
    }
    return m_classesList.at(index);
}

std::optional<std::string> ModelBase::CheckNetSize(int netHeight, int netWidth, const int* netStride, int strideSize) {
	if (netHeight % netStride[strideSize - 1] != 0 || netWidth % netStride[strideSize - 1] != 0)
	{
		return "error: _netHeight and _netWidth must be multiple of max stride";
	}
	return std::nullopt;
}

std::optional<std::string> ModelBase::CheckPath(std::string path) {
	if (0 != _access(path.c_str(), 0)) {
		return "error: path does not exist,  please check " + path;   
	}
	else
		return std::nullopt;
}


void resizeAndPadImg(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
	bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
{
	 // 获取原始图像尺寸
    cv::Size shape = image.size();
    
    // 计算缩放比例，保持宽高比
    float r = std::min(static_cast<float>(newShape.height) / shape.height,
                       static_cast<float>(newShape.width) / shape.width);
    
    // 如果不允许放大，则限制最大缩放比例为1.0
    if (!scaleUp) {
        r = std::min(r, 1.0f);
    }
    
    // 计算缩放后的尺寸
    float ratio[2] = { r, r };
    int new_un_pad[2] = { 
        static_cast<int>(std::round(shape.width * r)),
        static_cast<int>(std::round(shape.height * r))
    };
    
    // 计算填充量
    auto dw = static_cast<float>(newShape.width - new_un_pad[0]);
    auto dh = static_cast<float>(newShape.height - new_un_pad[1]);
    
    // 如果使用自动形状，则调整填充量以满足步长要求
    if (autoShape) {
        dw = static_cast<float>(static_cast<int>(dw) % stride);
        dh = static_cast<float>(static_cast<int>(dh) % stride);
    }
    // 如果使用拉伸填充，则直接填充到目标尺寸
    else if (scaleFill) {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = static_cast<float>(newShape.width) / shape.width;
        ratio[1] = static_cast<float>(newShape.height) / shape.height;
    }
    
    // 将填充量平均分配到四周
    dw /= 2.0f;
    dh /= 2.0f;
    
    // 执行缩放
    if (shape.width != new_un_pad[0] || shape.height != new_un_pad[1]) {
        cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    } else {
        outImage = image.clone();
    }
    
    // 计算填充参数
    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));
    
    // 记录变换参数
    params[0] = ratio[0];  // 宽度缩放比例
    params[1] = ratio[1];  // 高度缩放比例
    params[2] = left;      // 左填充量
    params[3] = top;       // 上填充量
    
    // 执行填充
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, 
                       cv::BORDER_CONSTANT, color);
}
