#ifndef _BASE_H
#define _BASE_H
#include <print>
#include <memory>
#include <vector>
#include <iostream>
#include <functional>
#include <expected>


#ifndef Q_MOC_RUN
#ifdef emit
#undef emit
#include <tbb/tbb.h>
#define emit
#else
#include <tbb/tbb.h>
#endif // emit
#endif // Q_MOC_RUN


#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <variant>
#include <opencv2/ximgproc.hpp>
#include "centerSearch.hpp"
#include "linesfit.hpp"
#include "myutil.h"
#include <fstream>
#include <optional>


namespace yo
{
    enum class ModelType {
        yolov10_det=0,
        yolov10_sam=1,
        sam2 =2,
        yolo11_seg=3,
        yolo11_pose=4,
    };

    struct PoseKeyPoint {
        float x = 0;
        float y = 0;
        float confidence = 0;
    };

    struct OutputParams {
        int id;             //结果类别id
        float confidence;   //结果置信度
        cv::Rect box;       //矩形框
        cv::RotatedRect rotatedBox;  //obb结果矩形框
        cv::Mat boxMask;       //矩形框内mask，节省内存空间和加快速度
        std::vector<PoseKeyPoint> keyPoints; //pose key points

    };
    struct MaskParams {
        int netWidth = 640;
        int netHeight = 640;
        float maskThreshold = 0.5;
        cv::Size srcImgShape;
        cv::Vec4d params;
    };

    struct Node
    {
        std::vector<int64_t> dim; // batch,channel,height,width
        char *name = nullptr;
    };

    class ModelBase
    {
    public:
        virtual ~ModelBase() {};
        virtual std::optional<std::string> inference(cv::Mat &image) = 0;
        virtual std::optional<std::string> initialize(std::vector<std::string> &onnx_paths, bool is_cuda) = 0;

    protected:
        virtual void preprocess(cv::Mat &image) = 0;
        virtual void postprocess(std::vector<Ort::Value> &output_tensors) = 0;
        virtual std::optional<std::string> setClassesList(std::string classesPath);
        virtual std::variant<std::string,std::string> getClassName(int index);
        virtual std::optional<std::string> CheckPath(std::string path);
        virtual std::optional<std::string> CheckNetSize(int netHeight, int netWidth, const int* netStride, int strideSize);

    	std::vector<std::string> m_classesList = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"};

        bool m_isDraw = false;
    };

    // 没考虑线程安全的问题
    // 环形队列
    template <typename T, size_t N>
    class FixedSizeQueue
    {
    private:
        std::vector<T> data;
        size_t head;
        size_t tail;
        size_t count;

    public:
        FixedSizeQueue() : head(0), tail(0), count(0) {}
        bool push(T &&value)
        {
            if (this->full())
            {
                tail = (tail + 1) % N;
                data[head] = std::move(value);
            }
            else
                {
                    data.push_back(std::move(value));
                    count++;
                }
            head = (head + 1) % N;
            return true;
        }
        bool push(const T &value)
        {
            return push(T(value)); // Create a copy and use the rvalue overload
        }
        T &at(size_t idx)
        {
            if (idx >= this->count)
                throw std::out_of_range("Index out of range");
            idx = (tail + idx) % N;
            return data[idx];
        }
        bool empty() const { return count == 0; }
        bool full() const { return count == N; }
        size_t size() const { return count; }
    };

    void resizeAndPadImg(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
	bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color);
}
#endif // _BASE_H