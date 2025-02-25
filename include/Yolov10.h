#pragma once
#include "Model.h"
#include <fstream>
#include <print>


class Yolov10:public yo::Model{

struct ParamsV10{
    float score = 0.5f;
    float nms = 0.5f;
};

private:
    bool is_inited = false;
    cv::Mat* ori_img = nullptr;

    ParamsV10 parms;
    std::vector<yo::Node> input_nodes;
    std::vector<yo::Node> output_nodes;
    std::vector<cv::Mat> input_images;

    Ort::Session* session = nullptr;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"yolov10");
    Ort::SessionOptions session_options = Ort::SessionOptions();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
protected:
    void preprocess(cv::Mat &image);
    void postprocess(std::vector<Ort::Value>& output_tensors);
    void sortBoxesByNames(std::vector<int>& names, std::vector<cv::Rect>& boxes);
public:
    Yolov10(){};
    Yolov10(const Yolov10&) = delete;// 删除拷贝构造函数
    Yolov10& operator=(const Yolov10&) = delete;// 删除赋值运算符
    ~Yolov10(){if(session != nullptr) delete session;};
    int setparms(ParamsV10 parms);
    std::variant<bool,std::string> initialize(std::vector<std::string>& onnx_paths, bool is_cuda) override;
    std::variant<bool,std::string> inference(cv::Mat &image) override;
    std::variant<bool,std::string> prewarm_model() ;
    cv::Mat drawBoxes(std::vector<int>indices,std::vector<int>labels,std::vector<float>scores,std::vector<cv::Rect>boxes);

    void outputClear();
    cv::Mat output_img{};
    std::vector<cv::Point2f> output_point{};
    std::vector<int> output_labels{};
    std::vector<cv::Rect> output_boxes{};
};