#pragma once
#include "model_base.h"
#include <fstream>
#include <print>
#include <memory>

class Yolov10:public yo::ModelBase{

struct Params{
    float score = 0.5f;
    float nms = 0.5f;
};

private:
    bool is_inited = false;
    cv::Mat* ori_img = nullptr;

    Params parms;
    std::vector<yo::Node> input_nodes;
    std::vector<yo::Node> output_nodes;
    std::vector<cv::Mat> input_images;

    std::unique_ptr<Ort::Session> session;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"yolov10");
    Ort::SessionOptions session_options = Ort::SessionOptions();
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
protected:
    void preprocess(cv::Mat &image);
    void postprocess(std::vector<Ort::Value>& output_tensors);
    void sortBoxesByNames(std::vector<int>& names, std::vector<cv::Rect>& boxes);
public:
    Yolov10() = default;
    Yolov10(const Yolov10&) = delete;
    Yolov10& operator=(const Yolov10&) = delete;
    ~Yolov10() = default;

    bool setparms(Params parms);
    std::variant<bool,std::string> initialize(std::vector<std::string>& onnx_paths, bool is_cuda) override;
    std::variant<bool,std::string> inference(cv::Mat &image) override;
    std::variant<bool,std::string> prewarm_model() ;

    void outputClear();
    cv::Mat drawMarkers(std::vector<int> indices, std::vector<int> labels, std::vector<float> scores, std::vector<cv::Rect> boxes);

    std::vector<yo::OutputParams> m_outputs{};
    std::vector<cv::Rect> output_boxes{};
    std::vector<int> output_labels{};
    std::vector<cv::Point2f> output_point{};

    std::vector<yo::OutputParams> m_output_params{};
    cv::Mat output_img{};
};