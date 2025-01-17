#include <iostream>
#include <filesystem>
#include "Yolov10.h"
#include "Yolov10SAM.h"
#include "Yolov10Trace.h"
#include "SAM2.h"
#include <chrono>
#include "ortfortcp.h"



void yolo()
{
    auto yolov10 = std::make_unique<Yolov10>();
    std::vector<std::string> onnx_paths{"D:\\m_code\\sam2_layout\\OrtInference-main\\models\\yolov10\\yolov10s.onnx"};
    auto r = yolov10->initialize(onnx_paths, true);
    if (r.index() != 0)
    {
        std::string error = std::get<std::string>(r);
        std::println("错误：{}", error);
        return;
    }
    yolov10->setparms({.score = 0.5f, .nms = 0.8f});

    //list file
    std::string folder_path = "D:/m_code/sam2_layout/OrtInference-main/assets/input/*.jpg";
    std::string output_path = "D:\\m_code\\sam2_layout\\OrtInference-main\\assets\\output\\";

    std::vector<cv::String> paths;
    cv::glob(folder_path, paths, false);

    for (const auto &path : paths)
    {
        std::println("path={}", path);
        cv::Mat image = cv::imread(path);
        auto start = std::chrono::high_resolution_clock::now();
        auto result = yolov10->inference(image);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::println("duration = {}ms", duration);
        if (result.index() == 0)
        {
            auto filename = std::filesystem::path(path).filename().string();
            cv::imwrite(output_path + filename, image);
            cv::imshow("Image", image);
            cv::waitKey(0);
        }
        else
        {
            std::string error = std::get<std::string>(result);
            std::println("错误：{}", error);
            continue;
        }
    }
}
void yolosam()
{
    auto yolov10sam = std::make_unique<Yolov10SAM>();
    std::vector<std::string> onnx_paths{
        "../models/yolov10/yolov10m.onnx",
        "../models/sam/ESAM_encoder.onnx",
        "../models/sam/ESAM_deocder.onnx"};
    auto r = yolov10sam->initialize(onnx_paths, true);
    if (r.index() != 0)
    {
        std::string error = std::get<std::string>(r);
        std::println("错误：{}", error);
        return;
    }
    yolov10sam->setparms({.score = 0.5f, .nms = 0.8f});

    std::string folder_path = "../assets/input/*.jpg";
    std::string output_path = "../assets/output/";

    std::vector<cv::String> paths;
    cv::glob(folder_path, paths, false);

    for (const auto &path : paths)
    {
        std::println("path={}", path);
        cv::Mat image = cv::imread(path);
        auto start = std::chrono::high_resolution_clock::now();
        auto result = yolov10sam->inference(image);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::println("duration = {}ms", duration);
        if (result.index() == 0)
        {
            // auto filename = std::filesystem::path(path).filename().string();
            // cv::imwrite(output_path+filename,image);
            // cv::imshow("Image", image);
            // cv::waitKey(0);
        }
        else
        {
            std::string error = std::get<std::string>(result);
            std::println("错误：{}", error);
            continue;
        }
    }
}
void yolotrace()
{
    auto yolov10trace = std::make_unique<Yolov10Trace>();
    std::vector<std::string> onnx_paths{"../models/yolov10/yolov10m.onnx"};
    auto r = yolov10trace->initialize(onnx_paths, true);
    if (r.index() != 0)
    {
        std::string error = std::get<std::string>(r);
        std::println("错误：{}", error);
        return;
    }
    std::string video_path = "../assets/video/test.mp4";
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened())
        return;
    //************************************************************
    std::cout << "视频中图像的宽度=" << capture.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "视频中图像的高度=" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "视频帧率=" << capture.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "视频的总帧数=" << capture.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
    //************************************************************
    yolov10trace->setparms({.camera_fps = 25,
                            .buffer_size = 20,
                            .score = 0.5f,
                            .nms = 0.5f});
    //************************************************************
    cv::Mat frame;
    while (true)
    {
        if (!capture.read(frame) || frame.empty())
            break;
        auto result = yolov10trace->inference(frame);
        if (result.index() == 0)
        {
            cv::imshow("frame", frame);
            int key = cv::waitKey(10);
            if (key == 'q' || key == 27)
                break;
        }
        else
        {
            std::string error = std::get<std::string>(result);
            std::println("错误：{}", error);
            break;
        }
    }
    capture.release();
}

void sam2()
{
    auto sam2 = std::make_unique<SAM2>();
    std::vector<std::string> onnx_paths{
        "D:/m_code/sam2_layout/OrtInference-main/models/sam2/large/image_encoder.onnx",
        "D:/m_code/sam2_layout/OrtInference-main/models/sam2/large/memory_attention.onnx",
        "D:/m_code/sam2_layout/OrtInference-main/models/sam2/large/image_decoder.onnx",
        "D:/m_code/sam2_layout/OrtInference-main/models/sam2/large/memory_encoder.onnx"};
    auto r = sam2->initialize(onnx_paths, true);
    if (r.index() != 0)
    {
        std::string error = std::get<std::string>(r);
        std::println("错误：{}", error);
        return;
    }
    sam2->setparms({.type = 1,
                    .prompt_box = {745, 695, 145, 230},
                    .prompt_point = {700, 500}}); // 在原始图像上的box,point

    std::string video_path = "D:/m_code/sam2_layout/OrtInference-main/assets/video/test.mkv";
    std::cout << "video_path = " << video_path << std::endl;
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened())
        return;
    //************************************************************
    std::cout << "*********************info************************\n";
    std::cout << "width=" << capture.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "height=" << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "rate of frame=" << capture.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "count of frame=" << capture.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;
    std::cout << "*************************************************\n";
    //************************************************************
    cv::Mat frame;
    size_t idx = 0;
    while (true)
    {
        if (!capture.read(frame) || frame.empty())
            break;
        auto start = std::chrono::high_resolution_clock::now();
        auto result = sam2->inference(frame);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::println("frame = {},duration = {}ms", idx++, duration);
        if (result.index() == 0)
        {
            std::string text = std::format("frame = {},fps={:.1f}", idx, 1000.0f / duration);
            cv::putText(frame, text, cv::Point{30, 40}, 1, 2, cv::Scalar(0, 0, 255), 2);
            cv::imshow("frame", frame);
            int key = cv::waitKey(5);
            if (key == 'q' || key == 27)
                break;
        }
        else
        {
            std::string error = std::get<std::string>(result);
            std::println("错误：{}", error);
            break;
        }
    }
    capture.release();
}

void sam2pic()
{
    /// 1、开辟对象
    auto sam2 = std::make_unique<SAM2>();
    /// 2、初始化模型参数路径
    std::vector<std::string> onnx_paths{
        "D:/m_code/sam2_layout/OrtInference-main/models/sam2/large/image_encoder.onnx",
        "D:/m_code/sam2_layout/OrtInference-main/models/sam2/large/memory_attention.onnx",
        "D:/m_code/sam2_layout/OrtInference-main/models/sam2/large/image_decoder.onnx",
        "D:/m_code/sam2_layout/OrtInference-main/models/sam2/large/memory_encoder.onnx"};
    /// 3、初始化模型
    auto r = sam2->initialize(onnx_paths, true);
    if (r.index() != 0)
    {
        std::string error = std::get<std::string>(r);
        std::println("错误：{}", error);
        return;
    }

    //开始计时
    auto start = std::chrono::high_resolution_clock::now();

    /// 4、设置prompt
    sam2->setparms({.type = 1,
                    .prompt_box = {745, 695, 145, 230},
                    .prompt_point = {2279, 1500}}); // 在原始图像上的box,point
    /// 5、加载图片
    std::string image_path =string("C:\\Users\\zydon\\Desktop\\JH_pic\\12.5\\x\\left\\left0_20241205091330561.bmp");
    cv::Mat image = cv::imread(image_path);
    /// 6、推理
    auto result = sam2->inference(image);
    /// 成功推理
    if (result.index() == 0)
    {
        cv::namedWindow("image", cv::WINDOW_NORMAL);
        cv::imshow("image", image);
        // 结束计时
        auto end = std::chrono::high_resolution_clock::now();
        // 计算耗时
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // 输出耗时
        std::cout << "推理耗时：" << duration << "ms" << std::endl;
        cv::waitKey(0);
     
    }
       /// 推理失败
    else
    {
        std::string error = std::get<std::string>(result);
        std::println("错误：{}", error);
    }
}

int yolo11_seg()
{

    std::unique_ptr<Yolov8SegOnnx> yolov8seg= std::make_unique<Yolov8SegOnnx>();
    std::string model_path_seg = "D:\\m_code\\sam2_layout\\OrtInference-main\\models\\yolo11_seg\\yolo11_0113.onnx";
    if (yolov8seg.get()->ReadModel(model_path_seg, true, 0, true))
    {
        std::cout << "yolov8seg loaded!" << std::endl;
    }
    else
    {
        std::cout << "yolov8seg loaded failed!" << std::endl;
        return 1;
    }

    ///所有的图像输出的结果
    std::vector<std::vector<cv::Point2f> > ptoutputs_all;
    ///每个图像输出的结果
    std::vector<OutputParams> outputs;
    // list file
    std::string folder_path = "C:\\Users\\zydon\\Desktop\\JH_pic\\12.5\\x\\left\\*.bmp";
    std::string output_path = "C:\\Users\\zydon\\Desktop\\JH_pic\\12.5\\x\\left";

    std::vector<cv::String> paths;
    cv::glob(folder_path, paths, false);

    for (const auto &path : paths)
    {
        std::println("path={}", path);
        cv::Mat image = cv::imread(path);
        auto start = std::chrono::high_resolution_clock::now();

        auto img = image.clone();
        /// 成功推理
        if (yolov8seg.get()->OnnxDetect(img, outputs))
        {
            std::vector<cv::Point2f> out_points;
            std::unique_ptr<CenterSearch> centerSearch_ptr = std::make_unique<CenterSearch>();
            /// 设置模式
            centerSearch_ptr.get()->m_mode = CenterSearch::CenterMode::CBASE;

            for (const auto &output : outputs)
            {
                auto ptfilter = getEdgePointsFromMask(output);
                auto pt = centerSearch_ptr.get()->contourCenter(myutil::cvpt2cvptf(ptfilter));
                if (pt.index())
                {
                    std::string error = std::get<std::string>(pt);
                    std::println("错误：{}", error);
                }
                else
                {
                    out_points.push_back(std::get<cv::Point2f>(pt));
                }
            }
            ptoutputs_all.push_back(out_points);
            auto end = std::chrono::high_resolution_clock::now();
            // 计算耗时
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            // 输出耗时
            std::cout << "推理总耗时：" << duration << "ms" << std::endl;
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    std::string input;
    std::string input2;  
    uint portNum; 
    // 提示用户输入  
    std::cout << "请选择加载的模型(1、分割,2、yolo检测,3、yolo分割):";  
    std::getline(std::cin, input); // 从终端获取输入 

    while (true) {  
        try {  
            std::cout << "请输入端口号(2048~65535): ";  
            std::getline(std::cin, input2); // 从终端获取输入  
            portNum = std::stoi(input2);  
            if (portNum < 2048 || portNum > 65535)  
                throw std::invalid_argument("端口号必须在2048~65535之间");  
            // 输入有效，跳出循环  
            break;  
        } catch (const std::exception &e) {  
            std::cout << "错误：" << e.what() << std::endl;  
        }  
    }  

    // 根据用户输入执行相应的操作
    if (input == "1") {
        std::cout << "加载分割模型..." << std::endl;
        // 加载分割模型的代码
          ortsam2fortcp(portNum);
    } else if (input == "2") {
        std::cout << "加载yolo检测模型..." << std::endl;
        // 加载检测模型的代码
          ortyolofortcp(2);

    } else if (input == "3") {
        std::cout << "加载yolo分割模型..." << std::endl;
        // 加载检测模型的代码
          ortyolofortcp(3);
    }
    else {
        std::cout << "无效的输入，请重新输入。" << std::endl;
    }

  
    // yolo11_seg();
    // yolo();
    // yolosam();
    // yolotrace();
    // sam2();
    // sam2pic();
    return a.exec();
}
