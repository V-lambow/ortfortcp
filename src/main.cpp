#include <iostream>
#include <filesystem>
#include "Yolov10.h"
#include "Yolov10SAM.h"
#include "Yolov10Trace.h"
#include "SAM2.h"
#include <chrono>
#include "ortfortcp.h"
#include "yolov8_pose_onnx.h"




void yolo()
{
    auto yolov10 = std::make_unique<Yolov10>();
    std::vector<std::string> onnx_paths{"..\\..\\models\\yolov10\\yolov10m_0117.onnx"};
    auto r = yolov10->initialize(onnx_paths, true);
    if (r.index() != 0)
    {
        std::string error = std::get<std::string>(r);
        std::println("错误：{}", error);
        return;
    }
    yolov10->setparms({.score = 0.5f, .nms = 0.8f});

    //list file
    std::string folder_path = "C:\\Users\\zydon\\Desktop\\JH_pic\\01.24\\*.bmp";
    std::string output_path = "D:\\m_code\\sam2_layout\\OrtInference-main\\assets\\output\\0117\\";

    std::vector<cv::String> paths;
    cv::glob(folder_path, paths, false);

    for (const auto &path : paths)
    {
        std::println("path={}", path);
        cv::Mat image = cv::imread(path);
        auto start = std::chrono::high_resolution_clock::now();
        auto result = yolov10->inference(image);
        std::cout << "找到" << yolov10.get()->output_boxes.size() << "个目标" << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::println("yolo inference duration = {}ms", duration);
        if (result.index() == 0)
        {
            auto filename = std::filesystem::path(path).filename().string();
            cv::imwrite(output_path + filename, image);
            cv::namedWindow("Image", cv::WINDOW_NORMAL);
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
        "../models/sam2/large/image_encoder.onnx",
        "../models/sam2/large/memory_attention.onnx",
        "../models/sam2/large/image_decoder.onnx",
        "../models/sam2/large/memory_encoder.onnx"};

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
    std::string model_path_seg = "D:\\m_code\\sam2_layout\\OrtInference-main\\models\\yolo11_seg\\yolo11_seg0211.onnx";
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
    std::string folder_path = "C:\\Users\\zydon\\Desktop\\JH_pic\\01.24\\*.bmp";
    std::string output_path = "D:\\m_code\\sam2_layout\\OrtInference-main\\assets\\output\\0117\\";

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
                    std::vector<std::vector<cv::Point>> contours ={ptfilter};
                    cv::drawContours(img, contours, -1, cv::Scalar(0, 0, 255), 2);
                }
            }
            ptoutputs_all.push_back(out_points);
            cv::namedWindow("Image", cv::WINDOW_NORMAL);
            std::for_each(out_points.begin(), out_points.end(), [&](const auto &pt)
                           { cv::circle(img, pt, 5, cv::Scalar(0, 0, 255), -1); });
            // cv::drawContours(img, ptoutputs_all, -1, cv::Scalar(0, 0, 255), 2);
            cv::imshow("Image", img);
            cv::waitKey(0);

            auto end = std::chrono::high_resolution_clock::now();
            // 计算耗时
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            // 输出耗时
            std::cout << "推理总耗时：" << duration << "ms" << std::endl;
        }
    }
    return 0;
}

int yolosam2()
{
    auto yolov10 = std::make_unique<Yolov10>();
    std::vector<std::string> onnx_paths{"..\\..\\models\\yolov10\\yolov10m_0117.onnx"};
    auto r = yolov10->initialize(onnx_paths, true);
    if (r.index() != 0)
    {
        std::string error = std::get<std::string>(r);
        std::println("错误：{}", error);
        return 1;
    }
    yolov10->setparms({.score = 0.5f, .nms = 0.8f});

    auto sam2 = std::make_unique<SAM2>();
    std::vector<std::string> onnx_paths_sam{
        "../../models/sam2/large/image_encoder.onnx",
        "../../models/sam2/large/memory_attention.onnx",
        "../../models/sam2/large/image_decoder.onnx",
        "../../models/sam2/large/memory_encoder.onnx"};

    auto r_sam = sam2->initialize(onnx_paths_sam, true);
    if (r_sam.index() != 0)
    {
        std::string error = std::get<std::string>(r_sam);
        std::println("错误：{}", error);
        return 1;
    }

    // list file
    std::string folder_path = "C:\\Users\\zydon\\Desktop\\JH_pic\\01.24\\*.bmp";
    std::string output_path = "D:\\m_code\\sam2_layout\\OrtInference-main\\assets\\output\\0117\\";

    std::vector<cv::String> paths;
    cv::glob(folder_path, paths, false);

    for (const auto &path : paths)
    {
        auto filename = std::filesystem::path(path).filename().string();
        std::println("path={}", path);
        cv::Mat image = cv::imread(path);
        cv::Mat paintedimage=image.clone();
        auto start = std::chrono::high_resolution_clock::now();
        auto result = yolov10->inference(paintedimage);
        std::cout << "找到" << yolov10.get()->output_boxes.size() << "个目标" << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::println("yolo inference duration = {}ms", duration);

        if (result.index() == 1)
        {
            std::string error = std::get<std::string>(result);
            std::println("错误：{}", error);
            continue;
        }
        for (const auto &box : yolov10.get()->output_boxes)
        {
            // 设置prompt
            sam2->setparms({.type = 0,
                            .prompt_box = box,
                            .prompt_point = {700, 500}}); // 在原始图像上的box,point

            // // 设置prompt
            // sam2->setparms({.type = 1,
            //                 .prompt_box = {0,0,0,0},
            //                 .prompt_point = {box.x+box.height/2-1,box.y+box.width/2-1 }}); // 在原始图像上的box,point
            auto result_sam = sam2->inference(image);
            std::vector<std::vector<cv::Point>> contours_painted={sam2->output_contour};
            if (result_sam.index() == 0)
            {
                drawContours(paintedimage, contours_painted, -1, cv::Scalar(0, 255, 0), 2);
                cv::circle(paintedimage, sam2->output_point, 5, cv::Scalar(0, 255, 0), 2);
            }
            else
            {
                std::string error = std::get<std::string>(result_sam);
                std::println("错误：{}", error);
                continue;
            }
        }
        cv::imwrite(output_path + filename, paintedimage);
        cv::namedWindow("Image", cv::WINDOW_NORMAL);
        cv::imshow("Image", paintedimage);
        cv::waitKey(0);
    }
    return 0;
}

int yolo11_pose(){
    std::unique_ptr<Yolov8PoseOnnx> yolo11_pose = std::make_unique<Yolov8PoseOnnx>();
    std::string model_path_pose = "D:\\m_code\\sam2_layout\\OrtInference-main\\models\\yolo11_pose\\yolo11_pose0214.onnx";
    if (yolo11_pose.get()->ReadModel(model_path_pose, true, 0, true))
    {
        std::cout << "yolo11_pose loaded!" << std::endl;
    }
    else
    {
        std::cout << "yolo11_pose loaded failed!" << std::endl;
        return 1;
    }
    // list file
    std::string folder_path = "C:\\Users\\zydon\\Desktop\\JH_pic\\01.24\\*.bmp";
    std::string output_path = "D:\\m_code\\sam2_layout\\OrtInference-main\\assets\\output\\0117\\";
    std::vector<cv::String> paths;
    cv::glob(folder_path, paths, false);
    for (const auto &path : paths)
    {
        auto filename = std::filesystem::path(path).filename().string();
        std::println("path={}", path);
        cv::Mat image = cv::imread(path);
        cv::Mat paintedimage=image.clone();
        std::vector<OutputParams> outputs;
        auto start = std::chrono::high_resolution_clock::now();
        auto result = yolo11_pose.get()->OnnxDetect(paintedimage,outputs);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::println("yolo inference duration = {}ms", duration);
        if (result)
        {
            for (const auto &output : outputs)
            {
                for(const auto &kpt:output.keyPoints)
                {
                    cv::circle(paintedimage, cv::Point((int)kpt.x, (int)kpt.y), 5, cv::Scalar(0, 255, 0), 2);
                } 

            }
        }
        cv::imwrite(output_path + filename, paintedimage);
        cv::namedWindow("Image", cv::WINDOW_NORMAL);
        cv::imshow("Image", paintedimage);
        cv::waitKey(0);
    }
    return 0;


}


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    yolosam2();
    return 0;

    std::string input;
    uint portNum{}; 
    QString ip{};
    

    while (true)
    {
        std::cout << "please input ip adress: (xx.xx.xx.xx)";
        std::getline(std::cin, input); // 从终端获取输入
        if(input==""){
            input ="127.0.0.1";
        }
        if (isValidIp(input.c_str())) {
            ip = QString{input.c_str()};
            break;
        } else {
            std::cout << "invalid ip adress,retry." << std::endl;
        }
    }
    

    while (true) {  
        try {  
            std::cout << "please input port number[2048,65535]: ";  
            std::getline(std::cin, input); // 从终端获取输入  
            portNum = std::stoi(input);  
            if (portNum < 2048 || portNum > 65535)  
                throw std::invalid_argument("port number must be [2048,65535]");  
            // 输入有效，跳出循环  
            break;  
        } catch (const std::exception &e) {  
            std::cout << "error:" << e.what() << std::endl;  
        }  
    }  

        // 提示用户输入  
    std::cout << "please choose the model(1.segment,2.yolo detection,3.yolo segment,4.yolosam segment):";  
    std::getline(std::cin, input); // 从终端获取输入 


    // 根据用户输入执行相应的操作
    if (input == "1") {
        std::cout << "load 1 ai toolkit..." << std::endl;
        // 加载分割模型的代码
          ortsam2fortcp(ip,portNum);
    } else if (input == "2") {
        std::cout << "load 2 ai toolkit..." << std::endl;
        // 加载检测模型的代码
          ortyolofortcp(ip,portNum,2);

    } else if (input == "3") {
        std::cout << "load 3 ai toolkit..." << std::endl;
        // 加载检测模型的代码
          ortyolofortcp(ip,portNum,3);
    }
    else if (input == "4") {
        std::cout << "load 4 ai toolkit..." << std::endl;
        // 加载检测模型的代码
          ortyolosam2fortcp(ip,portNum);
    }
    else {
        std::cout << "error input。" << std::endl;
    }

  
    // yolo11_seg();
    
    // yolo();
    // yolosam();
    // yolotrace();
    // sam2();
    // sam2pic();
    return a.exec();
}
