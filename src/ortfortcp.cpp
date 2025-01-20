#include "ortfortcp.h"
#include "Model.h"
#include "SAM2.h"
#include "Yolov10.h"
#include "yolov8_seg_onnx.h"

/// @brief 连接的客户端数量
static uint connectedNum =0;
enum class SKORDER
{
    RECEIVE_IMG =0,
    RECEIVE_POINT =1,
    TIMEOUT =2,
    UNDEFINED =99
};

std::vector<cv::Point2f>  keyptsFliter(std::vector<PoseKeyPoint> kpts,float conf_thres=0) noexcept{
    std::vector<cv::Point2f> res;
    for(auto& kpt:kpts){
        if(kpt.confidence>conf_thres){
            res.push_back(cv::Point2f(kpt.x,kpt.y));
        }
    }
    return res;
}

/// prompt 点收到信号
static bool isPointReceived =false;
/// 图片收到信号
static bool isImageReceived =false;

/// @brief 重置收到信号
void resetReceived(){
    isImageReceived =false;
    isPointReceived =false;
    SKORDER::UNDEFINED;
}


/// @brief 指令集
static const QVector<QString> skorder={"imagein","pointin","timeout"};

/// @brief 当前的指令模式
static SKORDER curOrderMode =SKORDER::UNDEFINED;

/// @brief 将收到的二进制保存为图片
/// @param fileByte 二进制bit流
void saveImage(const QByteArray& fileByte){

    QFile file("C:\\Users\\zydon\\Pictures\\Screenshots\\cellscopy.png");
    if(!file.open(QIODevice::WriteOnly)){
        qDebug()<<"无法打开文件";
        return ;
    }
    file.write(fileByte);
    qDebug()<<"file writed!";
    file.close();
}

QByteArray vecpts2QByteArr(const std::vector<cv::Point2f>& points) {  
    QByteArray byteArray;  
    QDataStream dataStream(&byteArray, QIODevice::WriteOnly);  
    
    // 写入点的数量  
    int size = points.size();  
    dataStream << size;  

    // 写入每个 Point 的数据  
    for (const auto& point : points) {  
        dataStream << point.x << point.y;  
    }  
    return byteArray;  
}  

std::variant<cv::Point,std::string> QByteArr2cvPt(QByteArray& byteArray) {  
    // 确保 byteArray 至少包含 8 个字节（两个整型）  
    if (byteArray.size() < sizeof(int) * 2) {  
        return "ByteArray size is insufficient to convert to cv::Point.";
    }  
    // 从 QByteArray 中提取两个整型数  
    int x, y;  
    memcpy(&x, byteArray.data(), sizeof(int));  
    memcpy(&y, byteArray.data() + sizeof(int), sizeof(int));  

    return cv::Point(x, y);  
}

std::variant<cv::Mat, std::string> QByteArr2Mat(QByteArray& byteArray)
{   
    QImage image;
    if (!image.loadFromData(byteArray))
    {
        return "Failed to load QImage from QByteArray.";
    }
    // image.save("C:\\Users\\zydon\\Pictures\\Screenshots\\cellscopy.png");
    cv::Mat mat = QImage2cvMat(image);
    if (mat.empty())
    {
        return "The image to convert to cv::Mat is empty.";
    }

    // cv::namedWindow("QByteArr2Mat");
    // cv::imshow("QByteArr2Mat", mat);
    // cv::waitKey(0);

    // cv::Mat res = mat.clone();


    return mat;
    
}

cv::Mat QImage2cvMat(QImage image)
{
    cv::Mat mat;
    qDebug() << image.format();
    switch(image.format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
    {
        // 复制数据以避免数据指针失效问题
        QImage swapped = image.rgbSwapped();
        mat = cv::Mat(swapped.height(), swapped.width(), CV_8UC4, swapped.bits(), swapped.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);
        break;
    }
    case QImage::Format_RGB888:
    {
        // 复制数据以避免数据指针失效问题
        QImage swapped = image.rgbSwapped();
        mat = cv::Mat(swapped.height(), swapped.width(), CV_8UC3, swapped.bits(), swapped.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        break;
    }
    case QImage::Format_Indexed8:
    {
        // 复制数据以避免数据指针失效问题
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, image.bits(), image.bytesPerLine());
        break;
    }

    default:
        break;
    }
    //到这步没有问题


    cv::Mat res = mat.clone();
    return res;
}


/// @brief 从tcp信号推理结果
/// @return 是否成功
int ortsam2fortcp(uint port)
{
    /// 推理输入图片
    cv::Mat image;
    /// @brief 推理输入点
    cv::Point point;

#pragma region sam2模型初始化

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
        std::println("模型初始化失败错误：{}", error);
        return 1;
    }
    std::println("model intilize done!");
#pragma endregion

#pragma region 服务器初始化
    QString ip{"192.168.100.1"};
    TCPpkg::UnPack unpacktool;
    QTcpServer *server = new QTcpServer();
    server->listen(QHostAddress(ip), port);
    QTcpSocket *psocket = nullptr;

    if (server->isListening())
        qDebug() << "listening  on " << ip << ":" << port;
    else
    {
        qDebug() << "listening  failed: " << ip << ":" << port;
    }
#pragma endregion

    /// 有新链接
    QObject::connect(server, &QTcpServer::newConnection, [&]()
                     {
                         connectedNum++;
                         qDebug() << "current connectnum:" << connectedNum;

                         
                         psocket = server->nextPendingConnection();
                         auto socketpot = psocket->localPort();

                         psocket->write(QString::number(socketpot).toUtf8() + "linked");
                    

    /// 客户端失去连接
    QObject::connect(psocket, &QTcpSocket::disconnected, [&]()
                     {
            connectedNum--;
            qDebug()<<"current connectnum"<<connectedNum; 
            psocket->deleteLater(); });

    /// 客户端状态信号回复
    QObject::connect(psocket, &QTcpSocket::stateChanged, [&]()
                     {
            auto state =   psocket->state();
            switch (state) {

            case QAbstractSocket::ConnectingState:{
                psocket->write("connecting...");
            };break;

            case QAbstractSocket::ConnectedState:{
                psocket->write("connected!");
            };break;
            case QAbstractSocket::ClosingState:{
                psocket->write("closing!");
            };break;
            default:{};break;
            }
            psocket->write("socket will close later"); });

    QObject::connect(psocket, &QTcpSocket::readyRead, [&]()
    {
                         auto message = psocket->readAll();

                         QDataStream os(message);
                         auto orderbyte = message.mid(0, sizeof(char[7]));
                         auto caseidx = skorder.indexOf(orderbyte);
                         switch (caseidx)
                         {
                         /// 转换为接收图像模式
                         case static_cast<int>(SKORDER::RECEIVE_IMG):
                         {
                             qDebug() << "switch to receive image mode!";
                             curOrderMode = SKORDER::RECEIVE_IMG;
                             psocket->write("switch to receive image mode!");
                         };
                         break;
                         case static_cast<int>(SKORDER::RECEIVE_POINT):
                         {
                             qDebug() << "switch to receive point mode!";
                             curOrderMode = SKORDER::RECEIVE_POINT;
                             psocket->write("switch to receive point mode!");
                             resetReceived();
                         };
                         break;
                         case static_cast<int>(SKORDER::TIMEOUT):
                         {
                             qDebug() << "time out!";
                             // curOrderMode=SKORDER::TIMEOUT;
                             // todo 返回当前的状态
                         };
                         break;
                         case -1:
                         {
                             switch (curOrderMode)
                             {
                             case SKORDER::RECEIVE_IMG:
                             {
                                 QElapsedTimer timer;
                                 timer.start();
                                 qDebug() << "收到一条IMAGE信息";
                                 auto unpackedret = unpacktool(message);

                                 /// 输出
                                 if (!unpackedret.index())
                                 {
                                     qDebug() << "收到完整IMAGE图像";
                                     psocket->write("image received!");
                                    //  saveImage(std::get<QByteArray>(unpackedret));
                                     qint64 elapsed = timer.elapsed();
                                     qDebug() << "image reveice in" << elapsed << " milliseconds";
                                     auto unpackedbit=std::get<QByteArray>(unpackedret);
                                     auto resimage = QByteArr2Mat(unpackedbit);

                                     /// 图像转流错误
                                     if (resimage.index())
                                     {
                                        std::string error = std::get<std::string>(resimage);
                                         std::cout << error << std::endl;
                                         psocket->write(error.c_str());
                                         break;
                                     }
                                     image = std::get<cv::Mat>(resimage);
                                    // cv::namedWindow("image", cv::WINDOW_NORMAL);
                                    // cv::imshow("image", image);
                                    // cv::waitKey(0);
         
                                     isImageReceived = true;

                                     /// 5、加载图片
                                     // std::string image_path =std::string("C:\\Users\\zydon\\Desktop\\JH_pic\\12.5\\x\\left\\left0_20241205091330561.bmp");
                                     // cv::Mat image = cv::imread(image_path);
                                 }
                                 /// 图像解包失败
                                 else
                                 {
                                    ///暂时无法解包
                                    //  qDebug() << "图像解包失败";
                                    //  psocket->write("the image you send is unpacked failed!");
                                 }
                             };
                             break;
                             case SKORDER::RECEIVE_POINT:
                             {
                                 qDebug() << "收到一条POINT信息";
                                 auto respoint = QByteArr2cvPt(message);
                                 // 解析错误
                                 if (respoint.index())
                                 {
                                    auto str = std::get<std::string>(respoint);
                                     std::cout <<str << std::endl;
                                     psocket->write(str.c_str());
                                     break;
                                 }

                                 // 解析正确
                                 point = std::get<cv::Point>(respoint);

                                 /// 4、设置prompt
                                 sam2->setparms({
                                     .type = 1,
                                     .prompt_box = {745, 695, 145, 230},
                                     .prompt_point = point,
                                 });

                                 std::cout << "收到的point:" << point << std::endl;
                                 std::ostringstream ptos;
                                 ptos << point.x << point.y;
                                 psocket->write("point received!");

                                 isPointReceived = true;
                             };
                             break;
                             case SKORDER::TIMEOUT:
                             {
                                resetReceived();
                             };
                             break;
                             default:
                             {
                                 psocket->write("wrong order!");
                             };
                             break;
                             }
                         };
                         break;
                         }
                     });

   

    });
    // 满足条件时进行推理
    while (true)
    {
        QCoreApplication::processEvents();

        if (!(isPointReceived && isImageReceived))
        {
            continue;
        }
        cv::Mat colorImage;
        cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);
        
        // cv::Mat copyimage = cv::imread("C:\\Users\\zydon\\Pictures\\Screenshots\\cellscopy.png");
        /// 6、推理
        auto result = sam2->inference(colorImage);
        /// 成功推理
        if (result.index() == 0)
        {

            // 开始计时
            auto start = std::chrono::high_resolution_clock::now();

            auto pt = sam2->output_point;
            QByteArray ptbytearr =vecpts2QByteArr({pt});

            // psocket->write("result");
            // psocket->flush();
            // psocket->waitForBytesWritten();

            std::this_thread::sleep_for(std::chrono::milliseconds(200));


            // psocket->write(ptbytearr);
            // psocket->flush();
            // psocket->waitForBytesWritten();

            psocket->write(("result,"+std::to_string(pt.x)+","+std::to_string(pt.y)).c_str());
            psocket->flush();
            psocket->waitForBytesWritten();
            resetReceived();
            // 结束计时
            auto end = std::chrono::high_resolution_clock::now();
            // 计算耗时
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            // 输出耗时
            std::cout << "推理总耗时：" << duration << "ms" << std::endl;
        }
        /// 推理失败
        else
        {
            std::string error = std::get<std::string>(result);
            resetReceived();
            std::println("错误：{}", error);
            psocket->write("inference failed");
        }
    } 



    ///服务器关闭
    QObject::connect(server,&QTcpServer::destroyed,[&](){
        connectedNum=0;
        qDebug()<<"current connectnum"<<connectedNum;
    });




    //    QFile file("C:\\Users\\zydon\\Pictures\\Screenshots\\cells.png");
    //    if(file.open(QIODevice::ReadOnly))
    //    {
    //        QByteArray fileByte =file.readAll();
    //        auto fileBytepacked =TCPpkg::pack(fileByte);
    //        psocket.get()->write(fileBytepacked);
    //    }
 return 0;
}

int ortyolofortcp(int model_id){
        ///推理输入图片
cv::Mat image;
    std::unique_ptr<Yolov10>yolov10;
    std::unique_ptr<Yolov8SegOnnx>yolov8seg;
    /// yolo检测模型
    if (model_id == 2)
    {
#pragma region yolo模型初始化

        yolov10 = std::make_unique<Yolov10>();
        std::vector<std::string> onnx_paths{"D:\\m_code\\sam2_layout\\OrtInference-main\\models\\yolov10\\yolov10s.onnx"};
        auto r = yolov10->initialize(onnx_paths, true);
        if (r.index() != 0)
        {
            std::string error = std::get<std::string>(r);
            std::println("错误：{}", error);
            return 1;
        }
        yolov10->setparms({.score = 0.5f, .nms = 0.8f});

#pragma endregion
    }

    /// yolo分割模型
    else if (model_id == 3)
    {
        yolov8seg = std::make_unique<Yolov8SegOnnx>();
        std::string model_path_seg = "./models/yolov8s-seg.onnx";

        if (yolov8seg.get()->ReadModel(model_path_seg, true, 0, true))
        {
            std::cout << "yolov8seg loaded!" << std::endl;
        }
        else
        {
            std::cout << "yolov8seg loaded failed!" << std::endl;
            return 1;
        }
    }
    else{
    std::println("错误：{}","wrong model id");
    return 1;
}


#pragma region 服务器初始化
    QString ip{"127.0.0.1"};
    quint16 port = 8001;
    TCPpkg::UnPack unpacktool;
    QTcpServer *server = new QTcpServer();
    server->listen(QHostAddress(ip), port);
    QTcpSocket *psocket = nullptr;

    if (server->isListening())
        qDebug() << "listening  on " << ip << ":" << port;
    else
    {
        qDebug() << "listening  failed: " << ip << ":" << port;
    }
#pragma endregion


 /// 有新链接
    QObject::connect(server, &QTcpServer::newConnection, [&]()
                     {
                         connectedNum++;
                         qDebug() << "current connectnum:" << connectedNum;

                         
                         psocket = server->nextPendingConnection();
                         auto socketpot = psocket->localPort();

                         psocket->write(QString::number(socketpot).toUtf8() + "linked");
                    

    /// 客户端失去连接
    QObject::connect(psocket, &QTcpSocket::disconnected, [&]()
                     {
            connectedNum--;
            qDebug()<<"current connectnum:"<<connectedNum; 
            psocket->deleteLater(); });

    /// 客户端状态信号回复
    QObject::connect(psocket, &QTcpSocket::stateChanged, [&]()
                     {
            auto state =   psocket->state();
            switch (state) {

            case QAbstractSocket::ConnectingState:{
                psocket->write("connecting...");
            };break;

            case QAbstractSocket::ConnectedState:{
                psocket->write("connected!");
            };break;
            case QAbstractSocket::ClosingState:{
                psocket->write("closing!");
            };break;
            default:{};break;
            }
            psocket->write("socket will close later"); });

    QObject::connect(psocket, &QTcpSocket::readyRead, [&]()
    {
                         auto message = psocket->readAll();

                         QDataStream os(message);
                         auto orderbyte = message.mid(0, sizeof(char[7]));
                         auto caseidx = skorder.indexOf(orderbyte);
                         switch (caseidx)
                         {
                         /// 转换为接收图像模式
                         case static_cast<int>(SKORDER::RECEIVE_IMG):
                         {
                             qDebug() << "switch to receive image mode!";
                             curOrderMode = SKORDER::RECEIVE_IMG;
                             psocket->write("switch to receive image mode!");
                         };
                         break;
                         case static_cast<int>(SKORDER::TIMEOUT):
                         {
                             qDebug() << "time out!";
                             // curOrderMode=SKORDER::TIMEOUT;
                             // todo 返回当前的状态
                         };
                         break;
                         case -1:
                         {
                             switch (curOrderMode)
                             {
                             case SKORDER::RECEIVE_IMG:
                             {
                                 QElapsedTimer timer;
                                 timer.start();
                                 qDebug() << "收到一条IMAGE信息";
                                 auto unpackedret = unpacktool(message);

                                 /// 输出
                                 if (!unpackedret.index())
                                 {
                                     qDebug() << "收到完整IMAGE图像";
                                     saveImage(std::get<QByteArray>(unpackedret));
                                     qint64 elapsed = timer.elapsed();
                                     qDebug() << "image reveice in" << elapsed << " milliseconds";
                                     auto resimage = QByteArr2Mat(std::get<QByteArray>(unpackedret));

                                     /// 图像转流错误
                                     if (resimage.index())
                                     {
                                        std::string error = std::get<std::string>(resimage);
                                         std::cout << error << std::endl;
                                         psocket->write(error.c_str());
                                         break;
                                     }
                                     image = std::get<cv::Mat>(resimage);
                                     isImageReceived = true;

                                     /// 5、加载图片
                                     // std::string image_path =std::string("C:\\Users\\zydon\\Desktop\\JH_pic\\12.5\\x\\left\\left0_20241205091330561.bmp");
                                     // cv::Mat image = cv::imread(image_path);
                                 }
                                 /// 图像解包失败
                                 else
                                 {
                                     qDebug() << "图像解包失败";
                                     psocket->write("the image you send is unpacked failed!");
                                 }
                             };
                             break;                           
                             case SKORDER::TIMEOUT:
                             {
                             };
                             break;
                             default:
                             {
                                 psocket->write("wrong order!");
                             };
                             break;
                             }
                         };
                         break;
                         }
                     });

   

    });
    // 满足条件时进行推理
    while (true)
    {
        QCoreApplication::processEvents();

        if (!isImageReceived)
        {
            continue;
        }
    // 开始计时
            auto start = std::chrono::high_resolution_clock::now();

            if (model_id == 2)
            {
                // cv::namedWindow("image", cv::WINDOW_NORMAL);
                // cv::imshow("image", image);
                // cv::waitKey(0);
                /// 6、推理
                auto result = yolov10->inference(image);
                /// 成功推理
                if (result.index() == 0)
                {

                    cv::namedWindow("image", cv::WINDOW_NORMAL);
                    cv::imshow("image", image);
                    auto pts = yolov10->output_point;

                    QByteArray resos = vecpts2QByteArr(pts);
                    psocket->write(resos);

                    // 结束计时
                    auto end = std::chrono::high_resolution_clock::now();
                    // 计算耗时
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                    // 输出耗时
                    std::cout << "推理总耗时：" << duration << "ms" << std::endl;
                }
                /// 推理失败
                else
                {
                    std::string error = std::get<std::string>(result);

                    std::println("错误：{}", error);
                    psocket->write("inference failed!");
                }
                resetReceived();
            }
        // yolov8 onnxruntime segment
        else if (model_id == 3)
        {
            std::vector<OutputParams> outputs;
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
                        psocket->write(error.c_str());
                    }
                    else
                    {
                        out_points.push_back(std::get<cv::Point2f>(pt));
                    }
                }
                QByteArray resos = vecpts2QByteArr(out_points);
                psocket->write(resos);

                auto end = std::chrono::high_resolution_clock::now();
                // 计算耗时
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                // 输出耗时
                std::cout << "推理总耗时：" << duration << "ms" << std::endl;
            }
            /// 推理失败
            else
            {

                std::cout << "inference failed!" << std::endl;
                psocket->write("inference failed!");
            }
            resetReceived();

        }
    } 



    ///服务器关闭
    QObject::connect(server,&QTcpServer::destroyed,[&](){
        connectedNum=0;
        qDebug()<<"current connectnum:"<<connectedNum;
    });




    //    QFile file("C:\\Users\\zydon\\Pictures\\Screenshots\\cells.png");
    //    if(file.open(QIODevice::ReadOnly))
    //    {
    //        QByteArray fileByte =file.readAll();
    //        auto fileBytepacked =TCPpkg::pack(fileByte);
    //        psocket.get()->write(fileBytepacked);
    //    }
 return 0;
}


/// @brief 从tcp信号推理结果
/// @return 是否成功
int ortyolosam2fortcp(uint port)
{
    /// 推理输入图片
    cv::Mat image;


#pragma region sam2模型初始化

    /// 1、开辟对象
    auto sam2 = std::make_unique<SAM2>();
    /// 2、初始化模型参数路径
    std::vector<std::string> sam2onnx_paths{
        "D:/m_code/sam2_layout/OrtInference-main/models/sam2/large/image_encoder.onnx",
        "D:/m_code/sam2_layout/OrtInference-main/models/sam2/large/memory_attention.onnx",
        "D:/m_code/sam2_layout/OrtInference-main/models/sam2/large/image_decoder.onnx",
        "D:/m_code/sam2_layout/OrtInference-main/models/sam2/large/memory_encoder.onnx"};
    /// 3、初始化模型
    auto rsam = sam2->initialize(sam2onnx_paths, true);
    if (rsam.index() != 0)
    {
        std::string error = std::get<std::string>(rsam);
        std::println("模型初始化失败错误：{}", error);
        return 1;
    }
    std::println("sam intilize done!");
#pragma endregion

#pragma region yolo模型初始化

    /// 1、开辟对象
    auto yolov10 = std::make_unique<Yolov10>();
    /// 2、初始化模型参数路径
    std::vector<std::string> yoloonnx_paths{"D:\\m_code\\sam2_layout\\OrtInference-main\\models\\yolov10\\yolov10m_0117.onnx"};
    /// 3、初始化模型
    auto ryolo = yolov10->initialize(yoloonnx_paths, true);
    if (ryolo.index()!= 0)
    {
        std::string error = std::get<std::string>(ryolo);
        std::println("模型初始化失败错误：{}", error);
        return 1;
    }
    yolov10.get()->setparms({.score = 0.5f,.nms = 0.8f});
    std::println("yolo intilize done!");

#pragma endregion


#pragma region 服务器初始化
    QString ip{"192.168.100.1"};
    TCPpkg::UnPack unpacktool;
    QTcpServer *server = new QTcpServer();
    server->listen(QHostAddress(ip), port);
    QTcpSocket *psocket = nullptr;

    if (server->isListening())
        qDebug() << "listening  on " << ip << ":" << port;
    else
    {
        qDebug() << "listening  failed: " << ip << ":" << port;
    }
#pragma endregion

    /// 有新链接
    QObject::connect(server, &QTcpServer::newConnection, [&]()
                     {
                         connectedNum++;
                         qDebug() << "current connectnum:" << connectedNum;

                         
                         psocket = server->nextPendingConnection();
                         auto socketpot = psocket->localPort();

                         psocket->write(QString::number(socketpot).toUtf8() + "linked");
                    

    /// 客户端失去连接
    QObject::connect(psocket, &QTcpSocket::disconnected, [&]()
                     {
            connectedNum--;
            qDebug()<<"current connectnum"<<connectedNum; 
            psocket->deleteLater(); });


    QObject::connect(psocket, &QTcpSocket::readyRead, [&]()
    {
                         auto message = psocket->readAll();

                         QDataStream os(message);
                         auto orderbyte = message.mid(0, sizeof(char[7]));
                         auto caseidx = skorder.indexOf(orderbyte);
                         switch (caseidx)
                         {
                         /// 转换为接收图像模式
                         case static_cast<int>(SKORDER::RECEIVE_IMG):
                         {
                             qDebug() << "switch to receive image mode!";
                             curOrderMode = SKORDER::RECEIVE_IMG;
                             psocket->write("switch to receive image mode!");
                         };
                         break;
                         case static_cast<int>(SKORDER::TIMEOUT):
                         {
                             qDebug() << "time out!";
                             resetReceived();
                             psocket->write("reseted!");
                         };
                         break;
                         case -1:
                         {
                             switch (curOrderMode)
                             {
                             case SKORDER::RECEIVE_IMG:
                             {
                                 QElapsedTimer timer;
                                 timer.start();
                                 qDebug() << "收到一条IMAGE信息";
                                 auto unpackedret = unpacktool(message);

                                 /// 输出
                                 if (!unpackedret.index())
                                 {
                                     qDebug() << "收到完整IMAGE图像";
                                     psocket->write("image received!");
                                    //  saveImage(std::get<QByteArray>(unpackedret));
                                     qint64 elapsed = timer.elapsed();
                                     qDebug() << "image reveice in" << elapsed << " milliseconds";
                                     auto unpackedbit=std::get<QByteArray>(unpackedret);
                                     auto resimage = QByteArr2Mat(unpackedbit);

                                     /// 图像转流错误
                                     if (resimage.index())
                                     {
                                        std::string error = std::get<std::string>(resimage);
                                         std::cout << error << std::endl;
                                         psocket->write(error.c_str());
                                         break;
                                     }
                                     image = std::get<cv::Mat>(resimage);
                                     std::cout<<"成功收到并解析图像"<<"height:"<<image.rows<<"width:"<<image.cols<<std::endl;

                                     isImageReceived = true;
                                    
                                    auto yoloRes = yolov10->inference(image);
                                    if (yoloRes.index())
                                    {
                                        std::cout<<"yolo推理失败"<<std::endl;
                                        psocket->write("yolo inference failed!");
                                        break;
                                    }
                                    std::vector<cv::Rect> out_boxes = yolov10->boxes;
                                    std::vector<cv::Point2f> out_ceters;
                                    ///开始sam2推理
                                    for(auto &box:out_boxes){
                                        sam2->setparms({.type=0,.prompt_box=box});
                                        auto samRes = sam2->inference(image);
                                        if (samRes.index())
                                        {
                                            std::cout<<"sam2推理失败"<<std::endl;
                                            psocket->write("sam2 inference failed!");
                                            break;
                                        }
                                        out_ceters.push_back(sam2->output_point);
                                    }
                                    /// 发送字符串结果
                                    psocket->write("result,");
                                    psocket->flush();
                                    psocket->waitForBytesWritten();
                                    for(auto &pt:out_ceters){
                                        std::string resstr = std::to_string(pt.x)+","+std::to_string(pt.y)+";";
                                        psocket->write(resstr.c_str());
                                        psocket->flush();
                                        psocket->waitForBytesWritten();

                                    }
                                 }
                             };
                             break;
                             default:
                             {
                                 psocket->write("wrong order!");
                             };
                             break;
                             }
                         };
                         break;
                         }
                     });

   

    });

    ///服务器关闭
    QObject::connect(server,&QTcpServer::destroyed,[&](){
        connectedNum=0;
        qDebug()<<"current connectnum"<<connectedNum;
    });




    //    QFile file("C:\\Users\\zydon\\Pictures\\Screenshots\\cells.png");
    //    if(file.open(QIODevice::ReadOnly))
    //    {
    //        QByteArray fileByte =file.readAll();
    //        auto fileBytepacked =TCPpkg::pack(fileByte);
    //        psocket.get()->write(fileBytepacked);
    //    }
 return 0;
}
