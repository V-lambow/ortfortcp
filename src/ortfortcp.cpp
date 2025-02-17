#include "ortfortcp.h"
#include "Model.h"
#include "SAM2.h"
#include "Yolov10.h"
#include "yolov8_seg_onnx.h"
#include <QRegExp>


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
    curOrderMode =SKORDER::UNDEFINED; 
}

bool isValidIp(const QString& ip) {
        QStringList segments = ip.split('.'); // 按 '.' 分割  
    if (segments.size() != 4) return false; // 必须有 4 个部分  

    for (const QString& segment : segments) {  
        bool ok;  
        int num = segment.toInt(&ok); // 转换为整数  

        // 验证这些条件  
        if (!ok || segment.isEmpty() || segment.length() > 3 || num < 0 || num > 255) {  
            return false;  
        }  

        // 检查前导零  
        if (segment.startsWith('0') && segment.length() > 1) {  
            return false; // 不能有前导零，如 01, 100  
        }  
    }  

    return true;  
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
        qDebug()<<"can not open file";
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
int ortsam2fortcp(QString ip, uint port)
{
    /// 推理输入图片
    cv::Mat image;
    /// @brief 推理输入点
    cv::Point point;

    std::string curObjName;

#pragma region sam2模型初始化

    /// 1、开辟对象
    auto sam2 = std::make_unique<SAM2>();
    /// 2、初始化模型参数路径
    std::vector<std::string> onnx_paths{
        "..//../models/sam2/large/image_encoder.onnx",
        "..//../models/sam2/large/memory_attention.onnx",
        "..//../models/sam2/large/image_decoder.onnx",
        "..//../models/sam2/large/memory_encoder.onnx"};
    /// 3、初始化模型
    auto r = sam2->initialize(onnx_paths, true);
    if (r.index() != 0)
    {
        std::string error = std::get<std::string>(r);
        std::println("model inti failed:{}", error);
        return 1;
    }
    std::println("model intilize done!");
#pragma endregion

#pragma region 服务器初始化
    // QString ip{"192.168.100.1"};
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
                         auto orderbyte = message.mid(0, 7);
                         auto caseidx = skorder.indexOf(orderbyte);
                         switch (caseidx)
                         {
                         /// 转换为接收图像模式
                         case static_cast<int>(SKORDER::RECEIVE_IMG):
                         {

                             curObjName = std::string(message.mid(7, 5).data());
                             qDebug() << "switch to receive image mode!";
                             curOrderMode = SKORDER::RECEIVE_IMG;
                             psocket->write("switch to receive image mode!");
                             unpacktool.clear();
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
                             resetReceived();
                             unpacktool.clear();
                             // todo 返回当前的状态
                         };
                         break;
                         case -1:
                         {
                             switch (curOrderMode)
                             {
                             case SKORDER::RECEIVE_IMG:
                             {
                               
                                auto startimgRec = std::chrono::high_resolution_clock::now();
                                

                                if (unpacktool.getStoredsize() == 0)
                                {
                                    startimgRec = std::chrono::high_resolution_clock::now();
                                }

                                //  qDebug() << "收到一条IMAGE信息";
                                 auto unpackedret = unpacktool(message);
                                
                                 /// 输出
                                 if (!unpackedret.index())
                                 {

                                     //  saveImage(std::get<QByteArray>(unpackedret));
                                     auto endimgRec = std::chrono::high_resolution_clock::now();
                                     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endimgRec - startimgRec).count();
                                     std::cout << "integrated Image received in " << duration << " ms." << std::endl;

                                     psocket->write("image received!");
                                     psocket->flush();
                                     psocket->waitForBytesWritten();

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
                                 qDebug() << "a POINT message received!";
                                 auto respoint = QByteArr2cvPt(message);
                                 // 解析错误
                                 if (respoint.index())
                                 {
                                    auto str = std::get<std::string>(respoint);
                                     std::cout <<str<< std::endl;
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

                                 std::cout << "point received:" << point << std::endl;
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
            // 开始计时
            auto start = std::chrono::high_resolution_clock::now();
        cv::Mat colorImage;
        cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);
        
        // cv::Mat copyimage = cv::imread("C:\\Users\\zydon\\Pictures\\Screenshots\\cellscopy.png");
        /// 6、推理
        auto result = sam2->inference(colorImage);
        /// 成功推理
        if (result.index() == 0)
        {

        

            auto pt = sam2->output_point;
            // QByteArray ptbytearr =vecpts2QByteArr({pt});

            // psocket->write("result");
            // psocket->flush();
            // psocket->waitForBytesWritten();
                       // 结束计时
            auto end = std::chrono::high_resolution_clock::now();
            // 计算耗时
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            // 输出耗时
            std::cout << "inference total duratiion:" << duration << "ms" << std::endl;

            if (duration < 500)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
            }

            // psocket->write(ptbytearr);
            // psocket->flush();
            // psocket->waitForBytesWritten();
            // psocket->write("result");
            // psocket->write(curObjName);
            psocket->write(("result"+curObjName+","+std::to_string(pt.x)+","+std::to_string(pt.y)).c_str());
            psocket->flush();
            psocket->waitForBytesWritten();
            resetReceived();
 
        }
        /// 推理失败
        else
        {
            std::string error = std::get<std::string>(result);
            resetReceived();
            std::println("error:{}", error);
            psocket->write(("inference failed"+error).c_str());
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

int ortyolofortcp(QString ip,uint portNumint ,int model_id){
    /// 推理输入图片
    cv::Mat image;
    std::unique_ptr<Yolov10> yolov10;
    std::unique_ptr<Yolov8SegOnnx> yolov8seg;
    std::map<std::string, uint> classesDef{{"masks", 0}, {"wafer", 1}};
    std::string curObjName;
    std::unique_ptr<CenterSearch> centerSearch_ptr = std::make_unique<CenterSearch>();
    std::chrono::steady_clock::time_point startimgRec ;
    /// 设置模式
    centerSearch_ptr->m_mode = CenterSearch::CenterMode::DBASE;

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
            std::println("error:{}", error);
            return 1;
        }
        yolov10->setparms({.score = 0.5f, .nms = 0.8f});

#pragma endregion
    }

    /// yolo分割模型
    else if (model_id == 3)
    {
        yolov8seg = std::make_unique<Yolov8SegOnnx>();
        std::string model_path_seg = "../../models/yolo11_seg/yolo11_seg0211.onnx";

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
    std::println("error:{}","wrong model id");
    return 1;
}


#pragma region 服务器初始化
    // QString ip{"127.0.0.1"};
    quint16 port = portNumint;
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
                         auto orderbyte = message.mid(0, 7);
                         auto caseidx = skorder.indexOf(orderbyte);
                         switch (caseidx)
                         {                          
                         /// 转换为接收图像模式
                         case static_cast<int>(SKORDER::RECEIVE_IMG):
                         {
                             curObjName = std::string(message.mid(7, 5).data());
                             qDebug() << "switch to receive image mode!";
                             curOrderMode = SKORDER::RECEIVE_IMG;
                             psocket->write("switch to receive image mode!");
                             unpacktool.clear();
                         };
                         break;
                         case static_cast<int>(SKORDER::TIMEOUT):
                         {
                             qDebug() << "time out!";
                             resetReceived();
                            
                         };
                         break;
                         case -1:
                         {
                             switch (curOrderMode)
                             {
                             case SKORDER::RECEIVE_IMG:
                             {
                                 

                                 if (unpacktool.getStoredsize() == 0)
                                 {
                                     startimgRec = std::chrono::high_resolution_clock::now();
                                 }
                                //  qDebug() << "收到一条IMAGE信息";
                                 auto unpackedret = unpacktool(message);

                                 /// 输出
                                 if (!unpackedret.index())
                                 {

                                     //  saveImage(std::get<QByteArray>(unpackedret));
                                     auto endimgRec = std::chrono::high_resolution_clock::now();
                                     auto imgRecDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endimgRec - startimgRec).count();
                                     std::cout << "integrated image received in: " << imgRecDuration << "ms" << std::endl;
                                     psocket->write("image received!");
                                     psocket->flush();
                                     psocket->waitForBytesWritten();

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
                                    //  qDebug() << "图像解包失败";
                                    //  psocket->write("the image you send is unpacked failed!");
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
                    std::cout << "total inference duration:" << duration << "ms" << std::endl;
                }
                /// 推理失败
                else
                {
                    std::string error = std::get<std::string>(result);

                    std::println("error :{}", error);
                    psocket->write(("inference failed!"+error).c_str());
                }
                resetReceived();
            }
        // yolov8 onnxruntime segment
        else if (model_id == 3)
        {
            std::vector<OutputParams> outputs{};
            cv::Mat colorImage;
            cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);
            /// 成功推理
            if (yolov8seg->OnnxDetect(colorImage, outputs))
            {
                cv::Point2f out_point{0.0f,0.0f};
                std::for_each(outputs.begin(), outputs.end(), [=, &out_point, &centerSearch_ptr](auto output)
                              {
                    //不是要找的目标
                    if(output.id!=classesDef.at(curObjName)){
                        return;
                    }
                    else{
                        auto ptfilter = getEdgePointsFromMask(output);
                        auto pt = centerSearch_ptr->contourCenter(myutil::cvpt2cvptf(ptfilter));
                        if (pt.index())
                        {
                            std::string error = std::get<std::string>(pt);
                            std::println("error: {}", error);
                            psocket->write(error.c_str());
                            psocket->flush();
                            psocket->waitForBytesWritten();
                        }
                        else
                        {
                            out_point = std::get<cv::Point2f>(pt);
                            psocket->write(("result" + curObjName + "," + std::to_string(out_point.x) + "," + std::to_string(out_point.y)).c_str());
                            psocket->flush();
                            psocket->waitForBytesWritten();
                            //不能break
                        }
                    } 
                    });
                    if(out_point.x==0.0f&&out_point.y==0.0f){
                        psocket->write(std::string("result"+curObjName+"not found!").c_str());
                        psocket->flush();
                        psocket->waitForBytesWritten();
                    }

                unpacktool.clear();

                auto end = std::chrono::high_resolution_clock::now();
                // 计算耗时
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                // 输出耗时
                std::cout << "total inference duration:" << duration << "ms" << std::endl;
            }
            /// 推理失败
            else
            {

                std::cout << "inference failed!" << std::endl;
                psocket->write("inference failed!");
                psocket->flush();
                psocket->waitForBytesWritten();
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
int ortyolosam2fortcp(QString ip, uint port)
{
    /// 推理输入图片
    cv::Mat image;
    std::map<std::string,uint>classesDef{{"masks",0},{"wafer",1}};
    std::string curObjName;

#pragma region sam2模型初始化

    /// 1、开辟对象
    auto sam2 = std::make_unique<SAM2>();
    /// 2、初始化模型参数路径
    std::vector<std::string> sam2onnx_paths{
        "../../models/sam2/large/image_encoder.onnx",
        "../../models/sam2/large/memory_attention.onnx",
        "../../models/sam2/large/image_decoder.onnx",
        "../../models/sam2/large/memory_encoder.onnx"};
    /// 3、初始化模型
    auto rsam = sam2->initialize(sam2onnx_paths, true);
    if (rsam.index() != 0)
    {
        std::string error = std::get<std::string>(rsam);
        std::println("sam intilize failed :{}", error);
        return 1;
    }
    std::println("sam intilize done!");
#pragma endregion

#pragma region yolo模型初始化

    /// 1、开辟对象
    auto yolov10 = std::make_unique<Yolov10>();
    /// 2、初始化模型参数路径
    std::vector<std::string> yoloonnx_paths{"..\\..\\models\\yolov10\\yolov10m_0117.onnx"};
    /// 3、初始化模型
    auto ryolo = yolov10->initialize(yoloonnx_paths, true);
    if (ryolo.index()!= 0)
    {
        std::string error = std::get<std::string>(ryolo);
        std::println("yolo intilize failed:{}", error);
        return 1;
    }
    yolov10.get()->setparms({.score = 0.5f,.nms = 0.8f});
    std::println("yolo intilize done!");

#pragma endregion


#pragma region 服务器初始化
    // QString ip{"127.0.0.1"};
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
                         auto orderbyte = message.mid(0, 7);
                         auto caseidx = skorder.indexOf(orderbyte);
                         switch (caseidx)
                         {                          
                         /// 转换为接收图像模式
                         case static_cast<int>(SKORDER::RECEIVE_IMG):
                         {
                             curObjName = std::string(message.mid(7, 5).data());
                             qDebug() << "switch to receive image mode!";
                             curOrderMode = SKORDER::RECEIVE_IMG;
                             psocket->write("switch to receive image mode!");
                             unpacktool.clear();
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
                                 auto startimgRec = std::chrono::high_resolution_clock::now();

                                 if (unpacktool.getStoredsize() == 0)
                                 {
                                     startimgRec = std::chrono::high_resolution_clock::now();
                                 }
                                //  qDebug() << "收到一条IMAGE信息";
                                 auto unpackedret = unpacktool(message);

                                 /// 输出
                                 if (!unpackedret.index())
                                 {

                                     //  saveImage(std::get<QByteArray>(unpackedret));
                                     auto endimgRec = std::chrono::high_resolution_clock::now();
                                     auto imgRecDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endimgRec - startimgRec).count();
                                     std::cout << "integrated image received in: " << imgRecDuration << " ms" << std::endl;
                                     psocket->write("image received!");
                                     psocket->flush();
                                     psocket->waitForBytesWritten();

                                     auto unpackedbit=std::get<QByteArray>(unpackedret);
                                     auto resimage = QByteArr2Mat(unpackedbit);


                                     /// 图像转流错误
                                     if (resimage.index())
                                     {
                                        std::string error = std::get<std::string>(resimage);
                                         std::cout << error << std::endl;
                                         psocket->write(error.c_str());

                                         //!error图像转流失败
                                         resetReceived();
                                         unpacktool.clear();
                                         break;
                                     }
                                     image = std::get<cv::Mat>(resimage);
                                     std::cout << "sucessed recive image: " << "height:" << image.rows << "width:" << image.cols << std::endl;

                                     isImageReceived = true;
                                     cv::Mat colorImage;
                                     cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);

                                     ///yolo计时
                                     auto start = std::chrono::high_resolution_clock::now();
                                     std::variant<bool,std::string> yoloRes = yolov10.get()->inference(colorImage);
                                     if (yoloRes.index())
                                     {
                                         std::cout << "yolo inference failed"+std::get<std::string>(yoloRes) << std::endl;
                                         psocket->write(("yolo inference failed:"+std::get<std::string>(yoloRes)).c_str());

                                         //!error yolo推理失败
                                         resetReceived();
                                         unpacktool.clear();
                                         break;
                                     }
                                     
                                     auto end = std::chrono::high_resolution_clock::now();
                                     // 计算耗时
                                     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                                     // 输出耗时
                                     std::cout << "yolo total duration:" << duration << "ms" << std::endl;
                                     std::vector<cv::Rect> out_boxes = yolov10->output_boxes;
                                     std::cout << "find obj:" << out_boxes.size() << "(num)" << std::endl;
                                     std::vector<cv::Point2f> out_ceters;
                                     /// sam2计时
                                     auto start2 = std::chrono::high_resolution_clock::now();
                                     /// 开始sam2推理

                                     //当前需要找的目标idx代号
                                    uint classesIdx =classesDef.at(curObjName);
                                    if (classesIdx>=classesDef.size()||classesIdx<0)
                                    {
                                        psocket->write("wrong order");
                                        psocket->flush();
                                        psocket->waitForBytesWritten();

                                        //!error 目标不存在已有的label表中
                                        resetReceived();
                                        unpacktool.clear();
                                        break;
                                    }

                                    //判断是否检测到目标
                                    std::vector<int> detectedLabels= yolov10->output_labels;
                                    auto detectedIdx =std::find(detectedLabels.begin(),detectedLabels.end(),classesIdx);
                                    if(detectedIdx == detectedLabels.end()){
                                        psocket->write("yolo not find the obj");
                                        psocket->flush();
                                        psocket->waitForBytesWritten();

                                        //!error yolo未检测到目标
                                        resetReceived();
                                        unpacktool.clear();
                                        break;
                                    }

                                    auto boxChosed = out_boxes[detectedIdx[0]];
                                    // 设置新矩形的宽和高
                                    int newWidth = 1440;
                                    int newHeight = 1440;

                                    // 定义 Lambda 表达式，生成新的 Rect
                                    auto ROI_corr = [&newWidth, &newHeight,&boxChosed]()
                                    {
                                        int centerX = boxChosed.x + boxChosed.width / 2;
                                        int centerY = boxChosed.y + boxChosed.height / 2;
                                        return cv::Rect(centerX - newWidth / 2, centerY - newHeight / 2, newWidth, newHeight);
                                    };
                                    cv::Rect croppedRect = ROI_corr();
                                    cv::Mat croppedImg = colorImage(croppedRect);



                                    sam2->setparms({.type = 0, .prompt_box = {boxChosed.x-croppedRect.x,boxChosed.y-croppedRect.y,boxChosed.width,boxChosed.height}});
                                    auto samRes = sam2->inference(croppedImg);
                                    if (samRes.index())
                                    {
                                        std::cout << "sam2 inference failed:" + std::get<std::string>(samRes) << std::endl;
                                        psocket->write(std::get<std::string>(samRes).c_str());

                                        //!error sam2推理失败
                                        resetReceived();
                                        unpacktool.clear();
                                        break;
                                    }

                                    auto end2 = std::chrono::high_resolution_clock::now();
                                    // 计算耗时
                                    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
                                    // 输出耗时
                                    std::cout << "sam inference duration:" << duration2 << "ms" << std::endl;
                                    /// 发送字符串结果
                                    psocket->write(("result"+curObjName+","+std::to_string(sam2->output_point.x+croppedRect.x)+","+std::to_string(sam2->output_point.y+croppedRect.y)).c_str());
                                    psocket->flush();
                                    psocket->waitForBytesWritten();

                                    resetReceived();
                                    unpacktool.clear();

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
    while(true){
        QCoreApplication::processEvents();

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


int ortyolodtsgfortcp(QString ip, uint port)
{
    /// 推理输入图片
    cv::Mat image;
    std::map<std::string,uint>classesDef{{"masks",0},{"wafer",1}};
    std::string curObjName;


#pragma region yolov10_detect模型初始化

    /// 1、开辟对象
    auto yolov10 = std::make_unique<Yolov10>();
    /// 2、初始化模型参数路径
    std::vector<std::string> yoloonnx_paths{"..\\..\\models\\yolov10\\yolov10m_0117.onnx"};
    /// 3、初始化模型
    auto ryolo = yolov10->initialize(yoloonnx_paths, true);
    if (ryolo.index()!= 0)
    {
        std::string error = std::get<std::string>(ryolo);
        std::println("yolo intilize failed:{}", error);
        return 1;
    }
    yolov10.get()->setparms({.score = 0.5f,.nms = 0.8f});
    std::println("yolo intilize done!");

#pragma endregion

#pragma region yolo11_seg检测模型初始化

    /// 1、开辟对象
    std::unique_ptr<Yolov8SegOnnx> yolov8seg;
    yolov8seg = std::make_unique<Yolov8SegOnnx>();
    std::string model_path_seg = "../../models/yolo11_seg/yolo11_seg0211.onnx";

    if (yolov8seg.get()->ReadModel(model_path_seg, true, 0, true))
    {
        std::cout << "yolov8seg loaded!" << std::endl;
    }
    else
    {
        std::cout << "yolov8seg loaded failed!" << std::endl;
        return 1;
    }

#pragma endregion

#pragma region 服务器初始化
    // QString ip{"127.0.0.1"};
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
                         auto orderbyte = message.mid(0, 7);
                         auto caseidx = skorder.indexOf(orderbyte);
                         switch (caseidx)
                         {                          
                         /// 转换为接收图像模式
                         case static_cast<int>(SKORDER::RECEIVE_IMG):
                         {
                             curObjName = std::string(message.mid(7, 5).data());
                             qDebug() << "switch to receive image mode!";
                             curOrderMode = SKORDER::RECEIVE_IMG;
                             psocket->write("switch to receive image mode!");
                             unpacktool.clear();
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
                                 auto startimgRec = std::chrono::high_resolution_clock::now();

                                 if (unpacktool.getStoredsize() == 0)
                                 {
                                     startimgRec = std::chrono::high_resolution_clock::now();
                                 }
                                //  qDebug() << "收到一条IMAGE信息";
                                 auto unpackedret = unpacktool(message);

                                 /// 输出
                                 if (!unpackedret.index())
                                 {

                                     //  saveImage(std::get<QByteArray>(unpackedret));
                                     auto endimgRec = std::chrono::high_resolution_clock::now();
                                     auto imgRecDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endimgRec - startimgRec).count();
                                     std::cout << "integrated image received in: " << imgRecDuration << " ms" << std::endl;
                                     psocket->write("image received!");
                                     psocket->flush();
                                     psocket->waitForBytesWritten();

                                     auto unpackedbit=std::get<QByteArray>(unpackedret);
                                     auto resimage = QByteArr2Mat(unpackedbit);


                                     /// 图像转流错误
                                     if (resimage.index())
                                     {
                                        std::string error = std::get<std::string>(resimage);
                                         std::cout << error << std::endl;
                                         psocket->write(error.c_str());

                                         //!error 图像转流错误
                                         resetReceived();
                                         unpacktool.clear();
                                         break;
                                     }
                                     image = std::get<cv::Mat>(resimage);
                                     std::cout << "sucessed recive image: " << "height:" << image.rows << "width:" << image.cols << std::endl;

                                     isImageReceived = true;
                                     cv::Mat colorImage;
                                     cv::cvtColor(image, colorImage, cv::COLOR_GRAY2BGR);

                                     ///yolo计时
                                     auto start = std::chrono::high_resolution_clock::now();
                                     std::variant<bool,std::string> yoloRes = yolov10.get()->inference(colorImage);
                                     if (yoloRes.index())
                                     {
                                         std::cout << "yolo inference failed"+std::get<std::string>(yoloRes) << std::endl;
                                         psocket->write(("yolo inference failed:"+std::get<std::string>(yoloRes)).c_str());

                                         //!error yolo推理失败
                                         resetReceived();
                                         unpacktool.clear();
                                         break;
                                     }
                                     
                                     auto end = std::chrono::high_resolution_clock::now();
                                     // 计算耗时
                                     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                                     // 输出耗时
                                     std::cout << "yolo total duration:" << duration << "ms" << std::endl;
                                     std::vector<cv::Rect> out_boxes = yolov10->output_boxes;
                                     std::cout << "find obj:" << out_boxes.size() << "(num)" << std::endl;
                                     std::vector<cv::Point2f> out_ceters;
                                     /// sam2计时
                                     auto start2 = std::chrono::high_resolution_clock::now();
                                     /// 开始sam2推理

                                     //当前需要找的目标idx代号
                                    uint classesIdx =classesDef.at(curObjName);
                                    if (classesIdx>=classesDef.size()||classesIdx<0)
                                    {
                                        psocket->write("wrong order");
                                        psocket->flush();
                                        psocket->waitForBytesWritten();

                                        //!error 在已设定的label表中未找到给出的label名
                                        resetReceived();
                                        unpacktool.clear();
                                        break;
                                    }

                                    //判断是否检测到目标
                                    std::vector<int> detectedLabels= yolov10->output_labels;
                                    auto detectedIdx =std::find(detectedLabels.begin(),detectedLabels.end(),classesIdx);
                                    if(detectedIdx == detectedLabels.end()){
                                        psocket->write("yolo not find the obj");
                                        psocket->flush();
                                        psocket->waitForBytesWritten();

                                        //!error yolov10未检测到给出的label名的目标
                                        resetReceived();
                                        unpacktool.clear();
                                        break;
                                    }

                                    auto boxChosed = out_boxes[detectedIdx[0]];
                                    // 设置新矩形的宽和高
                                    int newWidth = 1440;
                                    int newHeight = 1440;

                                    // 定义 Lambda 表达式，生成新的 Rect
                                    auto ROI_corr = [&newWidth, &newHeight,&boxChosed]()
                                    {
                                        int centerX = boxChosed.x + boxChosed.width / 2;
                                        int centerY = boxChosed.y + boxChosed.height / 2;
                                        return cv::Rect(centerX - newWidth / 2, centerY - newHeight / 2, newWidth, newHeight);
                                    };
                                    cv::Rect corppedRect = ROI_corr();

                                    //浅拷贝裁切图像
                                     cv::Mat croppedImg = colorImage(corppedRect);
                              
                                     /// 开始yolo11_seg推理
                                     std::vector<OutputParams> seg_outputs{};
                                     //输出
                                     cv::Point2f ret_center{};
                                     //是否成功
                                     bool isYolov8segSuccess = yolov8seg->OnnxDetect(croppedImg,seg_outputs);
                                     if(isYolov8segSuccess&&seg_outputs.size()>0)
                                     {
                                        //推理裁切图，就一个目标直接拿
                                        auto seg_output = seg_outputs[0];
                                        auto output_center =[=](){
                                            cv::Moments m = cv::moments(seg_output.boxMask, true);
                                            float cx = m.m10 / m.m00;
                                            float cy = m.m01 / m.m00;
                                            return cv::Point2f(cx,cy);
                                        };
                                        ret_center = {
                                            output_center().x + seg_output.box.x +  corppedRect.x,
                                            output_center().y + seg_output.box.y + corppedRect.y};

                                        // 输出结果
                                        auto end2 = std::chrono::high_resolution_clock::now();
                                        // 计算耗时
                                        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
                                        // 输出耗时
                                        std::cout << "yolo_seg inference duration:" << duration2 << "ms" << std::endl;
                                        /// 发送字符串结果
                                        psocket->write(("result" + curObjName + "," + std::to_string(ret_center.x) + "," + std::to_string(ret_center.y)).c_str());
                                        psocket->flush();
                                        psocket->waitForBytesWritten();
                                     }
                                     else if(isYolov8segSuccess&&seg_outputs.size()==0){
                                        std::cout << "yolo11_seg inference success but no obj!" << std::endl;
                                        psocket->write("yolo11_seg inference success but no obj!");
                                        psocket->flush();
                                        psocket->waitForBytesWritten();

                                        //!error yolo11_seg推理成功但未检测到目标
                                        resetReceived();
                                        unpacktool.clear();
                                        break;
                                     }
                                     else
                                     {
                                        std::cout << "yolo11_seg inference failed!" << std::endl;
                                        psocket->write("yolo11_seg inference failed!");
                                        psocket->flush();
                                        psocket->waitForBytesWritten();

                                        //!error yolo11_seg推理失败
                                        resetReceived();
                                        unpacktool.clear();
                                        break;
                                     }
                                     
                                     resetReceived();
                                     unpacktool.clear();
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
                         } });

   

    });
    while(true){
        QCoreApplication::processEvents();

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
