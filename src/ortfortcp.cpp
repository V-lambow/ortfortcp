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

///推理输入图片
cv::Mat image;
/// @brief 推理输入点
cv::Point point;



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

cv::Mat QImage2cvMat(QImage image)
{
    cv::Mat mat;
    qDebug() << image.format();
    switch(image.format())
    {
    case QImage::Format_ARGB32:
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32_Premultiplied:
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
        break;
    case QImage::Format_RGB888:
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
        break;
    case QImage::Format_Indexed8:
        mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
        break;
    }
    return mat;
}

std::variant<cv::Point,std::string> QByteArr2cvPt(const QByteArray& byteArray) {  
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

std::variant<cv::Mat, std::string> QByteArr2Mat(const QByteArray &byteArray)
{   
    QImage image;
    if (!image.loadFromData(byteArray))
    {
        return "Failed to load QImage from QByteArray.";
    }
    cv::Mat mat = QImage2cvMat(image);
    if (mat.empty())
    {
        return "The image to convert to cv::Mat is empty.";
    }
    return mat;
    
}


/// @brief 从tcp信号推理结果
/// @return 是否成功
int ortsam2fortcp()
{

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
#pragma endregion

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
                                     cv::Mat image = std::get<cv::Mat>(resimage);
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

   
                                 /// 4、设置prompt
                                 sam2->setparms({
                                     .type = 1,
                                     .prompt_box = {745, 695, 145, 230},
                                     .prompt_point = point,
                                 });

                                //解析正确
                                 point = std::get<cv::Point>(respoint);
                                 std::cout<<"收到的point:" << point << std::endl;
                                 std::ostringstream ptos;
                                 ptos<<point.x<<point.y;
                                 psocket->write(ptos.str().c_str());


                                 isPointReceived = true;
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

        if (!(isPointReceived && isImageReceived))
        {
            continue;
        }

        /// 6、推理
        auto result = sam2->inference(image);
        /// 成功推理
        if (result.index() == 0)
        {

            // 开始计时
            auto start = std::chrono::high_resolution_clock::now();
            cv::namedWindow("image", cv::WINDOW_NORMAL);
            cv::imshow("image", image);
            auto pt = sam2->output_point;
            std::ostringstream ptos;
            ptos<<pt.x<<pt.y;
            psocket->write(ptos.str().c_str());
            

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
                                     cv::Mat image = std::get<cv::Mat>(resimage);
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
                    /// 6、推理
        auto result = yolov10->inference(image);
        /// 成功推理
        if (result.index() == 0)
        {

        
            cv::namedWindow("image", cv::WINDOW_NORMAL);
            cv::imshow("image", image);
            auto pts = yolov10->output_point;
            std::ostringstream ptos;
            for(auto pt:pts){
                ptos<<pt.x<<pt.y;
            }
            psocket->write(ptos.str().c_str());
            

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
            psocket->write("inference failed!");
        }
        }
        else if (model_id == 3)
        {
            std::vector<OutputParams> outputs;
            auto img = image.clone();
            yolov8seg.get()->OnnxDetect(img,outputs); // yolov8 onnxruntime segment

            resetReceived();
            // 结束计时
            auto end = std::chrono::high_resolution_clock::now();
            // 计算耗时
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            // 输出耗时
            std::cout << "推理总耗时：" << duration << "ms" << std::endl;
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


