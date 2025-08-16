#include "mainserver.h"

MainServer::MainServer(QObject *parent)
    : QTcpServer(parent)
{
    m_threadPool.setMaxThreadCount(8); // 根据CPU核心数调整
}

void MainServer::startServer(QString ip, quint16 port)
{
    if (!listen(QHostAddress(ip), port))
    {
        throw std::runtime_error(errorString().toStdString());
    }
    qInfo() << "Server started on ip :  " << ip << "port : " << port;
}

void MainServer::incomingConnection(qintptr socketDescriptor)
{
    ClientTask *task = new ClientTask(socketDescriptor);
    m_threadPool.start(task);
}

// clienthandler.cpp
// 3. 修改 ClientHandler 构造函数
ClientHandler::ClientHandler(QTcpSocket *socket, QObject *parent)
    : QObject(nullptr), // 显式设置无父对象
      m_socket(socket)
{
    m_socket->moveToThread(this->thread());

    connect(m_socket, &QTcpSocket::readyRead,
            this, &ClientHandler::handleClientData,
            Qt::QueuedConnection); // 强制队列连接

    connect(m_socket, &QTcpSocket::disconnected,
            this, &ClientHandler::handleDisconnected,
            Qt::QueuedConnection);
}
// 4. 添加必要的析构处理
ClientHandler::~ClientHandler()
{
    if (m_socket)
    {
        // 确保断开所有信号连接
        m_socket->disconnect(this);

        // 安全终止连接
        if (m_socket->state() == QAbstractSocket::ConnectedState)
        {
            m_socket->abort(); // 立即终止连接
        }

        // 使用 deleteLater 安全释放
        m_socket->deleteLater();
        m_socket = nullptr;
    }
}

void ClientHandler::startProcessing()
{
    qInfo() << "Client handler started in thread:" << QThread::currentThreadId();

    // 确保模型已释放
    cleanupResources();

    if (!initializeModels())
    {
        emit errorOccurred("Model initialization failed");
        emit processingFinished();
        return;
    }

    // 连接客户端信号
    connect(m_socket, &QTcpSocket::readyRead,
            this, &ClientHandler::handleClientData);
    // connect(m_socket, &QTcpSocket::disconnected,
    //         this, &ClientHandler::handleDisconnected);
}

bool ClientHandler::initializeModels()
{
    try
    {
        switch (m_dltoolkit)
        {
        case DLTOOLKIT::YOLOV10:
        {
            m_yolo = std::make_unique<Yolov10>();
            std::vector<std::string> yoloPaths = {"..\\..\\models\\yolov10\\jh0311_dectectcircle.onnx"};
            if (auto ret = m_yolo->initialize(yoloPaths, true); ret.index() != 0)
            {
                throw std::runtime_error("YOLO init: " + std::get<std::string>(ret));
            }
        }

        break;
        case DLTOOLKIT::YOLOSAM:
        {
            // SAM模型初始化
            m_sam = std::make_unique<SAM2>();
            std::vector<std::string> samPaths = {
                "../../models/sam2/large/image_encoder.onnx",
                "../../models/sam2/large/memory_attention.onnx",
                "../../models/sam2/large/image_decoder.onnx",
                "../../models/sam2/large/memory_encoder.onnx"};
            if (auto ret = m_sam->initialize(samPaths, true); ret.index() != 0)
            {
                throw std::runtime_error("SAM init: " + std::get<std::string>(ret));
            }

            // YOLO模型初始化
            m_yolo = std::make_unique<Yolov10>();
            std::vector<std::string> yoloPaths = {"..\\..\\models\\yolov10\\yolov10m_0226.onnx"};
            if (auto ret = m_yolo->initialize(yoloPaths, true); ret.index() != 0)
            {
                throw std::runtime_error("YOLO init: " + std::get<std::string>(ret));
            }
        }
        break;

        default:
        {
            throw std::runtime_error("Unsupported DLTOOLKIT");
        }
        break;
        }
        //* 模型加载成功
        qInfo() << "Model init all success";
        m_socket->write("Model init all success");
        m_socket->flush();
        m_socket->waitForBytesWritten();

        return true;
    }
    catch (const std::exception &e)
    {
        qCritical() << "Model init failed:" << e.what();
        return false;
    }
}

void ClientHandler::handleClientData()
{
    if (!m_socket || m_socket->state() != QAbstractSocket::ConnectedState)
    {
        return; // 提前终止无效操作
    }
    while (m_socket->bytesAvailable() > 0)
    {
        QByteArray message = m_socket->readAll();
        // 处理接收到的数据
        auto orderbyte = message.mid(0, 7);
        auto caseidx = skorder.indexOf(orderbyte);

        switch (caseidx)
        {

        /// 转换为接收图像模式
        case static_cast<int>(SKORDER::RECEIVE_IMG):
        {
            m_curObjName = std::string(message.mid(7, 5).data());

            m_curOrder = SKORDER::RECEIVE_IMG;
            qDebug() << "switch to receive image mode!";
            m_socket->write("switch to receive image mode!");

            m_unpacker.clear();
            // 如果指令后面有尾随数据，则加入解码
            if (message.size() > 13)
            {
                m_unpacker(message.mid(12));
            }
        };
        break;
        case -1:
        {
            switch (m_curOrder)
            {
            case SKORDER::RECEIVE_IMG:
            {
                auto startimgRec = std::chrono::high_resolution_clock::now();

                if (m_unpacker.getStoredsize() == 0)
                {
                    startimgRec = std::chrono::high_resolution_clock::now();
                }
                //  qDebug() << "收到一条IMAGE信息";
                auto unpackedret = m_unpacker(message);

                /// 输出
                if (!unpackedret.index())
                {

                    //  saveImage(std::get<QByteArray>(unpackedret));
                    auto endimgRec = std::chrono::high_resolution_clock::now();
                    auto imgRecDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endimgRec - startimgRec).count();
                    std::cout << "integrated image received in: " << imgRecDuration << " ms" << std::endl;
                    //  logger->info(m_curThread_id + "image received!");
                    m_socket->write("image received!");
                    m_socket->flush();
                    m_socket->waitForBytesWritten();

                    auto unpackedbit = std::get<QByteArray>(unpackedret);
                    auto resimage = ClientHandler::QByteArr2Mat(std::move(unpackedbit));

                    /// 图像转流错误
                    if (resimage.index())
                    {
                        std::string error = std::get<std::string>(resimage);
                        std::cout << error << std::endl;
                        m_socket->write(error.c_str());

                        //! error图像转流失败
                        m_curOrder = SKORDER::UNDEFINED;
                        m_unpacker.clear();
                        break;
                    }
                    auto mono_image = std::get<cv::Mat>(resimage);

                    cv::Mat colorImage;
                    cv::cvtColor(std::move(mono_image), m_image, cv::COLOR_GRAY2BGR);
                    std::cout << "sucessed recive image: " << "height:" << m_image.rows << "; width:" << m_image.cols << std::endl;

                    // 开始推理
                    switch (m_dltoolkit)
                    {
                    case DLTOOLKIT::YOLOV10:
                    {
                        processImage_yolo(); 
                    }
                    break;
                    case DLTOOLKIT::YOLOSAM:
                    {
                        processImage_yolosam();
                    }
                    break;
                    default:
                    {
                        m_socket->write("unsupported dltoolkit");
                        m_socket->flush();
                        m_socket->waitForBytesWritten();

                        //! error 不支持的dltoolkit
                        m_curOrder = SKORDER::UNDEFINED;
                        m_unpacker.clear();                       
                    }
                    break;
                    }
                }
            }
            }
        }
        }
    }
}
void ClientHandler::processImage_yolosam()
{
    try
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::variant<bool, std::string> yoloRes = m_yolo->inference(m_image);
        if (yoloRes.index())
        {
            std::cout << "yolo inference failed" + std::get<std::string>(yoloRes) << std::endl;
            m_socket->write(("yolo inference failed:" + std::get<std::string>(yoloRes)).c_str());

            //! error yolo推理失败
            m_curOrder = SKORDER::UNDEFINED;
            m_unpacker.clear();
            return;
        }

        auto end = std::chrono::high_resolution_clock::now();
        // 计算耗时
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // 输出耗时
        std::cout << "yolo total duration:" << duration << "ms" << std::endl;
        std::vector<cv::Rect> out_boxes = m_yolo->output_boxes;
        std::cout << "find obj:" << out_boxes.size() << "(num)" << std::endl;
        std::vector<cv::Point2f> out_ceters;
        /// sam2计时
        auto start2 = std::chrono::high_resolution_clock::now();
        /// 开始sam2推理

        // 将名称转为代号
        auto classesIdx = classesDef.find(m_curObjName);
        if (classesIdx == classesDef.end())
        {
            m_socket->write("wrong label name");
            m_socket->flush();
            m_socket->waitForBytesWritten();

            //! error 目标不存在已有的label表中
            m_curOrder = SKORDER::UNDEFINED;
            m_unpacker.clear();
            return;
        }

        // 判断是否检测到目标
        std::vector<int> detectedLabels = m_yolo->output_labels;
        auto detectedIdx = std::find(detectedLabels.begin(), detectedLabels.end(), classesIdx->second);
        if (detectedIdx == detectedLabels.end())
        {
            m_socket->write("yolo not find the obj");
            m_socket->flush();
            m_socket->waitForBytesWritten();

            //! error yolo未检测到目标
            m_curOrder = SKORDER::UNDEFINED;
            m_unpacker.clear();
            return;
        }
        auto boxChosed = out_boxes[detectedIdx - detectedLabels.begin()];
        // 设置新矩形的宽和高
        int newWidth = 1440;
        int newHeight = 1440;

        // 定义 Lambda 表达式，生成新的 Rect
        auto croppedRect = [=]()
        {
            int centerX = boxChosed.x + boxChosed.width / 2;
            int centerY = boxChosed.y + boxChosed.height / 2;
            return cv::Rect(centerX - newWidth / 2, centerY - newHeight / 2, newWidth, newHeight);
        }();
        // cv::Mat croppedImg = colorImage(croppedRect);
        cv::Mat croppedImg = myutil::safeCrop(m_image, croppedRect);

        m_sam->setparms({.is_mem_attention = false, .type = 0, .prompt_box = {boxChosed.x - croppedRect.x, boxChosed.y - croppedRect.y, boxChosed.width, boxChosed.height}});
        auto samRes = m_sam->inference(croppedImg);
        if (samRes.index())
        {
            std::cout << "sam2 inference failed:" + std::get<std::string>(samRes) << std::endl;
            m_socket->write(std::get<std::string>(samRes).c_str());

            //! error sam2推理失败
            m_curOrder = SKORDER::UNDEFINED;
            m_unpacker.clear();
            return;
        }

        auto end2 = std::chrono::high_resolution_clock::now();
        // 计算耗时
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
        // 输出耗时
        std::cout << "sam inference duration:" << duration2 << "ms" << std::endl;
        /// 发送字符串结果
        m_socket->write(("result" + m_curObjName + "," + std::to_string(m_sam->output_point.x + croppedRect.x) + "," + std::to_string(m_sam->output_point.y + croppedRect.y)).c_str());
        m_socket->flush();
        m_socket->waitForBytesWritten();
        std::cout << "send point2f:" << m_sam->output_point.x + croppedRect.x << "," << m_sam->output_point.y + croppedRect.y << std::endl;
        //* 发送图片结果
        m_curOrder = SKORDER::UNDEFINED;
        m_unpacker.clear();
    }
    catch (const std::exception &e)
    {
        m_socket->write("ERROR: " + QByteArray(e.what()));
        m_socket->flush();
        m_socket->waitForBytesWritten();
        qCritical() << "Error processing image:" << e.what();
    }
    return;
}

void ClientHandler::handleDisconnected()
{
    qInfo() << "Client disconnected:" << m_socket->peerAddress().toString() << "; port:" << m_socket->peerPort();
    cleanupResources();
    emit processingFinished();
}

void ClientHandler::cleanupResources()
{
    // 清理模型资源
    m_sam.reset(nullptr);
    m_yolo.reset(nullptr);

    // 确保析构顺序
    m_sam = nullptr;
    m_yolo = nullptr;
}

std::string ClientTask::getCurThreadId()
{
    std::stringstream ss;
    ss << std::this_thread::get_id();
    return ss.str();
}
cv::Mat ClientHandler::QImage2cvMat(QImage &&image)
{

    cv::Mat mat{};

    // qDebug() << image.format();

    switch (image.format())
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
    return mat.clone();
}
std::variant<cv::Mat, std::string> ClientHandler::QByteArr2Mat(QByteArray &&byteArray)

{
    QImage image;
    if (!image.loadFromData(byteArray))
    {
        return "Failed to load QImage from QByteArray.";
    }
    // image.save("C:\\Users\\zydon\\Pictures\\Screenshots\\cellscopy.png");
    cv::Mat mat = ClientHandler::QImage2cvMat(std::move(image));
    if (mat.empty())
    {
        return "The image to convert to cv::Mat is empty.";
    }

    return mat;
}


void ClientHandler::processImage_yolo(){
    auto start = std::chrono::high_resolution_clock::now();
         auto result = m_yolo->inference(m_image);
            /// 成功推理
            if (result.index() == 0)
            {

                // cv::namedWindow("image", cv::WINDOW_NORMAL);
                // cv::imshow("image", image);
                // cv::waitKey(0);
                auto pts = m_yolo->output_point;
                //! 没找到目标
                if (pts.size() <= 0)
                {
                    m_socket->write("wafer not found!");
                    m_socket->flush();
                    m_socket->waitForBytesWritten();                  
                    m_curOrder = SKORDER::UNDEFINED;
                    m_unpacker.clear();
                    return;
                }

                auto wafercenter = pts[0];
                m_socket->write(("resultwafer," + std::to_string(wafercenter.x) + "," +
                                std::to_string(wafercenter.y))
                                   .c_str());
                m_socket->flush();
                m_socket->waitForBytesWritten();

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
                m_socket->write(("inference failed!" + error).c_str());
                m_socket->flush();
                m_socket->waitForBytesWritten();
            }
            m_curOrder = SKORDER::UNDEFINED;
            m_unpacker.clear();
            return;
            
}