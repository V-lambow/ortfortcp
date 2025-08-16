#ifndef MAIN_SERVER_H_
#define MAIN_SERVER_H_
#include <iostream>
#include <string>
#include <QTcpSocket>
#include <QTcpServer>
#include <QDebug>
#include <QFile>
#include <QByteArray>
#include <QThread>
#include <QImage>
#include <variant>
#include <QTimer>
#include <QElapsedTimer>
#include <opencv2/opencv.hpp>

#include "Model.h"
#include "SAM2.h"
#include "Yolov10.h"
#include "yolov8_seg_onnx.h"
#include <QObject>
#include <QRegExp>

#include "tcp_package.hpp"
#include "yolov8_seg_onnx.h"
#include "myutil.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"
#include <QThreadPool>
#include <QPointer>

enum class SKORDER
{
    RECEIVE_IMG = 0,
    RECEIVE_POINT = 1,
    RESET = 2,
    UNDEFINED = 99
};

enum class DLTOOLKIT
{
  YOLOV10=0,
  YOLOSAM=1,
  YOLODS=2,
  YOLOPOSE=3  
};

static const QVector<QString> skorder = {"imagein", "pointin", "resetin"};
static const std::map<std::string, uint> classesDef{{"masks", 0}, {"wafer", 1}};

class MainServer : public QTcpServer
{
    Q_OBJECT
public:
    explicit MainServer(QObject *parent = nullptr);

    void startServer(QString ip, quint16 port);

protected:
    void incomingConnection(qintptr socketDescriptor) override;

private:
    QThreadPool m_threadPool;
};

class ClientHandler : public QObject
{
    Q_OBJECT
public:
    explicit ClientHandler(QTcpSocket *socket, QObject *parent = nullptr);
    ~ClientHandler();

public slots:
    void startProcessing();

signals:
    void processingFinished();
    void errorOccurred(const QString &error);

private slots:
    void handleClientData();
    void handleDisconnected();

private:
    // 成员函数
    bool initializeModels();
    void processImage_yolosam();
    void processImage_yolo(); 
    void cleanupResources();
    cv::Mat QImage2cvMat(QImage &&image);
    std::variant<cv::Mat, std::string> QByteArr2Mat(QByteArray &&byteArray);

    // 成员变量
    QTcpSocket *m_socket;
    std::unique_ptr<SAM2> m_sam;
    std::unique_ptr<Yolov10> m_yolo;
    TCPpkg::UnPack m_unpacker;
    SKORDER m_curOrder;
    std::string m_curObjName;
    cv::Mat m_image;
    DLTOOLKIT m_dltoolkit = DLTOOLKIT::YOLOV10;
};

class ClientTask : public QRunnable
{
public:
    explicit ClientTask(qintptr socketDescriptor) // 改为接收socket描述符
        : m_socketDescriptor(socketDescriptor)
    {
        setAutoDelete(true);
    }

    std::string getCurThreadId();

    void run() override
    {
        QThread *taskThread = QThread::currentThread();
        taskThread->setObjectName("ClientTaskThread_" + QString::number(m_socketDescriptor));

        QTcpSocket *socket = new QTcpSocket();
        socket->moveToThread(taskThread); // 绑定到当前线程

        if (!socket->setSocketDescriptor(m_socketDescriptor))
        {
            delete socket;
            return;
        }

        ClientHandler *handler = new ClientHandler(socket);
        handler->moveToThread(taskThread); // 确保 handler 在任务线程

        QEventLoop loop;
        QObject::connect(handler, &ClientHandler::processingFinished, &loop, &QEventLoop::quit);

        // 直接调用而非跨线程 invoke
        QMetaObject::invokeMethod(handler, &ClientHandler::startProcessing, Qt::DirectConnection);

        loop.exec();

        // 安全清理
        handler->deleteLater();
        socket->deleteLater();
    }

private:
    qintptr m_socketDescriptor;
    QTcpSocket *m_socket;
    std::string m_curThread_id;
};

#endif // MAIN_SERVER_H_