#ifndef _ortfortcp_h_
#define _ortfortcp_h_

#include <iostream>
#include <string>
#include <QCoreApplication>
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

void saveImage(const QByteArray &fileByte);
int ortsam2fortcp(QString ip,uint port);
int ortyolofortcp(QString ip,uint portNumint ,int model_id);
int ortyolosam2fortcp(QString ip, uint port);
void resetReceived() noexcept;
bool isValidIp(const QString& ip);
cv::Mat QImage2cvMat(QImage image);
std::variant<cv::Point, std::string> QByteArr2cvPt(const QByteArray &byteArray);
std::variant<cv::Mat, std::string> QByteArr2Mat(const QByteArray &byteArray);
template<typename _Tp>
int yolov8_onnx(_Tp& task, cv::Mat& img, std::string& model_path);
std::vector<cv::Point2f>  keyptsFliter(std::vector<PoseKeyPoint> kpts,float conf_thres) noexcept;
int ortyolodtsgfortcp(QString ip, uint port);

class Aa:public QObject{
    Q_OBJECT
};

//tcp封装线程
class ServerWorker : public QObject {
    Q_OBJECT
public:
    ServerWorker(QString ip, quint16 port, QObject* parent = nullptr) 
        : QObject(parent), m_ip(ip), m_port(port) {
            m_curThread_id = getCurThreadId();
        }
    ~ServerWorker()
    {
        if (m_server)
        {
            m_server->close();
            m_server->deleteLater();
        }
        if (m_psocket)
        {
            m_psocket->close();
            m_psocket->deleteLater();
        }
        
    }

public slots:
    bool start();

signals:
    void finished();

private slots:
    void handleNewConnection();
    std::string getCurThreadId();


private:
    QString m_ip;
    quint16 m_port;
    cv::Mat m_image;
    std::string m_curObjName;
    TCPpkg::UnPack m_unpacktool;
    QTcpServer *m_server = nullptr;
    QTcpSocket *m_psocket = nullptr;
    std::unique_ptr<SAM2>m_sam2;
    std::unique_ptr<Yolov10>m_yolov10;
    std::string m_curThread_id;

};
QThread* createServerThread(QString ip, quint16 port) ;

#endif