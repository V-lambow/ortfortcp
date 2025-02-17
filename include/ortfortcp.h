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

#include "tcp_package.hpp"
#include "yolov8_seg_onnx.h"

void saveImage(const QByteArray &fileByte);
int ortsam2fortcp(QString ip,uint port);
int ortyolofortcp(QString ip,uint portNumint ,int model_id);
int ortyolosam2fortcp(QString ip, uint port);
void resetReceived();
bool isValidIp(const QString& ip);
cv::Mat QImage2cvMat(QImage image);
std::variant<cv::Point, std::string> QByteArr2cvPt(const QByteArray &byteArray);
std::variant<cv::Mat, std::string> QByteArr2Mat(const QByteArray &byteArray);
template<typename _Tp>
int yolov8_onnx(_Tp& task, cv::Mat& img, std::string& model_path);
std::vector<cv::Point2f>  keyptsFliter(std::vector<PoseKeyPoint> kpts,float conf_thres) noexcept;
int ortyolodtsgfortcp(QString ip, uint port);

#endif