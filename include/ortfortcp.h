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

void saveImage(const QByteArray &fileByte);
int ortsam2fortcp();
int ortyolofortcp(int model_id);
void resetReceived();
cv::Mat QImage2cvMat(QImage image);
std::variant<cv::Point, std::string> QByteArr2cvPt(const QByteArray &byteArray);
std::variant<cv::Mat, std::string> QByteArr2Mat(const QByteArray &byteArray);
template<typename _Tp>
int yolov8_onnx(_Tp& task, cv::Mat& img, std::string& model_path);

#endif