#ifndef TCP_PACKAGE_HPP
#define TCP_PACKAGE_HPP

#include <QByteArray>
#include <QTcpServer>
#include <QTcpSocket>
#include <variant>
#include <QFile>
#include <QCoreApplication>
#include <QDataStream>
#include <QString>

namespace TCPpkg {

const int static headerLength = sizeof(quint32);

///先判断是否达到串大小，
/// 1、如果小于串大小，则存储到stored
/// 2、如果加上stored超过串大小，则进行解析,直接丢弃超出部分
/// 拆包。入参为连续的数据，返回值是拆出来的所有包列表

class UnPack{
    QByteArray m_storedbyte;
    quint32 m_storedpackedsize;
public:
    std::variant<QByteArray,QString> operator()(const QByteArray& data)
    {
        ///新串
        if(m_storedbyte.isEmpty()){
            QDataStream inStream(data);
            quint32 packageLen = 0;
            inStream >> packageLen;
            //串完整
            if(static_cast<quint32>(data.size())>=packageLen+headerLength){
                clear();
                return data.mid(headerLength, data.size()-headerLength);
            }
            //串不完整
            else{
                m_storedbyte=data.mid(headerLength, data.size()-headerLength);
                m_storedpackedsize =packageLen;
                return QString{"串长小于数据包，存储当前数据"};
            }
        }
        ///有保留串
        else{
            //串完整
            if(static_cast<quint32>(data.size()+m_storedbyte.size())>=m_storedpackedsize){
                m_storedbyte.append(data.mid(0,m_storedpackedsize-m_storedbyte.size()));
                auto res =m_storedbyte;
                clear();
                return res;
            }
            //串不完整
            else{
                m_storedbyte.append(data);
                return QString{"串长小于数据包，存储当前数据"};
            }
        }

    }
    void clear(){
        m_storedbyte.clear();
        m_storedpackedsize=0;
    }

};




// 封包。 入参为数据，返回值是 数据前面加一个头. 这就是一个数据包了
static QByteArray pack(const QByteArray& data)
{
    QByteArray header(headerLength, 0);
    QDataStream os(&header, QIODevice::WriteOnly);
    os << static_cast<quint32>(data.length());
    return header + data;
}

}
#endif // TCP_PACKAGE_HPP
