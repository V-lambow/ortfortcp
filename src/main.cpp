#include <QCoreApplication>
#include "mainserver.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    try {
        MainServer server;

        QString ip = "127.0.0.1";
        quint16 port = 8001;
        server.startServer(ip, port);

        return a.exec();
    } catch (const std::exception& e) {
        qCritical() << "Server fatal error:" << e.what();
        return EXIT_FAILURE;
    }
}
