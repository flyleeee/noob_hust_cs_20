#define _CRT_SECURE_NO_WARNINGS
#define _WINSOCK_DEPRECATED_NO_WARNINGS

#include<stdio.h>
#include<Winsock2.h>
#include <iostream>
#include<string>
#include<unordered_map>
using namespace std;



#pragma comment(lib,"ws2_32.lib")


void check_error(int backdata, int errordata, string error_info);
void send_http(SOCKET s, string filepath, int* i);
string http_analysis(char* recvBuf);
string file_type_analysis(string arg);

void check_error(int backdata, int errordata, string error_info)
{
	if (backdata == errordata)
	{
		perror(error_info.c_str());
		WSAGetLastError();
		getchar();
		return;
	}
	return;
}
int main()
{
	string delimiter = "================================================";
	WSADATA wsaData;
	fd_set rfds;				//用于检查socket是否有数据到来的的文件描述符，用于socket非阻塞模式下等待网络事件通知（有数据到来）
	fd_set wfds;				//用于检查socket是否可以发送的文件描述符，用于socket非阻塞模式下等待网络事件通知（可以发送数据）
	bool first_connetion = true;
	int port;
	string inaddr;
	string main_directory;

	//初始化winsock80
	int nRc = WSAStartup(0x0202, &wsaData);//确定socket版本信息2.2，makeword做一个字
	check_error(nRc, WSAEINVAL, "Winsock init error");

	if (wsaData.wVersion != 0x0202) 
		cout << "Winsock version is not correct!\n";

	cout << "Winsock  startup Ok!\n";

	//printf("LTT服务器：\n");
	//create socket
	SOCKET ser_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	check_error(ser_socket, INVALID_SOCKET, "Socket create error");

	//set port and ip
	
	cout << "plz enter port:\n";
	cin >> port;
	cout << "plz enter inaddr:\n";
	cin >> inaddr;
	cout << "plz enter main_directory:\n";
	cin >> main_directory;

	struct sockaddr_in seraddr;
	seraddr.sin_family = AF_INET;//使用 Internet 地址
	seraddr.sin_port = htons(port);//htons函数把主机字节顺序转换为网络字节顺序，用于短整型
	seraddr.sin_addr.s_addr = inet_addr(inaddr.c_str());//net_addr 函数可以将字符串形式的 IP 地址转换为 unsigned long 形式的值。

	//binding
	int rtn = bind(ser_socket, (LPSOCKADDR)&seraddr, sizeof(seraddr));
	check_error(rtn, SOCKET_ERROR, "Socket bind error");

	//listen
	rtn = listen(ser_socket, 5);
	check_error(rtn, SOCKET_ERROR, "Socket listen error");

	cout << "web server start! ヾ(≧▽≦*)o" << endl;
	struct sockaddr_in claddr;
	int addrLen = sizeof(claddr);


	SOCKET* socket_list[10];
	int i;
	for(i = 0; i < 10; i++)
	{
		//等待连接并生成会话socket
		SOCKET sessionSocket = accept(ser_socket, (LPSOCKADDR)&claddr, &addrLen);
		check_error(sessionSocket, INVALID_SOCKET, "connect error");
		socket_list[i] = &sessionSocket;

		cout << "client ip: " << inet_ntoa(claddr.sin_addr) << endl << "client port: " << htons(claddr.sin_port) << endl;
		// receiving data from client
		char recvBuf[4096];
		memset(recvBuf, '\0', 4096);
		rtn = recv(sessionSocket, recvBuf, 2000, 0);
		check_error(rtn, SOCKET_ERROR, "receive data error");


		if (rtn > 0)
			cout << "Received " << rtn << " bytes from client:" << endl << delimiter << endl << recvBuf << delimiter << endl;
		else 
		{
			cout << "Client leaving ...\n";
			// 既然client离开了，就关闭sessionSocket
			break;
		}

		string filename;
		filename = http_analysis(recvBuf);
		
		string filepath = main_directory;
		filepath += filename;
		cout << "path:" << filepath << endl;
		send_http(sessionSocket, filepath, &i);

	}
	if (i == 10)
		cout << "socker list full! web server cannot deal with any more request! TAT" << endl;
	for (int j= 0; j < i; j++)
		closesocket(*socket_list[i]);
	
	closesocket(ser_socket);
	WSACleanup();

	getchar();
	return 0;
}

string http_analysis(char* recvBuf)
{
	int i = 0, j = 0;
	string name = "";
	while (recvBuf[i] != '/')
		i++;
	while (recvBuf[i + 1] != ' ')
	{
		name+= recvBuf[i + 1];
		i++;
		j++;
	}
	name+='\0';
	cout << "file name：" << name << endl;
	return name;
}

void send_http(SOCKET s, string filepath, int* i)
{
	string file_extension = file_type_analysis(filepath);

	string content_type = "text/plain";
	string body_length = "Content-Length: ";
	unordered_map<string, string> type_map = { {"html", "text/html"},{"gif", "image/gif"},{"jpg", "image/jpg"}, {"png", "image/png"} };

	content_type = type_map[file_extension.c_str()];
	string ok_head = "HTTP/1.1 200 OK\r\n"; 
	string not_acc_head = "HTTP/1.1 406 Not Acceptable\r\n";
	string not_found_head = "HTTP/1.1 404 NOT FOUND\r\n";
	string temp_1 = "Content-type: ";
	if (content_type == "")
	{
		cout << "406 Not Acceptable!" << endl;
		send(s, not_acc_head.c_str(), not_acc_head.length(), 0);
		closesocket(s);
		*i--;
		return;
	}
	FILE* pfile = fopen(filepath.c_str(), "rb");
	if (pfile == NULL)
	{
		cout << "404 not found!" << endl;
		send(s, not_found_head.c_str(), not_found_head.length(), 0);
		closesocket(s);
		*i--;
		return;
	}
	else if (send(s, ok_head.c_str(), ok_head.length(), 0) == -1)
	{
		cout << "Sending error" << endl;
		closesocket(s);
		*i--;
		return;
	}
	if (content_type.c_str())
	{
		temp_1 += content_type;
		temp_1 += "\r\n";

		if (send(s, temp_1.c_str(), temp_1.length(), 0) == -1)
		{
			cout << "Sending error!" << endl;
			closesocket(s);
			*i--;
			return;
		}
	}
	send(s, "\r\n", 2, 0);

	fseek(pfile, 0L, SEEK_END);
	int flen = ftell(pfile);
	char* p = (char*)malloc(flen + 1);
	fseek(pfile, 0L, SEEK_SET);
	fread(p, flen, 1, pfile);
	send(s, p, flen, 0);

	cout << endl << "file " << filepath << " sent successfully! o(￣▽￣)ｄ" << endl;
	return;

}

string file_type_analysis(string filepath)
{
	int pos = filepath.find_last_of('.');
	if (pos != -1)
		return filepath.substr(pos + 1);
	else
		return "";
}
