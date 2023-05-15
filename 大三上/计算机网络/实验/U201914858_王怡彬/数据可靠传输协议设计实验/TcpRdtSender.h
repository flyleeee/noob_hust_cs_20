#pragma once
#include "RdtSender.h"
class TcpRdtSender :
	public RdtSender
{
private:
	//bool waitingState;
	int base;							//基序号
	int nextSeqnum;						//下一个待发分组的序号
	const int wndsize;					//滑动窗口大小
	const int seqsize;					//序号大小
	Packet* const sendBuf;				//发送缓冲区
	int dupAckNum;							//收到3个冗余ack快速重传

private:
	void Init();
	bool isInWindow(int seqnum);
	void printSlideWindow();

public:
	TcpRdtSender();
	TcpRdtSender(int wsize, int ssize);
	virtual ~TcpRdtSender();

public:
	bool getWaitingState();
	bool send(const Message& message);
	void timeoutHandler(int seqNum);
	void receive(const Packet& ackPkt);
};

