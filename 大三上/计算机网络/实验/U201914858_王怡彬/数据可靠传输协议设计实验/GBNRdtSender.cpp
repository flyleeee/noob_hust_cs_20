#include "stdafx.h"
#include "GBNRdtSender.h"
#include "Global.h"

enum SeqStateEnum
{
	CONFIRMED,					//已被确认
	AVAILABLE,					//可用，还未发送
	SENT,						//发送，还未确认
	UNAVAILABLE					//不可用
};

//初始化滑动窗口状态
void GBNRdtSender::Init()
{
	base = 0;
	nextSeqnum = 0;
}

bool GBNRdtSender::isInWindow(int seqnum)
{
	if (base < (base + wndsize) % seqsize)
		return seqnum >= base && seqnum < (base + wndsize) % seqsize;
	else
		return seqnum >= base || seqnum < (base + wndsize) % seqsize;
}

GBNRdtSender::GBNRdtSender() :
	wndsize(4), seqsize(8), sendBuf(new Packet[8])
{
	Init();
}

GBNRdtSender::GBNRdtSender(int wsize, int sSize) :
	wndsize(wsize), seqsize(sSize), sendBuf(new Packet[sSize])
{
	Init();
}


GBNRdtSender::~GBNRdtSender()
{
	delete[] sendBuf;
}

//上层调用send函数发送数据
bool GBNRdtSender::send(const Message& message)
{
	if (getWaitingState())
	{//窗口已满，无法继续发送数据
		std::cout << " 窗口已满 \n\n";
		return false;
	}
	else
	{
		sendBuf[nextSeqnum].acknum = -1;
		sendBuf[nextSeqnum].seqnum = nextSeqnum;
		memcpy(sendBuf[nextSeqnum].payload, message.data, sizeof(message.data));
		sendBuf[nextSeqnum].checksum = pUtils->calculateCheckSum(sendBuf[nextSeqnum]);
		pUtils->printPacket("发送方发送报文", sendBuf[nextSeqnum]);
		if (base == nextSeqnum)
		{
			pns->startTimer(SENDER, Configuration::TIME_OUT, 0);
		}
		pns->sendToNetworkLayer(RECEIVER, sendBuf[nextSeqnum]);
		//发送完毕，更新状态
		nextSeqnum = (nextSeqnum + 1) % seqsize;
		//显示滑动窗口
		std::cout << "发送方发送后窗口：";
		printSlideWindow();
		std::cout << std::endl;
		return true;
	}
}

//返回是否处于等待状态，窗口满返回true，否则返回false
bool GBNRdtSender::getWaitingState()
{
	//根据实验文档，gbn协议中滑动窗口满则无法接收上层应用数据
	return (base + wndsize) %  this->seqsize == (nextSeqnum) % seqsize;
}

//接收ack
void GBNRdtSender::receive(const Packet& ackPkt)
{
	int checkSum = pUtils->calculateCheckSum(ackPkt);
	if (checkSum != ackPkt.checksum)
	{
		pUtils->printPacket("发送方没有正确收到确认", ackPkt);
	}
	else
	{
		base = (ackPkt.acknum + 1) % seqsize;//累积确认
		if (base == nextSeqnum)
		{
			pns->stopTimer(SENDER, 0);
		}
		else
		{
			//重启计时器
			pns->stopTimer(SENDER, 0);
			pns->startTimer(SENDER, Configuration::TIME_OUT, 0);
		}
		pUtils->printPacket("发送方正确收到确认", ackPkt);
		std::cout << "滑动窗口状态：";
		printSlideWindow();
		std::cout << std::endl;
	}
}

//处理超时
void GBNRdtSender::timeoutHandler(int seqNum)
{
	//重发所有已经发送且未确认的分组
	if (nextSeqnum == base)
	{
		//超时特例，不做处理
		return;
	}
	else
	{
		pns->startTimer(SENDER, Configuration::TIME_OUT, 0);//重启计时器，重新计时
		int i;
		for (i = base; i != nextSeqnum; i = (i + 1) % seqsize)
		{
			pns->sendToNetworkLayer(RECEIVER, sendBuf[i]);
			pUtils->printPacket("超时重传:", sendBuf[i]);
		}
	}
}

void GBNRdtSender::printSlideWindow()
{
	
	for (int i = 0; i < seqsize; i++)
	{
		if (i == base)
			std::cout << " [ ";
		std::cout << i;
		if (i == nextSeqnum)
			std::cout << "*";
		if (i == (base + wndsize - 1) % seqsize)
			std::cout << " ] ";
		std::cout << " ";
	}
	std::cout << std::endl;
}

