#include "stdafx.h"
#include "TcpRdtSender.h"
#include "Global.h"

void TcpRdtSender::Init()
{
	base = 0;
	nextSeqnum = 0;
	dupAckNum = 0;
}

bool TcpRdtSender::isInWindow(int seqnum)
{
	if (base < (base + wndsize) % seqsize)
	{
		return seqnum >= base && seqnum < (base + wndsize) % seqsize;
	}
	else
	{
		return seqnum >= base || seqnum < (base + wndsize) % seqsize;
	}
}

void TcpRdtSender::printSlideWindow()
{
	int i;
	for (i = 0; i < seqsize; i++)
	{
		if (i == base)
			std::cout << "[";
		std::cout << i;
		if (i == nextSeqnum)
			std::cout << "*";
		if (i == (base + wndsize - 1) % seqsize)
			std::cout << "]";
		std::cout << " ";
	}
	std::cout << std::endl;
}

TcpRdtSender::TcpRdtSender() :
	wndsize(4), seqsize(8), sendBuf(new Packet[seqsize])
{
	Init();
}

TcpRdtSender::TcpRdtSender(int wsize, int ssize) :
	wndsize(wsize), seqsize(ssize), sendBuf(new Packet[ssize])
{
	Init();
}


TcpRdtSender::~TcpRdtSender()
{
}

bool TcpRdtSender::getWaitingState()
{
	return (base + wndsize) % seqsize == (nextSeqnum) % seqsize;
}

bool TcpRdtSender::send(const Message& message)
{
	if (getWaitingState())
	{
		std::cout << "窗口已满\n\n";
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
			pns->startTimer(SENDER, Configuration::TIME_OUT, 0);
		pns->sendToNetworkLayer(RECEIVER, sendBuf[nextSeqnum]);
		//发送完毕，更新状态
		nextSeqnum = (nextSeqnum + 1) % seqsize;
		std::cout << "发送方发送后窗口：";
		printSlideWindow();
		std::cout << std::endl;
		return true;
	}
}

void TcpRdtSender::timeoutHandler(int seqNum)
{
	pns->sendToNetworkLayer(RECEIVER, sendBuf[base]);
	pns->stopTimer(SENDER, 0);
	pns->startTimer(SENDER, Configuration::TIME_OUT, 0);
	pUtils->printPacket("超时重传:", sendBuf[base]);
}

void TcpRdtSender::receive(const Packet& ackPkt)
{
	int checkSum = pUtils->calculateCheckSum(ackPkt);
	if (checkSum != ackPkt.checksum)
	{
		pUtils->printPacket("发送方没有正确收到确认", ackPkt);
		return;
	}
	else
	{
		//if (ackPkt.acknum >= base)
		if (isInWindow(ackPkt.acknum))
		{
			base = (ackPkt.acknum + 1) % seqsize;
			pns->stopTimer(SENDER, 0);
			if (base != nextSeqnum)
			{
				pns->startTimer(SENDER, Configuration::TIME_OUT, 0);
			}
			dupAckNum = 0;
			pUtils->printPacket("发送方正确收到确认", ackPkt);
			std::cout << "滑动窗口状态：";
			printSlideWindow();
			std::cout << std::endl;
		}
		else
		{//已经确认的冗余的ack
			dupAckNum = (dupAckNum + 1) % 3;
			if (dupAckNum == 2)
			{//快速重传
				pns->sendToNetworkLayer(RECEIVER, sendBuf[base]);
				std::cout << "\n收到连续三个冗余ack，快速重传\n\n";
			}
		}
	}
}
