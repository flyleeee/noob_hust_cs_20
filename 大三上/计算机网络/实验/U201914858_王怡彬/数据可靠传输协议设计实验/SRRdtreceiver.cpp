#include "stdafx.h"
#include "SRRdtReceiver.h"
#include "Global.h"

void SRRdtReceiver::Init()
{
	base = 0;
	//nextSeqnum = 0;
	for (int i = 0; i < seqsize; i++)
		bufStatus[i] = false;
	lastAckPkt.acknum = -1; //初始状态下，上次发送的确认包的确认序号为0，使得当第一个接受的数据包出错时该确认报文的确认号为0
	lastAckPkt.checksum = 0;
	lastAckPkt.seqnum = -1;	//忽略该字段
	memset(lastAckPkt.payload, '.', Configuration::PAYLOAD_SIZE);
	lastAckPkt.checksum = pUtils->calculateCheckSum(lastAckPkt);
}

void SRRdtReceiver::printSlideWindow()
{
	int i;
	for (i = 0; i < seqsize; i++)
	{
		if (i == base)
			std::cout << " [ ";
		if (isInWindow(i) && bufStatus[i] == true)
			std::cout << i << "*";
		else if (isInWindow(i))
			std::cout << i;
		if (i == (base + wndsize) % seqsize)
			std::cout << " ] ";
		if (isInWindow(i) == false)
			std::cout << i;
		std::cout << " ";
	}
	std::cout << std::endl;
}

bool SRRdtReceiver::isInWindow(int seqnum)
{
	//return seqnum >= base && seqnum <= (base + wndsize) % seqsize;
	//cout << seqnum << ':' << base << ':' << (base + wndsize) % seqsize << endl;
	if (base < (base + wndsize) % seqsize)
		return seqnum >= base && seqnum < (base + wndsize) % seqsize;
	else
		return seqnum >= base || seqnum < (base + wndsize) % seqsize;
}

SRRdtReceiver::SRRdtReceiver() :
	seqsize(8), wndsize(4), recvBuf(new Message[seqsize]), bufStatus(new bool[seqsize])
{
	Init();
}

SRRdtReceiver::SRRdtReceiver(int sSize, int wsize) :
	seqsize(sSize), wndsize(wsize), recvBuf(new Message[seqsize]), bufStatus(new bool[seqsize])
{
	Init();
}

void SRRdtReceiver::receive(const Packet& packet)
{
	int checksum = pUtils->calculateCheckSum(packet);
	if (checksum != packet.checksum)
	{
		//数据包损坏，不作出应答
		pUtils->printPacket("接收方没有正确收到发送方的报文,数据校验错误", packet);
		return;
	}
	else
	{
		if (isInWindow(packet.seqnum) == false)
		{
			//不是想要的数据包，不作出应答
			pUtils->printPacket("不是窗口内的分组，忽略", packet);
			lastAckPkt.acknum = packet.seqnum;
			lastAckPkt.seqnum = -1;
			memset(lastAckPkt.payload, '.', Configuration::PAYLOAD_SIZE);
			lastAckPkt.checksum = pUtils->calculateCheckSum(lastAckPkt);
			pns->sendToNetworkLayer(SENDER, lastAckPkt);
			return;
		}
		else
		{
			//是自己想要的数据包，发送ack，更新缓冲区和滑动窗口

			if (packet.seqnum == base) 
			{
				Message msg;
				memcpy(msg.data, packet.payload, sizeof(packet.payload));
				int flag = base;  
				for (int i = (base + 1) % seqsize, j = 1; j < wndsize; j++, i = (i + 1) % seqsize) 
				{
					if (bufStatus[i] == true) 
						flag = i;
					else 
						break;
				}
				if (flag == base) 
				{
					pns->delivertoAppLayer(RECEIVER, msg);
				}
				else 
				{
					pns->delivertoAppLayer(RECEIVER, msg);
					for (int i = (base + 1) % seqsize, j = 0; j < (flag - base + seqsize) % seqsize; j++, i = (i + 1) % seqsize) {
						pns->delivertoAppLayer(RECEIVER, recvBuf[i]);
						bufStatus[i] = false;
					}
				}
				base = (flag + 1) % seqsize;
				std::cout << "\n接收方窗口移动：";
				printSlideWindow();
				std::cout << std::endl;
			}
			else 
			{
				memcpy(recvBuf[packet.seqnum].data, packet.payload, sizeof(packet.payload));
				bufStatus[packet.seqnum] = true;
				printf("报文序号中断，接受方缓存报文序号%d，base=%d\n", packet.seqnum, base);
			}

			lastAckPkt.acknum = packet.seqnum; //确认序号等于收到的报文序号
			lastAckPkt.checksum = pUtils->calculateCheckSum(lastAckPkt);
			pUtils->printPacket("接收方发送确认报文", lastAckPkt);
			pns->sendToNetworkLayer(SENDER, lastAckPkt);	//调用模拟网络环境的sendToNetworkLayer，通过网络层发送确认报文到对方
		}
	}

}


SRRdtReceiver::~SRRdtReceiver()
{
	delete[] recvBuf;
	delete[] bufStatus;
}
