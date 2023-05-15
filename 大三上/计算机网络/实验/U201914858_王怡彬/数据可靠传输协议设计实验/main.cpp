//// StopWait.cpp : 定义控制台应用程序的入口点。
////
#include "stdafx.h"
#include "Global.h"
#include "RdtSender.h"
#include "RdtReceiver.h"
#include "StopWaitRdtSender.h"
#include "StopWaitRdtReceiver.h"
#include "GBNRdtReceiver.h"
#include "GBNRdtSender.h"
#include "SRRdtReceiver.h"
#include "SRRdtSender.h"
#include "TcpRdtSender.h"
#include <algorithm>

using namespace std;

int main(int argc, char* argv[])
{
	int verbos = 1;
	//GBNRdtSender* ps = new GBNRdtSender();
	//GBNRdtReceiver* pr = new GBNRdtReceiver();
	//SRRdtSender* ps = new SRRdtSender();
	//SRRdtReceiver* pr = new SRRdtReceiver();
	TcpRdtSender* ps = new TcpRdtSender();
	GBNRdtReceiver* pr = new GBNRdtReceiver();


	pns->setRunMode(verbos);  //安静模式/VERBOS模式
	pns->init();
	pns->setRtdSender(ps);
	pns->setRtdReceiver(pr);
	pns->setInputFile("..\\input.txt");
	pns->setOutputFile("..\\output.txt");

	pns->start();

	delete ps;
	delete pr;
	delete pUtils;									//指向唯一的工具类实例，只在main函数结束前delete
	delete pns;										//指向唯一的模拟网络环境类实例，只在main函数结束前delete

	return 0;


}

